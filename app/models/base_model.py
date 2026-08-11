"""
Strategy interface for all detection / segmentation models.

This module defines the contract that every concrete model (YOLOv8, Detectron2,
Grounding DINO, SAM) must satisfy so that the service layer can operate on any
model through a single abstraction:

    BaseDetectionModel
        .load()                     - lazy weight loading
        .is_loaded                  - readiness signal
        .infer(image, **kwargs)     - normalized inference -> InferenceResult
        .predict(image, **kwargs)   - raw model output (legacy dict shape)
        .get_model_info()           - health-check metadata

Two data classes provide the strongly-typed transport between layers:

    Detection        - one detected object (label, confidence, bbox, mask)
    InferenceResult  - the full response from a single inference call
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


# ─── Domain dataclasses ────────────────────────────────────────────────────


@dataclass
class Detection:
    """A single detected object.

    ``bbox`` is stored as [x1, y1, x2, y2] in pixel coordinates.
    ``mask`` is optional and, when present, is the segmentation mask shape
    (H x W) as a list-of-lists for JSON serialization.
    """

    id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    class_id: Optional[int] = None
    mask_shape: Optional[Tuple[int, int]] = None
    mask_rle: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        if self.mask_shape is not None:
            d["mask_shape"] = list(self.mask_shape)
        return d


@dataclass
class InferenceResult:
    """Uniform result returned by every model.

    ``raw`` preserves the model-specific dict shape so downstream callers can
    still access model-native fields when they need them.
    """

    model_name: str
    detections: List[Detection]
    inference_time_ms: float
    image_shape: Tuple[int, int]
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
            "image_shape": list(self.image_shape),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "cached": self.cached,
            "metadata": self.metadata,
        }


# ─── Strategy base class ───────────────────────────────────────────────────


class BaseDetectionModel(ABC):
    """Strategy contract shared by every detection / segmentation model.

    Concrete subclasses implement two things:

    * ``load_model()`` - populate ``self.model`` with real weights.
    * ``predict(image, **kwargs)`` - run inference and return a dict in the
      model's native shape.  The default ``infer()`` implementation normalizes
      that dict into an :class:`InferenceResult`.  Subclasses may override
      ``infer()`` when the model has special-purpose outputs (e.g. SAM's
      automatic mask generator emits segments rather than detections).
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model: Any = None
        self._is_loaded = False
        logger.info("Initialized %s (weights not yet in memory)", model_name)

    # -- Lifecycle -----------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory. Called on first inference."""

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def ensure_loaded(self) -> None:
        if not self._is_loaded:
            logger.info("Lazy-loading %s ...", self.model_name)
            self.load_model()
            self._is_loaded = True
            logger.info("%s ready", self.model_name)

    # -- Inference -----------------------------------------------------------

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Run inference and return the model's native dict output."""

    def infer(self, image: np.ndarray, **kwargs: Any) -> InferenceResult:
        """Normalized inference entry point.

        Wraps :meth:`predict` with timing, ensures the model is loaded, and
        converts the result into a strongly typed :class:`InferenceResult`.
        Subclasses can override for custom translation logic.
        """
        self.ensure_loaded()
        started = time.perf_counter()
        raw = self.predict(image, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = self._raw_to_detections(raw)
        return InferenceResult(
            model_name=raw.get("model", self.model_name),
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_shape=tuple(raw.get("image_shape", image.shape[:2])),
            metadata=raw.get("metadata", {}),
            raw=raw,
        )

    # -- Introspection -------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "type": self.__class__.__name__,
        }

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _raw_to_detections(raw: Dict[str, Any]) -> List[Detection]:
        """Best-effort conversion from a model's raw dict to Detection objects.

        Handles the two shapes emitted by our current concrete models:
        ``{"detections": [...]}`` and ``{"segments": [...]}``.
        """
        items: List[Dict[str, Any]] = raw.get("detections") or raw.get("segments") or []
        out: List[Detection] = []
        for i, item in enumerate(items):
            bbox = item.get("bbox") or item.get("input_box") or [0, 0, 0, 0]
            bbox = tuple(float(v) for v in bbox[:4]) if len(bbox) >= 4 else (0.0, 0.0, 0.0, 0.0)
            out.append(
                Detection(
                    id=item.get("id", i),
                    label=(
                        item.get("class_name")
                        or item.get("label")
                        or item.get("name")
                        or "object"
                    ),
                    confidence=float(
                        item.get("confidence")
                        or item.get("score")
                        or item.get("predicted_iou")
                        or 0.0
                    ),
                    bbox=bbox,
                    class_id=item.get("class_id"),
                    mask_shape=tuple(item["mask_shape"]) if item.get("mask_shape") else None,
                    mask_rle=item.get("mask_rle"),
                    extra={
                        k: v
                        for k, v in item.items()
                        if k
                        not in {
                            "id",
                            "label",
                            "class_name",
                            "class_id",
                            "confidence",
                            "score",
                            "bbox",
                            "input_box",
                            "mask_shape",
                            "mask_rle",
                        }
                    },
                )
            )
        return out
