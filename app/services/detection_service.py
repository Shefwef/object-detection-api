"""
Detection service - the single entry point routers call.

Responsibilities:
    1. Resolve the requested :class:`ModelType` to a concrete model via
       :class:`ModelFactory`.
    2. Consult the inference cache (:class:`IInferenceRepository`) so that
       identical images hit a fast path.
    3. Time the inference and record a metric via
       :class:`IMetricsRepository`.
    4. Return an :class:`InferenceResult` regardless of the concrete model.

Routers never touch caches, factories, or model classes directly - they
depend on this service and receive a strongly-typed result.
"""

from __future__ import annotations

from typing import Any, Dict
import logging
import time

import numpy as np

from app.models.base_model import BaseDetectionModel, InferenceResult
from app.models.model_factory import ModelFactory, ModelType
from app.repositories.inference_repository import IInferenceRepository, compute_image_hash
from app.repositories.metrics_repository import IMetricsRepository, MetricRecord

logger = logging.getLogger(__name__)


class DetectionService:
    """Orchestrates every single-model inference call."""

    def __init__(
        self,
        inference_repo: IInferenceRepository,
        metrics_repo: IMetricsRepository,
        factory: type[ModelFactory] = ModelFactory,
    ) -> None:
        self._repo = inference_repo
        self._metrics = metrics_repo
        self._factory = factory

    # -- Public API ----------------------------------------------------------

    def get_model(self, model_type: ModelType) -> BaseDetectionModel:
        """Return the (lazily instantiated) singleton for ``model_type``."""
        return self._factory.get_or_create(model_type)

    async def detect(
        self,
        image: np.ndarray,
        model_type: ModelType,
        image_bytes: bytes | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> InferenceResult:
        """Run inference, honoring the cache and recording metrics.

        ``image_bytes`` is optional: when omitted we skip caching (we would
        otherwise need to re-encode the numpy array, which is wasteful).
        """
        cache_key = compute_image_hash(image_bytes) if (use_cache and image_bytes) else None

        # -- 1. Cache lookup -------------------------------------------------
        if cache_key:
            cached = await self._repo.get(cache_key, model_type.value)
            if cached is not None:
                logger.info(
                    "cache hit: model=%s key=%s", model_type.value, cache_key[:12]
                )
                # Rebuild the result and mark as cached so downstream callers know
                cached["cached"] = True
                return _dict_to_inference_result(cached, model_type.value)

        # -- 2. Model call ---------------------------------------------------
        model = self.get_model(model_type)
        started = time.perf_counter()
        try:
            result = model.infer(image, **kwargs)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000.0
            await self._metrics.record(
                MetricRecord.now(model_type.value, elapsed, 0, cached=False)
            )
            raise
        result.cached = False

        # -- 3. Persist ------------------------------------------------------
        if cache_key:
            await self._repo.set(cache_key, model_type.value, result.to_dict())

        await self._metrics.record(
            MetricRecord.now(
                model_type.value,
                result.inference_time_ms,
                result.count,
                cached=False,
            )
        )
        logger.info(
            "inference: model=%s ms=%.1f detections=%d",
            model_type.value,
            result.inference_time_ms,
            result.count,
        )
        return result


# ─── Helpers ───────────────────────────────────────────────────────────────


def _dict_to_inference_result(payload: Dict[str, Any], model_name: str) -> InferenceResult:
    """Rebuild an :class:`InferenceResult` from a cached dict payload."""
    from app.models.base_model import Detection

    detections = [
        Detection(
            id=d.get("id", i),
            label=d.get("label", "object"),
            confidence=float(d.get("confidence", 0.0)),
            bbox=tuple(d.get("bbox", (0, 0, 0, 0))),
            class_id=d.get("class_id"),
            mask_shape=tuple(d["mask_shape"]) if d.get("mask_shape") else None,
            mask_rle=d.get("mask_rle"),
            extra=d.get("extra", {}),
        )
        for i, d in enumerate(payload.get("detections", []))
    ]
    return InferenceResult(
        model_name=payload.get("model", model_name),
        detections=detections,
        inference_time_ms=float(payload.get("inference_time_ms", 0.0)),
        image_shape=tuple(payload.get("image_shape", (0, 0))),
        cached=True,
        metadata=payload.get("metadata", {}),
    )
