"""
Pipeline service - Grounding DINO + SAM.

Encapsulates the two-stage flow so the router stays a thin adapter:

    text prompt --> Grounding DINO --> boxes --> SAM --> masks

Metrics are recorded against a synthetic ``pipeline`` model name so that
the metrics dashboard shows the pipeline as its own row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import logging
import time

import numpy as np

from app.models.model_factory import ModelFactory, ModelType
from app.repositories.metrics_repository import IMetricsRepository, MetricRecord

if TYPE_CHECKING:
    # These imports pull in torch / transformers / segment-anything.
    # Gate them behind TYPE_CHECKING so the module is importable in
    # stripped-down environments (CI, minimal Docker layers). Duck typing
    # covers the actual method calls at runtime.
    from app.models.grounding_dino import GroundingDINODetector
    from app.models.sam_model import SAMSegmenter

logger = logging.getLogger(__name__)


class PipelineService:
    """Orchestrates the open-vocabulary detect-then-segment pipeline."""

    PIPELINE_NAME = "grounding_dino+sam"

    def __init__(self, metrics_repo: IMetricsRepository) -> None:
        self._metrics = metrics_repo

    # -- Model accessors -----------------------------------------------------

    def _gdino(self) -> "GroundingDINODetector":
        return ModelFactory.get_or_create(ModelType.GROUNDING_DINO)  # type: ignore[return-value]

    def _sam(self) -> "SAMSegmenter":
        return ModelFactory.get_or_create(ModelType.SAM)  # type: ignore[return-value]

    # -- Public API ----------------------------------------------------------

    async def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        return_masks: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return a JSON-ready dict."""
        started = time.perf_counter()
        gdino = self._gdino()
        sam = self._sam()

        # Stage 1: open-vocabulary detection
        boxes, labels, scores = gdino.get_boxes_for_sam(
            image, text_prompt, box_threshold=box_threshold
        )
        detections: List[Dict[str, Any]] = [
            {
                "id": i,
                "bbox": boxes[i].tolist(),
                "confidence": float(scores[i]),
                "label": labels[i],
            }
            for i in range(len(boxes))
        ]

        if len(boxes) == 0:
            elapsed = (time.perf_counter() - started) * 1000.0
            await self._metrics.record(
                MetricRecord.now(self.PIPELINE_NAME, elapsed, 0, cached=False)
            )
            return {
                "detection_model": "grounding_dino",
                "segmentation_model": "sam",
                "text_prompt": text_prompt,
                "detections": [],
                "segments": [],
                "detection_count": 0,
                "segment_count": 0,
                "image_shape": list(image.shape[:2]),
                "inference_time_ms": round(elapsed, 2),
                "message": "No objects matching the text prompt were detected.",
            }

        # Stage 2: pixel-level segmentation
        sam_result = sam.predict(image, mode="boxes", boxes=boxes)
        raw_masks: Optional[List[np.ndarray]] = None
        if return_masks:
            raw_masks = sam.get_masks_raw(image, boxes)

        elapsed = (time.perf_counter() - started) * 1000.0
        await self._metrics.record(
            MetricRecord.now(
                self.PIPELINE_NAME,
                elapsed,
                len(detections),
                cached=False,
            )
        )
        logger.info(
            "pipeline: prompt=%r detections=%d ms=%.1f",
            text_prompt,
            len(detections),
            elapsed,
        )

        payload: Dict[str, Any] = {
            "detection_model": "grounding_dino",
            "segmentation_model": "sam",
            "text_prompt": text_prompt,
            "detections": detections,
            "segments": sam_result["segments"],
            "detection_count": len(detections),
            "segment_count": len(sam_result["segments"]),
            "image_shape": list(image.shape[:2]),
            "inference_time_ms": round(elapsed, 2),
        }
        if raw_masks is not None:
            payload["raw_masks"] = raw_masks  # numpy arrays - consumer must handle
        return payload
