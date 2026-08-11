"""Detection service tests using a stub model - no ML weights required."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from app.models.base_model import BaseDetectionModel
from app.models.model_factory import ModelFactory, ModelType
from app.repositories.inference_repository import InMemoryInferenceRepository
from app.repositories.metrics_repository import InMemoryMetricsRepository
from app.services.detection_service import DetectionService


# ─── Stub model ────────────────────────────────────────────────────────────


class StubModel(BaseDetectionModel):
    """Tiny stand-in that mimics the shape of a real model output."""

    call_count = 0

    def __init__(self) -> None:
        super().__init__(model_name="stub-model")
        StubModel.call_count = 0

    def load_model(self) -> None:  # noqa: D401 - stub
        pass

    def predict(self, image: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        StubModel.call_count += 1
        return {
            "model": "stub",
            "detections": [
                {
                    "id": 0,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "widget",
                }
            ],
            "count": 1,
            "image_shape": list(image.shape[:2]),
            "metadata": {"note": "stub"},
        }


@pytest.fixture(autouse=True)
def _register_stub():
    """Swap the YOLO slot for the stub model during each test."""
    ModelFactory.register(ModelType.YOLO, StubModel)
    ModelFactory.reset()
    yield
    ModelFactory.reset()
    # Restore the real registration so subsequent tests get the real class.
    from app.models.yolo_model import YOLODetector
    ModelFactory.register(ModelType.YOLO, YOLODetector)


# ─── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_service_normalizes_inference_result(sample_image_array):
    service = DetectionService(
        inference_repo=InMemoryInferenceRepository(),
        metrics_repo=InMemoryMetricsRepository(),
    )
    result = await service.detect(sample_image_array, ModelType.YOLO, image_bytes=None)
    assert result.model_name == "stub"
    assert result.count == 1
    assert result.detections[0].label == "widget"
    assert result.inference_time_ms >= 0
    assert result.cached is False


@pytest.mark.asyncio
async def test_service_uses_cache_on_second_call(sample_image_array, sample_image_bytes):
    service = DetectionService(
        inference_repo=InMemoryInferenceRepository(),
        metrics_repo=InMemoryMetricsRepository(),
    )
    first = await service.detect(sample_image_array, ModelType.YOLO, image_bytes=sample_image_bytes)
    second = await service.detect(sample_image_array, ModelType.YOLO, image_bytes=sample_image_bytes)

    assert first.cached is False
    assert second.cached is True
    assert StubModel.call_count == 1  # second call served from cache


@pytest.mark.asyncio
async def test_service_records_metrics(sample_image_array, sample_image_bytes):
    metrics = InMemoryMetricsRepository()
    service = DetectionService(
        inference_repo=InMemoryInferenceRepository(),
        metrics_repo=metrics,
    )
    await service.detect(sample_image_array, ModelType.YOLO, image_bytes=sample_image_bytes)
    summary = await metrics.summary()
    assert "yolov8" in summary
    assert summary["yolov8"]["total_requests"] == 1
    assert summary["yolov8"]["avg_latency_ms"] >= 0.0
