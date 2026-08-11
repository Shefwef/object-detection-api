"""YOLOv8 API endpoints - routed through :class:`DetectionService`."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_detection_service
from app.models.model_factory import ModelFactory, ModelType
from app.services.detection_service import DetectionService
from app.utils.image_utils import (
    image_to_bytes,
    load_image_from_upload,
    read_upload_bytes,
    decode_image_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["YOLOv8"])


class Base64ImageRequest(BaseModel):
    """Payload for the low-latency webcam endpoint."""

    image: str = Field(
        ...,
        description="Base64-encoded image (may include the data: URL prefix).",
        examples=["data:image/jpeg;base64,/9j/4AAQ..."],
    )
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)


# ─── Endpoints ─────────────────────────────────────────────────────────────


def get_model():
    """Backwards-compatible accessor - some tests import this directly."""
    return ModelFactory.get_or_create(ModelType.YOLO)


@router.post("/detect", summary="Detect objects using YOLOv8")
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Form(0.25, description="Confidence threshold (0-1)"),
    iou_threshold: float = Form(0.45, description="IoU threshold for NMS"),
    max_detections: int = Form(300, description="Maximum detections"),
    service: DetectionService = Depends(get_detection_service),
):
    """Run YOLOv8 detection and return bounding boxes + class labels."""
    image_bytes = await read_upload_bytes(file)
    image = decode_image_bytes(image_bytes, filename=file.filename)

    result = await service.detect(
        image,
        ModelType.YOLO,
        image_bytes=image_bytes,
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    return _legacy_response(result, extra={})


@router.post("/detect-base64", summary="Detect from base64-encoded frame (webcam)")
async def detect_base64(
    payload: Base64ImageRequest,
    service: DetectionService = Depends(get_detection_service),
):
    """Detect objects in a base64 image - optimized for webcam streaming.

    Avoids the multipart round-trip so browsers can push frames every 200ms
    without a noticeable overhead on the client side.
    """
    b64 = payload.image.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:  # noqa: BLE001 - user-facing error
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")
    image = decode_image_bytes(raw, filename="frame.jpg")

    result = await service.detect(
        image,
        ModelType.YOLO,
        image_bytes=raw,
        confidence=payload.confidence,
        iou_threshold=payload.iou_threshold,
    )
    return _legacy_response(result, extra={})


@router.post("/detect-visualize", summary="Detect and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """Return the annotated JPEG produced by YOLOv8's own plotter."""
    image = await load_image_from_upload(file)
    model = get_model()
    _, annotated = model.detect_with_visualization(image, confidence)
    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get YOLOv8 model information")
async def model_info():
    return get_model().get_model_info()


# ─── Legacy response adapter ───────────────────────────────────────────────


def _legacy_response(result, extra: dict) -> dict:
    """Normalize an :class:`InferenceResult` into the pre-refactor dict shape.

    Preserves the fields that existing clients (and tests) expect while also
    surfacing the newly-added ``inference_time_ms`` / ``cached`` fields.
    """
    payload = result.to_dict()
    payload.setdefault("metadata", {})
    payload.update(extra)
    return payload
