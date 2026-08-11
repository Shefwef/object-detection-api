"""Detectron2 API endpoints - routed through :class:`DetectionService`."""

from __future__ import annotations

import logging
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies import get_detection_service
from app.models.model_factory import ModelFactory, ModelType
from app.services.detection_service import DetectionService
from app.utils.image_utils import (
    decode_image_bytes,
    image_to_bytes,
    load_image_from_upload,
    read_upload_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detectron2", tags=["Detectron2"])


def get_model():
    return ModelFactory.get_or_create(ModelType.DETECTRON2)


@router.post("/detect", summary="Detect objects using Detectron2")
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Form(0.5, description="Confidence threshold"),
    return_masks: bool = Form(True, description="Include segmentation masks"),
    service: DetectionService = Depends(get_detection_service),
):
    """Mask R-CNN instance segmentation over the uploaded image."""
    image_bytes = await read_upload_bytes(file)
    image = decode_image_bytes(image_bytes, filename=file.filename)
    result = await service.detect(
        image,
        ModelType.DETECTRON2,
        image_bytes=image_bytes,
        confidence=confidence,
        return_masks=return_masks,
    )
    return result.to_dict()


@router.post("/detect-visualize", summary="Detect and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    """Return Detectron2's own visualization (boxes + masks) as JPEG."""
    image = await load_image_from_upload(file)
    model = get_model()
    model.ensure_loaded()
    results = model.predict(image, confidence=confidence)
    annotated = model.visualize(image, results)
    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get Detectron2 model information")
async def model_info():
    return get_model().get_model_info()
