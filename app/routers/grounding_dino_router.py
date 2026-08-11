"""Grounding DINO API endpoints - open-set, text-prompted detection."""

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
from app.utils.visualization import draw_detections

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grounding-dino", tags=["Grounding DINO"])


def get_model():
    return ModelFactory.get_or_create(ModelType.GROUNDING_DINO)


@router.post("/detect", summary="Open-set object detection with text prompt")
async def detect_with_text(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_prompt: str = Form(
        ...,
        description="Text description of objects to find (e.g. 'cat . dog . person')",
    ),
    box_threshold: float = Form(0.35, description="Box confidence threshold"),
    text_threshold: float = Form(0.25, description="Text-matching threshold"),
    service: DetectionService = Depends(get_detection_service),
):
    """Detect anything describable in natural language - no fixed classes."""
    image_bytes = await read_upload_bytes(file)
    image = decode_image_bytes(image_bytes, filename=file.filename)
    result = await service.detect(
        image,
        ModelType.GROUNDING_DINO,
        image_bytes=image_bytes,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    payload = result.to_dict()
    payload["text_prompt"] = text_prompt
    return payload


@router.post("/detect-visualize", summary="Detect with text and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    text_prompt: str = Form(..., description="What to detect"),
    box_threshold: float = Form(0.35),
):
    image = await load_image_from_upload(file)
    model = get_model()
    model.ensure_loaded()
    results = model.predict(image, text_prompt=text_prompt, box_threshold=box_threshold)
    annotated = draw_detections(image, results["detections"])
    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get Grounding DINO model information")
async def model_info():
    return get_model().get_model_info()
