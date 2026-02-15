"""
Detectron2 API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO
import logging

from app.models.detectron2_model import Detectron2Detector
from app.utils.image_utils import load_image_from_upload, image_to_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detectron2", tags=["Detectron2"])

_model: Optional[Detectron2Detector] = None


def get_model() -> Detectron2Detector:
    global _model
    if _model is None:
        _model = Detectron2Detector()
    return _model


@router.post("/detect", summary="Detect objects using Detectron2")
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Form(0.5, description="Confidence threshold"),
    return_masks: bool = Form(True, description="Include segmentation masks"),
):
    """
    Run Detectron2 Mask R-CNN instance segmentation.
    
    Returns bounding boxes, class labels, confidence scores,
    and optionally instance segmentation masks (RLE encoded).
    """
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(
        image,
        confidence=confidence,
        return_masks=return_masks,
    )

    return results


@router.post("/detect-visualize", summary="Detect and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    """
    Run Detectron2 detection and return annotated image with masks.
    """
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(image, confidence=confidence)
    annotated = model.visualize(image, results)

    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get Detectron2 model information")
async def model_info():
    model = get_model()
    return model.get_model_info()
