"""
YOLOv8 API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
from io import BytesIO
import logging

from app.models.yolo_model import YOLODetector
from app.utils.image_utils import load_image_from_upload, image_to_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["YOLOv8"])

# Singleton model instance (lazy loaded)
_model: Optional[YOLODetector] = None


def get_model() -> YOLODetector:
    global _model
    if _model is None:
        _model = YOLODetector()
    return _model


@router.post("/detect", summary="Detect objects using YOLOv8")
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Form(0.25, description="Confidence threshold (0-1)"),
    iou_threshold: float = Form(0.45, description="IoU threshold for NMS"),
    max_detections: int = Form(300, description="Maximum detections"),
):
    """
    Run YOLOv8 object detection on an uploaded image.
    
    Returns bounding boxes, class labels, and confidence scores
    for all detected objects (80 COCO classes).
    """
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(
        image,
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    return results


@router.post("/detect-visualize", summary="Detect and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):
    """
    Run detection and return the annotated image with drawn bounding boxes.
    Returns a JPEG image.
    """
    image = await load_image_from_upload(file)
    model = get_model()

    _, annotated = model.detect_with_visualization(image, confidence)

    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get YOLOv8 model information")
async def model_info():
    """Get information about the loaded YOLOv8 model."""
    model = get_model()
    return model.get_model_info()
