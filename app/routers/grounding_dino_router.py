"""
Grounding DINO API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO
import logging

from app.models.grounding_dino import GroundingDINODetector
from app.utils.image_utils import load_image_from_upload, image_to_bytes
from app.utils.visualization import draw_detections

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grounding-dino", tags=["Grounding DINO"])

_model: Optional[GroundingDINODetector] = None


def get_model() -> GroundingDINODetector:
    global _model
    if _model is None:
        _model = GroundingDINODetector()
    return _model


@router.post("/detect", summary="Open-set object detection with text prompt")
async def detect_with_text(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_prompt: str = Form(..., description="Text description of objects to find (e.g., 'cat . dog . person')"),
    box_threshold: float = Form(0.35, description="Box confidence threshold"),
    text_threshold: float = Form(0.25, description="Text matching threshold"),
):
    """
    Detect objects matching a text description using Grounding DINO.
    
    This is OPEN-SET detection — you can detect ANY object by describing it.
    Separate multiple categories with periods: "cat . dog . person"
    
    Examples:
    - "person wearing red" — finds people wearing red clothing
    - "car . bicycle . truck" — finds vehicles
    - "damaged area" — finds damaged regions
    """
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(
        image,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    return results


@router.post("/detect-visualize", summary="Detect with text and return annotated image")
async def detect_and_visualize(
    file: UploadFile = File(...),
    text_prompt: str = Form(..., description="What to detect"),
    box_threshold: float = Form(0.35),
):
    """Detect with Grounding DINO and return annotated image."""
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(image, text_prompt=text_prompt, box_threshold=box_threshold)
    annotated = draw_detections(image, results["detections"])

    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")


@router.get("/info", summary="Get Grounding DINO model information")
async def model_info():
    model = get_model()
    return model.get_model_info()
