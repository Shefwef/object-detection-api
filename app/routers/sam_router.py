"""
SAM (Segment Anything Model) API endpoints.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO
import numpy as np
import json
import logging

from app.models.sam_model import SAMSegmenter
from app.utils.image_utils import load_image_from_upload, image_to_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sam", tags=["SAM (Segment Anything)"])

_model: Optional[SAMSegmenter] = None


def get_model() -> SAMSegmenter:
    global _model
    if _model is None:
        _model = SAMSegmenter()
    return _model


@router.post("/segment-auto", summary="Automatically segment everything")
async def segment_everything(
    file: UploadFile = File(..., description="Image file"),
):
    """
    Automatically segment ALL objects in the image.
    
    SAM uses a grid of point prompts to discover and segment every
    object/region in the image. Returns masks sorted by area.
    """
    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(image, mode="auto")
    return results


@router.post("/segment-points", summary="Segment with point prompts")
async def segment_with_points(
    file: UploadFile = File(...),
    points: str = Form(..., description='Point coordinates as JSON: [[x1,y1],[x2,y2]]'),
    labels: str = Form(..., description='Point labels as JSON: [1,0] (1=foreground, 0=background)'),
    multimask_output: bool = Form(True, description="Return multiple mask candidates"),
):
    """
    Segment objects using point prompts (click locations).
    
    Provide (x, y) coordinates where:
    - Label 1 = Include this point (foreground)
    - Label 0 = Exclude this point (background)
    
    SAM returns up to 3 masks at different granularity levels.
    """
    try:
        point_coords = np.array(json.loads(points), dtype=np.float32)
        point_labels = np.array(json.loads(labels), dtype=np.int32)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points/labels format: {e}")

    if len(point_coords) != len(point_labels):
        raise HTTPException(status_code=400, detail="Number of points and labels must match")

    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(
        image,
        mode="points",
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
    )

    return results


@router.post("/segment-boxes", summary="Segment with bounding box prompts")
async def segment_with_boxes(
    file: UploadFile = File(...),
    boxes: str = Form(..., description='Bounding boxes as JSON: [[x1,y1,x2,y2]]'),
):
    """
    Segment objects within provided bounding boxes.
    
    This endpoint is designed for the Grounding DINO → SAM pipeline:
    1. Grounding DINO detects objects → provides boxes
    2. SAM generates precise masks from those boxes
    
    Each box gets its own high-quality segmentation mask.
    """
    try:
        box_array = np.array(json.loads(boxes), dtype=np.float32)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid boxes format: {e}")

    image = await load_image_from_upload(file)
    model = get_model()

    results = model.predict(
        image,
        mode="boxes",
        boxes=box_array,
    )

    return results


@router.get("/info", summary="Get SAM model information")
async def model_info():
    model = get_model()
    return model.get_model_info()
