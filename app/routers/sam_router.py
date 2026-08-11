"""SAM (Segment Anything) API endpoints."""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.model_factory import ModelFactory, ModelType
from app.utils.image_utils import load_image_from_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sam", tags=["SAM (Segment Anything)"])


def get_model():
    return ModelFactory.get_or_create(ModelType.SAM)


@router.post("/segment-auto", summary="Automatically segment everything")
async def segment_everything(
    file: UploadFile = File(..., description="Image file"),
):
    """Grid-prompted segmentation over the entire image."""
    image = await load_image_from_upload(file)
    model = get_model()
    model.ensure_loaded()
    return model.predict(image, mode="auto")


@router.post("/segment-points", summary="Segment with point prompts")
async def segment_with_points(
    file: UploadFile = File(...),
    points: str = Form(..., description="Point coordinates as JSON: [[x1,y1],[x2,y2]]"),
    labels: str = Form(..., description="Point labels as JSON: [1,0] (1=fg, 0=bg)"),
    multimask_output: bool = Form(True, description="Return multiple mask candidates"),
):
    """Segment objects from user-supplied fg/bg click points."""
    try:
        point_coords = np.array(json.loads(points), dtype=np.float32)
        point_labels = np.array(json.loads(labels), dtype=np.int32)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points/labels format: {e}")

    if len(point_coords) != len(point_labels):
        raise HTTPException(status_code=400, detail="Number of points and labels must match")

    image = await load_image_from_upload(file)
    model = get_model()
    model.ensure_loaded()
    return model.predict(
        image,
        mode="points",
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
    )


@router.post("/segment-boxes", summary="Segment with bounding box prompts")
async def segment_with_boxes(
    file: UploadFile = File(...),
    boxes: str = Form(..., description="Bounding boxes as JSON: [[x1,y1,x2,y2]]"),
):
    """Segment inside supplied boxes (used by the G-DINO + SAM pipeline)."""
    try:
        box_array = np.array(json.loads(boxes), dtype=np.float32)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid boxes format: {e}")

    image = await load_image_from_upload(file)
    model = get_model()
    model.ensure_loaded()
    return model.predict(image, mode="boxes", boxes=box_array)


@router.get("/info", summary="Get SAM model information")
async def model_info():
    return get_model().get_model_info()
