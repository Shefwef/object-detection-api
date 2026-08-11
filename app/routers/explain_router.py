"""
Explainability endpoint - Grad-CAM / saliency for YOLOv8 detections.

Answers the question "*why* did the model pick that object?" by returning
a heatmap that highlights the pixels that most influenced the prediction.
Downstream UIs can toggle between the raw image, the annotated result, and
the heatmap.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import get_explainability_service
from app.services.explainability_service import ExplainabilityService
from app.utils.image_utils import load_image_from_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("/gradcam", summary="Grad-CAM / saliency heatmap for a YOLO detection")
async def explain_detection(
    file: UploadFile = File(..., description="Image file"),
    detection_index: Optional[int] = Form(
        None, description="Which detection to explain (default: highest confidence)"
    ),
    confidence: float = Form(0.25, description="Detection threshold used before explaining"),
    service: ExplainabilityService = Depends(get_explainability_service),
):
    """Return a base64 PNG heatmap showing pixel-level attribution.

    Uses ``pytorch-grad-cam`` when available and falls back to a Sobel + box
    Gaussian saliency map so the endpoint always returns something useful.
    """
    image = await load_image_from_upload(file)
    return service.explain(image, detection_index=detection_index, confidence=confidence)
