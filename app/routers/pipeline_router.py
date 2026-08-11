"""
Combined pipeline endpoint: Grounding DINO -> SAM.

Delegates all orchestration to :class:`PipelineService`; the router is a
thin FastAPI adapter that only translates between HTTP and the service.
"""

from __future__ import annotations

import logging
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies import get_pipeline_service
from app.services.pipeline_service import PipelineService
from app.utils.image_utils import image_to_bytes, load_image_from_upload
from app.utils.visualization import draw_detections, draw_masks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Combined Pipeline"])


@router.post("/detect-and-segment", summary="Text prompt -> Detect -> Segment")
async def detect_and_segment(
    file: UploadFile = File(..., description="Image file"),
    text_prompt: str = Form(..., description="Describe objects to find"),
    box_threshold: float = Form(0.35, description="Detection confidence threshold"),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Open-vocabulary instance segmentation.

    Grounding DINO turns natural language into bounding boxes; SAM turns
    those boxes into pixel-perfect masks.  Together they let a caller
    describe any object and get segmentation without retraining.
    """
    image = await load_image_from_upload(file)
    payload = await service.detect_and_segment(
        image, text_prompt=text_prompt, box_threshold=box_threshold
    )
    return payload


@router.post(
    "/detect-and-segment-visualize",
    summary="Full pipeline + annotated image response",
)
async def detect_segment_visualize(
    file: UploadFile = File(...),
    text_prompt: str = Form(..., description="Describe objects to find"),
    box_threshold: float = Form(0.35),
    service: PipelineService = Depends(get_pipeline_service),
):
    """Run the pipeline and return an annotated JPEG with masks + boxes."""
    image = await load_image_from_upload(file)
    payload = await service.detect_and_segment(
        image,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        return_masks=True,
    )

    if not payload.get("detections"):
        img_bytes = image_to_bytes(image, format="JPEG")
        return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")

    annotated = draw_masks(image, payload["raw_masks"], alpha=0.4)
    annotated = draw_detections(annotated, payload["detections"])
    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")
