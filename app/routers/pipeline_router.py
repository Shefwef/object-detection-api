"""
Combined Pipeline API endpoint: Grounding DINO + SAM.

This is the most impressive and practical pipeline — it combines
open-set text-based detection (Grounding DINO) with universal
segmentation (SAM) to achieve:

    "Describe anything → Detect it → Segment it precisely"

This pipeline is likely what IML's product uses: a flexible system
where users/clients can specify what to detect via text, and get
pixel-perfect segmentation masks without any model retraining.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO
import numpy as np
import logging

from app.models.grounding_dino import GroundingDINODetector
from app.models.sam_model import SAMSegmenter
from app.utils.image_utils import load_image_from_upload, image_to_bytes
from app.utils.visualization import draw_detections, draw_masks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Combined Pipeline"])

# Shared model instances
_gdino: Optional[GroundingDINODetector] = None
_sam: Optional[SAMSegmenter] = None


def get_gdino() -> GroundingDINODetector:
    global _gdino
    if _gdino is None:
        _gdino = GroundingDINODetector()
    return _gdino


def get_sam() -> SAMSegmenter:
    global _sam
    if _sam is None:
        _sam = SAMSegmenter()
    return _sam


@router.post("/detect-and-segment", summary="Detect with text → Segment with SAM")
async def detect_and_segment(
    file: UploadFile = File(..., description="Image file"),
    text_prompt: str = Form(..., description="Describe objects to find (e.g., 'cat . dog')"),
    box_threshold: float = Form(0.35, description="Detection confidence threshold"),
):
    """
    **Grounding DINO + SAM Pipeline**
    
    1. Grounding DINO detects objects matching your text description
    2. Detection boxes are passed to SAM as prompts
    3. SAM generates precise segmentation masks for each detection
    
    This enables open-vocabulary instance segmentation:
    describe ANY object in natural language and get pixel-perfect masks.
    
    Example prompts:
    - "person . car" → detect and segment people and cars
    - "damaged area on the wall" → segment damage
    - "product on shelf" → segment products
    """
    image = await load_image_from_upload(file)

    gdino = get_gdino()
    sam = get_sam()

    # Step 1: Detect with Grounding DINO
    logger.info(f"Pipeline Step 1: Detecting '{text_prompt}' with Grounding DINO")
    boxes, labels, scores = gdino.get_boxes_for_sam(
        image, text_prompt, box_threshold=box_threshold
    )

    if len(boxes) == 0:
        return {
            "detection_model": "grounding_dino",
            "segmentation_model": "sam",
            "text_prompt": text_prompt,
            "detections": [],
            "segments": [],
            "detection_count": 0,
            "segment_count": 0,
            "image_shape": list(image.shape[:2]),
            "message": "No objects matching the text prompt were detected.",
        }

    # Step 2: Segment with SAM
    logger.info(f"Pipeline Step 2: Segmenting {len(boxes)} detections with SAM")
    sam_results = sam.predict(image, mode="boxes", boxes=boxes)

    # Combine results
    detections = []
    for i in range(len(boxes)):
        det = {
            "id": i,
            "bbox": boxes[i].tolist(),
            "confidence": scores[i],
            "label": labels[i],
        }
        detections.append(det)

    return {
        "detection_model": "grounding_dino",
        "segmentation_model": "sam",
        "text_prompt": text_prompt,
        "detections": detections,
        "segments": sam_results["segments"],
        "detection_count": len(detections),
        "segment_count": len(sam_results["segments"]),
        "image_shape": list(image.shape[:2]),
    }


@router.post(
    "/detect-and-segment-visualize",
    summary="Detect + Segment + Return annotated image",
)
async def detect_segment_visualize(
    file: UploadFile = File(...),
    text_prompt: str = Form(..., description="Describe objects to find"),
    box_threshold: float = Form(0.35),
):
    """
    Full pipeline with visualization: detect → segment → draw results.
    Returns an annotated JPEG image with boxes and mask overlays.
    """
    image = await load_image_from_upload(file)

    gdino = get_gdino()
    sam = get_sam()

    # Step 1: Detect
    boxes, labels, scores = gdino.get_boxes_for_sam(
        image, text_prompt, box_threshold=box_threshold
    )

    if len(boxes) == 0:
        img_bytes = image_to_bytes(image, format="JPEG")
        return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")

    # Step 2: Get raw masks from SAM
    masks = sam.get_masks_raw(image, boxes)

    # Step 3: Visualize
    # Draw masks first (as overlay)
    annotated = draw_masks(image, masks, alpha=0.4)

    # Draw detection boxes and labels on top
    detections = [
        {"id": i, "bbox": boxes[i].tolist(), "label": labels[i], "confidence": scores[i]}
        for i in range(len(boxes))
    ]
    annotated = draw_detections(annotated, detections)

    img_bytes = image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")
