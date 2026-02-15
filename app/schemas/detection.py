"""
Pydantic schemas for API request/response validation.
Ensures type safety and generates automatic Swagger documentation.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────

class ModelType(str, Enum):
    YOLO = "yolov8"
    DETECTRON2 = "detectron2"
    GROUNDING_DINO = "grounding_dino"
    SAM = "sam"


class SAMMode(str, Enum):
    AUTO = "auto"
    POINTS = "points"
    BOXES = "boxes"
    POINTS_AND_BOXES = "points_and_boxes"


# ─── Request Schemas ──────────────────────────────────

class YOLORequest(BaseModel):
    """Parameters for YOLO detection."""
    confidence: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    classes: Optional[List[int]] = Field(None, description="Filter specific class IDs")
    max_detections: int = Field(300, ge=1, le=1000, description="Max detections")


class Detectron2Request(BaseModel):
    """Parameters for Detectron2 detection."""
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    return_masks: bool = Field(True, description="Include segmentation masks")


class GroundingDINORequest(BaseModel):
    """Parameters for Grounding DINO detection."""
    text_prompt: str = Field(..., description="Text description of objects to detect (separate multiple with '.')")
    box_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Box confidence threshold")
    text_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Text matching threshold")


class SAMRequest(BaseModel):
    """Parameters for SAM segmentation."""
    mode: SAMMode = Field(SAMMode.AUTO, description="Segmentation mode")
    point_coords: Optional[List[List[float]]] = Field(None, description="Point coordinates [[x1,y1], [x2,y2]]")
    point_labels: Optional[List[int]] = Field(None, description="Point labels (1=foreground, 0=background)")
    boxes: Optional[List[List[float]]] = Field(None, description="Bounding boxes [[x1,y1,x2,y2]]")
    multimask_output: bool = Field(True, description="Return multiple masks per prompt")


class PipelineRequest(BaseModel):
    """Parameters for Grounding DINO + SAM pipeline."""
    text_prompt: str = Field(..., description="Text description of objects to detect")
    box_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Detection confidence")
    return_visualization: bool = Field(False, description="Include annotated image")


# ─── Response Schemas ─────────────────────────────────

class Detection(BaseModel):
    """Single detection result."""
    id: int
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    label: Optional[str] = None
    mask_rle: Optional[Dict[str, Any]] = None


class Segment(BaseModel):
    """Single segmentation result."""
    id: int
    score: Optional[float] = None
    area: int
    mask_shape: List[int]
    bbox: Optional[List[float]] = None
    input_box: Optional[List[float]] = None
    predicted_iou: Optional[float] = None
    stability_score: Optional[float] = None


class DetectionResponse(BaseModel):
    """Response for detection endpoints."""
    model: str
    detections: List[Detection]
    count: int
    image_shape: List[int]
    metadata: Dict[str, Any] = {}
    text_prompt: Optional[str] = None


class SegmentationResponse(BaseModel):
    """Response for segmentation endpoints."""
    model: str
    mode: str
    segments: List[Segment]
    count: int
    image_shape: List[int]
    metadata: Dict[str, Any] = {}


class PipelineResponse(BaseModel):
    """Response for combined pipeline."""
    detection_model: str = "grounding_dino"
    segmentation_model: str = "sam"
    text_prompt: str
    detections: List[Detection]
    segments: List[Segment]
    detection_count: int
    segment_count: int
    image_shape: List[int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models: Dict[str, Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    is_loaded: bool
    type: str
