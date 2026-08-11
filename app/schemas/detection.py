"""
Pydantic schemas for request / response validation.

Kept broadly backwards compatible with v1 of the API - existing consumers
receive the same fields, plus the newly exposed ``inference_time_ms`` and
``cached`` flags on every detection response.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


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
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    classes: Optional[List[int]] = None
    max_detections: int = Field(300, ge=1, le=1000)


class Detectron2Request(BaseModel):
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    return_masks: bool = True


class GroundingDINORequest(BaseModel):
    text_prompt: str = Field(..., min_length=1, max_length=500)
    box_threshold: float = Field(0.35, ge=0.0, le=1.0)
    text_threshold: float = Field(0.25, ge=0.0, le=1.0)

    @field_validator("text_prompt")
    @classmethod
    def _clean(cls, v: str) -> str:
        return v.strip()


class SAMRequest(BaseModel):
    mode: SAMMode = SAMMode.AUTO
    point_coords: Optional[List[List[float]]] = None
    point_labels: Optional[List[int]] = None
    boxes: Optional[List[List[float]]] = None
    multimask_output: bool = True


class PipelineRequest(BaseModel):
    text_prompt: str = Field(..., min_length=1, max_length=500)
    box_threshold: float = Field(0.35, ge=0.0, le=1.0)
    return_visualization: bool = False


class Base64ImageRequest(BaseModel):
    """Used by the webcam detect endpoint to skip multipart overhead."""

    image: str = Field(..., description="Base64 image (may include data: prefix)")
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)


# ─── Response Schemas ─────────────────────────────────


class Detection(BaseModel):
    id: int
    bbox: List[float] = Field(description="[x1, y1, x2, y2]")
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    label: Optional[str] = None
    mask_rle: Optional[Dict[str, Any]] = None
    mask_shape: Optional[List[int]] = None


class Segment(BaseModel):
    id: int
    score: Optional[float] = None
    area: int
    mask_shape: List[int]
    bbox: Optional[List[float]] = None
    input_box: Optional[List[float]] = None
    predicted_iou: Optional[float] = None
    stability_score: Optional[float] = None


class DetectionResponse(BaseModel):
    model: str
    detections: List[Detection]
    count: int
    image_shape: List[int]
    inference_time_ms: Optional[float] = None
    cached: Optional[bool] = None
    metadata: Dict[str, Any] = {}
    text_prompt: Optional[str] = None


class SegmentationResponse(BaseModel):
    model: str
    mode: str
    segments: List[Segment]
    count: int
    image_shape: List[int]
    metadata: Dict[str, Any] = {}


class PipelineResponse(BaseModel):
    detection_model: str = "grounding_dino"
    segmentation_model: str = "sam"
    text_prompt: str
    detections: List[Detection]
    segments: List[Segment]
    detection_count: int
    segment_count: int
    image_shape: List[int]
    inference_time_ms: Optional[float] = None


class ExplainResponse(BaseModel):
    method: str = Field(description="'grad-cam' or 'saliency-fallback'")
    heatmap_base64: Optional[str] = None
    caption: Optional[str] = None
    detections: List[Dict[str, Any]] = []
    target_detection: Optional[Dict[str, Any]] = None


class MetricRow(BaseModel):
    total_requests: int
    cache_hit_rate: float
    avg_latency_ms: float
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    avg_detections: float
    last_seen: Optional[str] = None


# The summary endpoint returns Dict[str, MetricRow] directly. We avoid a
# wrapper model since Pydantic v2 dropped __root__ in favour of RootModel;
# a plain dict keeps the OpenAPI schema simple.


class HealthResponse(BaseModel):
    status: str
    version: str
    models: Dict[str, Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    is_loaded: bool
    type: str
