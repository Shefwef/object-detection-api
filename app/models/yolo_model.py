"""
YOLOv8 Object Detection Model Wrapper.

YOLOv8 (You Only Look Once v8) by Ultralytics is a state-of-the-art,
real-time object detection model. It processes the entire image in a
single forward pass, making it extremely fast.

Key Concepts:
- Single-stage detector: predicts bounding boxes and class probabilities in one pass
- Anchor-free detection: no predefined anchor boxes needed
- Multi-scale feature fusion: detects objects at different scales
- Supports detection, segmentation, classification, and pose estimation

COCO Classes (80): person, bicycle, car, motorcycle, airplane, bus, train, truck,
boat, traffic light, fire hydrant, stop sign, ... etc.
"""

import numpy as np
from typing import Any, Dict, List, Optional
import logging

from app.models.base_model import BaseDetectionModel
from app.config import get_settings, get_device

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetectionModel):
    """
    YOLOv8 object detection wrapper using the Ultralytics library.
    
    Supports:
    - Object Detection (bounding boxes + class labels)
    - Instance Segmentation (pixel-level masks)
    - Multiple model sizes: nano(n), small(s), medium(m), large(l), xlarge(x)
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(model_name=f"YOLOv8 ({settings.YOLO_MODEL_NAME})")
        self.model_path = settings.YOLO_MODEL_NAME
        self.default_confidence = settings.YOLO_CONFIDENCE
        self.default_iou = settings.YOLO_IOU_THRESHOLD
        self.device = get_device()

    def load_model(self) -> None:
        """Load YOLOv8 model using Ultralytics library."""
        from ultralytics import YOLO

        logger.info(f"Loading YOLOv8 model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        logger.info("YOLOv8 model loaded successfully")

    def predict(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        max_detections: int = 300,
    ) -> Dict[str, Any]:
        """
        Run YOLOv8 inference on an image.
        
        Args:
            image: Input image (BGR numpy array)
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
            classes: Filter by class indices (e.g., [0] for person only)
            max_detections: Maximum number of detections to return
            
        Returns:
            Dictionary with detections and metadata
        """
        self.ensure_loaded()

        conf = confidence or self.default_confidence
        iou = iou_threshold or self.default_iou

        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            classes=classes,
            max_det=max_detections,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = []
        result = results[0]  # Single image

        for i, box in enumerate(result.boxes):
            detection = {
                "id": i,
                "bbox": box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                "confidence": float(box.conf[0].cpu()),
                "class_id": int(box.cls[0].cpu()),
                "class_name": result.names[int(box.cls[0].cpu())],
            }

            # Include segmentation mask if available
            if result.masks is not None and i < len(result.masks):
                detection["mask"] = result.masks[i].data.cpu().numpy().tolist()

            detections.append(detection)

        return {
            "model": "yolov8",
            "detections": detections,
            "count": len(detections),
            "image_shape": list(image.shape[:2]),
            "metadata": {
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "device": self.device,
                "model_variant": self.model_path,
            },
        }

    def detect_with_visualization(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
    ) -> tuple:
        """
        Run detection and return both results and annotated image.
        
        Returns:
            Tuple of (detection_results, annotated_image)
        """
        self.ensure_loaded()

        conf = confidence or self.default_confidence
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False,
        )

        # Get annotated image from YOLO's built-in plotting
        annotated = results[0].plot()

        # Parse detections
        det_results = self.predict(image, confidence=confidence)

        return det_results, annotated
