"""
Detectron2 Object Detection & Instance Segmentation Model Wrapper.

Detectron2 is Meta AI's (Facebook AI Research) next-generation library for
object detection and segmentation. Built on PyTorch, it provides:

Key Concepts:
- Two-stage detector (Faster R-CNN family): first proposes regions, then classifies
- Feature Pyramid Network (FPN): multi-scale feature extraction
- Mask R-CNN: extends Faster R-CNN with a mask prediction branch
- Panoptic Segmentation: combines instance + semantic segmentation

Architecture (Mask R-CNN):
1. Backbone (ResNet-50/101): extracts feature maps from the image
2. FPN: creates multi-scale feature pyramid
3. Region Proposal Network (RPN): proposes candidate object regions
4. ROI Heads: classifies proposals + predicts boxes + generates masks

This is more heavyweight than YOLO but provides higher accuracy,
especially for instance segmentation tasks.
"""

import numpy as np
from typing import Any, Dict, List, Optional
import logging
import torch

from app.models.base_model import BaseDetectionModel
from app.config import get_settings, get_device

logger = logging.getLogger(__name__)


class Detectron2Detector(BaseDetectionModel):
    """
    Detectron2 instance segmentation and object detection wrapper.
    
    Uses Mask R-CNN with ResNet-50 + FPN backbone by default.
    Supports both detection-only and instance segmentation modes.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(model_name=f"Detectron2 ({settings.DETECTRON2_CONFIG})")
        self.config_name = settings.DETECTRON2_CONFIG
        self.default_confidence = settings.DETECTRON2_CONFIDENCE
        self.device = get_device()
        self.cfg = None
        self.predictor = None

    def load_model(self) -> None:
        """
        Load Detectron2 model with configuration.
        
        Detectron2 uses a config-based system where you specify:
        - Model architecture (e.g., Mask R-CNN with R50-FPN)
        - Pre-trained weights (from Model Zoo)
        - Inference thresholds
        """
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        logger.info(f"Loading Detectron2 config: {self.config_name}")

        # Build configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.config_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.default_confidence
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_name)
        self.cfg.MODEL.DEVICE = self.device

        # Create predictor (handles preprocessing + inference + postprocessing)
        self.predictor = DefaultPredictor(self.cfg)
        logger.info("Detectron2 model loaded successfully")

    def predict(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        return_masks: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Detectron2 inference on an image.
        
        Detectron2 outputs:
        - pred_boxes: predicted bounding boxes (Boxes object)
        - pred_classes: predicted class indices
        - scores: confidence scores
        - pred_masks: binary masks for each instance (if using Mask R-CNN)
        
        Args:
            image: Input image (BGR numpy array, as OpenCV loads)
            confidence: Override confidence threshold
            return_masks: Whether to include segmentation masks
            
        Returns:
            Dictionary with detections and metadata
        """
        self.ensure_loaded()

        # Override confidence if specified
        if confidence is not None:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence

        # Run inference
        outputs = self.predictor(image)

        # Extract predictions from Detectron2's Instances object
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Get class names from COCO metadata
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        class_names = metadata.get("thing_classes", [])

        detections = []
        for i in range(len(boxes)):
            detection = {
                "id": i,
                "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(scores[i]),
                "class_id": int(classes[i]),
                "class_name": class_names[int(classes[i])] if class_names else f"class_{classes[i]}",
            }

            # Include instance segmentation mask
            if return_masks and instances.has("pred_masks"):
                mask = instances.pred_masks[i].numpy()
                detection["mask_shape"] = list(mask.shape)
                # Store mask as run-length encoding for efficiency
                detection["mask_rle"] = self._mask_to_rle(mask)

            detections.append(detection)

        return {
            "model": "detectron2",
            "detections": detections,
            "count": len(detections),
            "image_shape": list(image.shape[:2]),
            "metadata": {
                "config": self.config_name,
                "confidence_threshold": confidence or self.default_confidence,
                "device": self.device,
                "has_masks": instances.has("pred_masks"),
            },
        }

    def _mask_to_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert binary mask to Run-Length Encoding (RLE) for efficient storage.
        
        RLE encodes consecutive runs of 0s and 1s, significantly reducing
        data size for sparse binary masks.
        """
        pixels = mask.flatten()
        runs = []
        current_val = pixels[0]
        run_length = 1

        for i in range(1, len(pixels)):
            if pixels[i] == current_val:
                run_length += 1
            else:
                runs.append(run_length)
                current_val = pixels[i]
                run_length = 1
        runs.append(run_length)

        return {
            "counts": runs,
            "size": list(mask.shape),
            "start_value": int(pixels[0]),
        }

    def visualize(self, image: np.ndarray, outputs: Dict) -> np.ndarray:
        """
        Generate visualization using Detectron2's built-in Visualizer.
        
        Returns:
            Annotated image with boxes, masks, and labels drawn
        """
        self.ensure_loaded()

        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog

        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        # Re-run prediction to get raw Detectron2 output for visualization
        raw_outputs = self.predictor(image)

        v = Visualizer(
            image[:, :, ::-1],  # Convert BGR to RGB
            metadata=metadata,
            scale=1.0,
        )
        out = v.draw_instance_predictions(raw_outputs["instances"].to("cpu"))

        return out.get_image()[:, :, ::-1]  # Convert back to BGR
