"""
SAM - Segment Anything Model Wrapper.

SAM (Segment Anything Model) by Meta AI is a foundation model for image
segmentation that can segment ANY object in ANY image with various prompts.

Key Concepts:
- **Foundation model**: Trained on 11M images with 1B+ masks (SA-1B dataset)
- **Promptable**: Accepts points, boxes, or text as input prompts
- **Zero-shot transfer**: Works on new images/domains without fine-tuning
- **Three components**:
  1. Image Encoder (ViT-based): Encodes the image once
  2. Prompt Encoder: Encodes points/boxes/masks/text prompts
  3. Mask Decoder: Lightweight decoder that produces masks from embeddings

Why SAM is powerful:
- Generates high-quality segmentation masks for any object
- Can be prompted with bounding boxes from other detectors (like Grounding DINO)
- Image embedding is computed once, then any number of prompts can be used fast
- Three output masks per prompt (whole, part, subpart) with confidence scores

SAM + Grounding DINO Pipeline:
1. Grounding DINO: "Find all cats" → returns bounding boxes
2. SAM: Takes those boxes → produces pixel-perfect segmentation masks
This combination gives you open-vocabulary instance segmentation!

Model Variants:
- vit_h: Largest, most accurate (2.4GB)
- vit_l: Large (1.2GB)  
- vit_b: Base, fastest (375MB) ← default for development
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import torch

from app.models.base_model import BaseDetectionModel
from app.config import get_settings, get_device

logger = logging.getLogger(__name__)

# SAM checkpoint URLs
SAM_CHECKPOINTS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class SAMSegmenter(BaseDetectionModel):
    """
    Segment Anything Model (SAM) wrapper.
    
    Supports multiple prompting modes:
    - Automatic mask generation (segment everything)
    - Point prompts (click on objects)
    - Box prompts (from other detectors like Grounding DINO)
    - Combined point + box prompts
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(model_name=f"SAM ({settings.SAM_MODEL_TYPE})")
        self.model_type = settings.SAM_MODEL_TYPE
        self.checkpoint_path = settings.SAM_CHECKPOINT
        self.device = get_device()
        self.predictor = None
        self.mask_generator = None

    def load_model(self) -> None:
        """
        Load SAM model and create both predictor and mask generator.
        
        - SamPredictor: For prompted segmentation (points, boxes)
        - SamAutomaticMaskGenerator: For automatic segmentation of everything
        """
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        import os
        import urllib.request

        # Download checkpoint if not provided
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            self.checkpoint_path = self._download_checkpoint()

        logger.info(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")

        # Build SAM model from registry
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)

        # Create predictor for prompted segmentation
        self.predictor = SamPredictor(sam)

        # Create automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,             # Grid density for auto-segmentation
            pred_iou_thresh=0.86,           # Filter low-quality masks
            stability_score_thresh=0.92,    # Filter unstable masks
            min_mask_region_area=100,       # Minimum mask size in pixels
        )

        self.model = sam
        logger.info("SAM model loaded successfully")

    def _download_checkpoint(self) -> str:
        """Download SAM checkpoint if not available locally."""
        import os
        import urllib.request
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "sam_checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = f"sam_{self.model_type}.pth"
        filepath = cache_dir / filename

        if not filepath.exists():
            url = SAM_CHECKPOINTS.get(self.model_type)
            if not url:
                raise ValueError(f"Unknown SAM model type: {self.model_type}")

            logger.info(f"Downloading SAM checkpoint: {url}")
            urllib.request.urlretrieve(url, str(filepath))
            logger.info(f"Downloaded SAM checkpoint to {filepath}")

        return str(filepath)

    def predict(
        self,
        image: np.ndarray,
        mode: str = "auto",
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SAM segmentation with various prompt modes.
        
        Modes:
        - "auto": Automatically segment everything in the image
        - "points": Use point prompts (click locations)
        - "boxes": Use bounding box prompts
        - "points_and_boxes": Combined prompts
        
        Args:
            image: Input image (BGR numpy array)
            mode: Segmentation mode
            point_coords: Nx2 array of (x, y) point coordinates
            point_labels: N array of labels (1=foreground, 0=background)
            boxes: Mx4 array of bounding boxes [x1, y1, x2, y2]
            multimask_output: Return 3 masks per prompt (True) or 1 (False)
            
        Returns:
            Dictionary with segmentation masks and metadata
        """
        self.ensure_loaded()

        if mode == "auto":
            return self._auto_segment(image)
        elif mode == "points":
            return self._segment_with_points(image, point_coords, point_labels, multimask_output)
        elif mode == "boxes":
            return self._segment_with_boxes(image, boxes, multimask_output)
        elif mode == "points_and_boxes":
            return self._segment_with_combined(image, point_coords, point_labels, boxes, multimask_output)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'points', 'boxes', or 'points_and_boxes'")

    def _auto_segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Automatically segment everything in the image.
        
        Uses a grid of point prompts to discover all objects.
        Returns masks sorted by area (largest first).
        """
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)

        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        segments = []
        for i, mask_data in enumerate(masks):
            segment = {
                "id": i,
                "area": int(mask_data["area"]),
                "bbox": mask_data["bbox"],  # [x, y, w, h] format
                "predicted_iou": float(mask_data["predicted_iou"]),
                "stability_score": float(mask_data["stability_score"]),
                "mask_shape": list(mask_data["segmentation"].shape),
            }
            segments.append(segment)

        return {
            "model": "sam",
            "mode": "auto",
            "segments": segments,
            "count": len(segments),
            "image_shape": list(image.shape[:2]),
            "metadata": {
                "model_type": self.model_type,
                "device": self.device,
            },
        }

    def _segment_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Segment using point prompts.
        
        Points are (x, y) coordinates on the image.
        Labels: 1 = foreground (include), 0 = background (exclude)
        
        SAM returns up to 3 masks per prompt at different granularity levels:
        - Whole object
        - Part of object
        - Sub-part of object
        """
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

        segments = []
        for i in range(len(masks)):
            segment = {
                "id": i,
                "score": float(scores[i]),
                "mask_shape": list(masks[i].shape),
                "area": int(masks[i].sum()),
            }
            segments.append(segment)

        return {
            "model": "sam",
            "mode": "points",
            "segments": segments,
            "count": len(segments),
            "image_shape": list(image.shape[:2]),
            "prompts": {
                "point_coords": point_coords.tolist(),
                "point_labels": point_labels.tolist(),
            },
            "metadata": {
                "model_type": self.model_type,
                "device": self.device,
                "multimask_output": multimask_output,
            },
        }

    def _segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        multimask_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Segment using bounding box prompts.
        
        This is the key method for the Grounding DINO → SAM pipeline:
        Grounding DINO provides boxes, SAM generates precise masks.
        
        Args:
            boxes: Mx4 array of [x1, y1, x2, y2] bounding boxes
        """
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        all_segments = []

        # Process each box
        for box_idx, box in enumerate(boxes):
            input_box = np.array(box)

            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=multimask_output,
            )

            # Take the best mask (highest score)
            best_idx = np.argmax(scores)
            segment = {
                "id": box_idx,
                "input_box": box.tolist() if isinstance(box, np.ndarray) else box,
                "score": float(scores[best_idx]),
                "mask_shape": list(masks[best_idx].shape),
                "area": int(masks[best_idx].sum()),
            }
            all_segments.append(segment)

        return {
            "model": "sam",
            "mode": "boxes",
            "segments": all_segments,
            "count": len(all_segments),
            "image_shape": list(image.shape[:2]),
            "metadata": {
                "model_type": self.model_type,
                "device": self.device,
                "num_input_boxes": len(boxes),
            },
        }

    def _segment_with_combined(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        boxes: np.ndarray,
        multimask_output: bool = False,
    ) -> Dict[str, Any]:
        """Segment using both points and boxes as prompts."""
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes[0] if len(boxes) == 1 else boxes,
            multimask_output=multimask_output,
        )

        segments = []
        for i in range(len(masks)):
            segment = {
                "id": i,
                "score": float(scores[i]),
                "mask_shape": list(masks[i].shape),
                "area": int(masks[i].sum()),
            }
            segments.append(segment)

        return {
            "model": "sam",
            "mode": "points_and_boxes",
            "segments": segments,
            "count": len(segments),
            "image_shape": list(image.shape[:2]),
            "metadata": {
                "model_type": self.model_type,
                "device": self.device,
            },
        }

    def get_masks_raw(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Get raw binary masks for given boxes (for pipeline use).
        
        Returns list of boolean numpy arrays (one per box).
        Used by the Grounding DINO + SAM pipeline.
        """
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        all_masks = []
        for box in boxes:
            masks, scores, _ = self.predictor.predict(
                box=np.array(box),
                multimask_output=False,
            )
            all_masks.append(masks[0])  # Best mask

        return all_masks
