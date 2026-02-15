"""
Grounding DINO - Open-Set Object Detection with Language Grounding.

Grounding DINO (DINO with Grounded Pre-Training) is a groundbreaking model
that combines a Transformer-based detector (DINO) with grounded pre-training
for open-set object detection.

What makes it special:
- **Open-set detection**: Can detect ANY object described by text, not just
  fixed categories like COCO's 80 classes
- **Language grounding**: You provide a text prompt like "red car" or
  "person wearing glasses" and it finds matching objects
- **Zero-shot capability**: Works on novel categories without fine-tuning

Architecture:
1. Image Backbone (Swin Transformer) → extracts visual features
2. Text Backbone (BERT) → extracts text features from the prompt
3. Feature Enhancer → cross-modality fusion (image ↔ text attention)
4. Language-Guided Query Selection → selects relevant queries
5. Cross-Modality Decoder → predicts boxes guided by text

This is one of the most powerful models for the IML project because:
- It can detect domain-specific objects without retraining
- Combined with SAM, it enables "detect anything, segment anything"
- Perfect for building flexible, prompt-driven detection systems

Example prompts:
- "person . car . dog"  (detect multiple categories, separated by '.')
- "red car"  (detect specific attributes)
- "person wearing a helmet"  (detect with context)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import torch

from app.models.base_model import BaseDetectionModel
from app.config import get_settings, get_device

logger = logging.getLogger(__name__)


class GroundingDINODetector(BaseDetectionModel):
    """
    Grounding DINO open-set object detector.
    
    Uses text prompts to detect arbitrary objects in images.
    This is the key model for flexible, prompt-driven detection
    that IML's product likely requires.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(model_name=f"Grounding DINO ({settings.GROUNDING_DINO_MODEL})")
        self.model_id = settings.GROUNDING_DINO_MODEL
        self.default_box_threshold = settings.GROUNDING_DINO_BOX_THRESHOLD
        self.default_text_threshold = settings.GROUNDING_DINO_TEXT_THRESHOLD
        self.device = get_device()
        self.processor = None

    def load_model(self) -> None:
        """
        Load Grounding DINO model from HuggingFace.
        
        Uses the transformers library's AutoModelForZeroShotObjectDetection
        which handles the complex multi-modal architecture automatically.
        """
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        logger.info(f"Loading Grounding DINO: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

        self.model.eval()  # Set to evaluation mode
        logger.info("Grounding DINO loaded successfully")

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str = "object",
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect objects matching a text description.
        
        The text prompt should describe what to look for. Multiple categories
        can be separated by periods: "cat . dog . person"
        
        Args:
            image: Input image (BGR numpy array)
            text_prompt: Text description of objects to detect
            box_threshold: Minimum confidence for box predictions
            text_threshold: Minimum confidence for text-box matching
            
        Returns:
            Dictionary with detections including matched text phrases
        """
        self.ensure_loaded()

        from PIL import Image
        import cv2

        box_thresh = box_threshold or self.default_box_threshold
        text_thresh = text_threshold or self.default_text_threshold

        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess: tokenize text + transform image
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: convert model outputs to boxes, scores, labels
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
        )[0]

        # Parse results
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        # Filter by thresholds
        valid_indices = (scores >= text_thresh) & (boxes is not None)
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # Skip low confidence detections
            if score < text_thresh:
                continue
            
            # Skip boxes below box threshold (bounding box area check)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area < 0.0001:  # Very small boxes
                continue
            
            detection = {
                "id": i,
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(score),
                "label": labels[i],  # The matched text phrase
                "text_prompt": text_prompt,
            }
            detections.append(detection)

        return {
            "model": "grounding_dino",
            "detections": detections,
            "count": len(detections),
            "image_shape": list(image.shape[:2]),
            "text_prompt": text_prompt,
            "metadata": {
                "box_threshold": box_thresh,
                "text_threshold": text_thresh,
                "device": self.device,
                "model_id": self.model_id,
            },
        }

    def get_boxes_for_sam(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Get detection boxes in format suitable for SAM input.
        
        This method enables the Grounding DINO → SAM pipeline:
        1. Grounding DINO detects objects by text prompt
        2. Detected boxes are passed to SAM as prompts
        3. SAM generates precise segmentation masks
        
        Args:
            image: Input image
            text_prompt: What to detect
            box_threshold: Confidence threshold
            
        Returns:
            Tuple of (boxes_array, labels, scores)
        """
        results = self.predict(image, text_prompt, box_threshold)

        if not results["detections"]:
            return np.array([]), [], []

        boxes = np.array([d["bbox"] for d in results["detections"]])
        labels = [d["label"] for d in results["detections"]]
        scores = [d["confidence"] for d in results["detections"]]

        return boxes, labels, scores
