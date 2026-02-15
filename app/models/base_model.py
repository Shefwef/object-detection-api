"""
Abstract base class for all detection/segmentation models.
Ensures a unified interface across YOLO, Detectron2, Grounding DINO, and SAM.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseDetectionModel(ABC):
    """
    Abstract base class that all detection models must implement.
    
    This abstraction allows the API layer to interact with any model
    through the same interface, making it easy to swap or add models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._is_loaded = False
        logger.info(f"Initialized {model_name} (not yet loaded)")

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights into memory.
        Called lazily on first inference request to conserve resources.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            **kwargs: Model-specific parameters (confidence, prompts, etc.)
            
        Returns:
            Dictionary containing:
                - 'detections': List of detection results
                - 'metadata': Model-specific metadata
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded in memory."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Load model if not already loaded (lazy loading pattern)."""
        if not self._is_loaded:
            logger.info(f"Lazy loading {self.model_name}...")
            self.load_model()
            self._is_loaded = True
            logger.info(f"{self.model_name} loaded successfully")

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about this model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "type": self.__class__.__name__,
        }
