"""
Factory + registry for detection models.

The factory decouples the service layer from concrete model classes.  New
models can be plugged in via :meth:`ModelFactory.register` without touching
callers - open for extension, closed for modification.

    ModelFactory.create(ModelType.YOLO) -> YOLODetector
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Type
import logging

from app.models.base_model import BaseDetectionModel

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Canonical identifiers for every supported model."""

    YOLO = "yolov8"
    DETECTRON2 = "detectron2"
    GROUNDING_DINO = "grounding_dino"
    SAM = "sam"


class ModelFactory:
    """Central registry that maps a :class:`ModelType` to a concrete class."""

    _registry: Dict[ModelType, Type[BaseDetectionModel]] = {}
    _singletons: Dict[ModelType, BaseDetectionModel] = {}

    # -- Registration --------------------------------------------------------

    @classmethod
    def register(
        cls, model_type: ModelType, model_class: Type[BaseDetectionModel]
    ) -> None:
        cls._registry[model_type] = model_class
        logger.debug("Registered %s -> %s", model_type, model_class.__name__)

    @classmethod
    def registered_types(cls) -> list[ModelType]:
        return list(cls._registry.keys())

    # -- Construction --------------------------------------------------------

    @classmethod
    def create(cls, model_type: ModelType) -> BaseDetectionModel:
        """Return a fresh instance of the requested model."""
        model_class = cls._registry.get(model_type)
        if model_class is None:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Registered: {[t.value for t in cls.registered_types()]}"
            )
        return model_class()

    @classmethod
    def get_or_create(cls, model_type: ModelType) -> BaseDetectionModel:
        """Return a lazily-cached singleton for the requested model.

        This is the preferred entry point for the API layer: model instances
        are heavy, and we only want one per process.
        """
        instance = cls._singletons.get(model_type)
        if instance is None:
            instance = cls.create(model_type)
            cls._singletons[model_type] = instance
        return instance

    # -- Test hooks ----------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Drop every cached singleton. Intended for tests."""
        cls._singletons.clear()


def _register_default_models() -> None:
    """Populate the registry with the four models shipped by this project.

    Imports are performed lazily inside the function to avoid pulling heavy
    ML dependencies at module import time (crucial for docs / unit tests).
    """
    from app.models.yolo_model import YOLODetector
    from app.models.detectron2_model import Detectron2Detector
    from app.models.grounding_dino import GroundingDINODetector
    from app.models.sam_model import SAMSegmenter

    ModelFactory.register(ModelType.YOLO, YOLODetector)
    ModelFactory.register(ModelType.DETECTRON2, Detectron2Detector)
    ModelFactory.register(ModelType.GROUNDING_DINO, GroundingDINODetector)
    ModelFactory.register(ModelType.SAM, SAMSegmenter)


_register_default_models()
