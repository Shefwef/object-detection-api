"""
Factory + registry for detection models.

The factory decouples the service layer from concrete model classes.  New
models can be plugged in via :meth:`ModelFactory.register` without touching
callers - open for extension, closed for modification.

    ModelFactory.create(ModelType.YOLO) -> YOLODetector

Registration is *tolerant*: if the heavy ML dependency for a particular
model isn't installed (torch, transformers, segment-anything, ...) we log
a warning and skip that model instead of crashing at import time.  This
keeps unit tests, CI, and stripped-down deployments happy while still
loading everything on a full runtime.
"""

from __future__ import annotations

import importlib
import logging
from enum import Enum
from typing import Dict, Tuple, Type

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
                f"Unknown or unavailable model type: {model_type!r}. "
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


# ─── Default registrations (tolerant of missing deps) ─────────────────────


_DEFAULT_REGISTRATIONS: Tuple[Tuple[ModelType, str, str], ...] = (
    (ModelType.YOLO, "app.models.yolo_model", "YOLODetector"),
    (ModelType.DETECTRON2, "app.models.detectron2_model", "Detectron2Detector"),
    (ModelType.GROUNDING_DINO, "app.models.grounding_dino", "GroundingDINODetector"),
    (ModelType.SAM, "app.models.sam_model", "SAMSegmenter"),
)


def _register_default_models() -> None:
    """Try to register every shipped model; skip any that can't import.

    Rationale: individual model modules pull in heavy ML libraries (torch,
    transformers, segment-anything, detectron2). If one is missing on the
    host (e.g. the CI runner, a stripped Docker image, or a laptop that
    hasn't set up CUDA yet), we don't want that to bring down the whole
    factory. The affected endpoints will still fail on invocation with a
    clear error, but the API - and every other model - keeps working.
    """
    for model_type, module_path, class_name in _DEFAULT_REGISTRATIONS:
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            ModelFactory.register(model_type, model_class)
        except Exception as exc:  # noqa: BLE001 - we truly want to swallow all
            logger.warning(
                "Skipping %s registration (missing dependency?): %s",
                model_type.value,
                exc,
            )


_register_default_models()
