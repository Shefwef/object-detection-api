"""
Application configuration.

All runtime knobs are collected here and hydrated from environment variables /
``.env``.  ``get_settings()`` is ``lru_cache``-d so import cost is amortized.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Application ---------------------------------------------------------
    APP_NAME: str = "CV Detection Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # -- Server --------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "*"  # comma-separated list, "*" allows any origin

    # -- Model runtime -------------------------------------------------------
    MODEL_CACHE_DIR: str = str(Path.home() / ".cache" / "cv_models")
    DEVICE: str = "auto"  # "auto" | "cuda" | "cpu"

    # -- YOLOv8 --------------------------------------------------------------
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45

    # -- Detectron2 ----------------------------------------------------------
    DETECTRON2_CONFIG: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    DETECTRON2_CONFIDENCE: float = 0.5

    # -- Grounding DINO ------------------------------------------------------
    GROUNDING_DINO_MODEL: str = "IDEA-Research/grounding-dino-tiny"
    GROUNDING_DINO_BOX_THRESHOLD: float = 0.35
    GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25

    # -- SAM -----------------------------------------------------------------
    SAM_MODEL_TYPE: str = "vit_b"
    SAM_CHECKPOINT: str = ""

    # -- Upload constraints --------------------------------------------------
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # -- Logging -------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False  # emit structured JSON logs via structlog

    # -- Cache backend -------------------------------------------------------
    CACHE_ENABLED: bool = True
    INFERENCE_CACHE_TTL: int = 3600  # seconds
    INFERENCE_CACHE_MAX_ENTRIES: int = 512  # only used by in-memory backend
    REDIS_URL: Optional[str] = None  # if set, RedisInferenceRepository is used

    # -- Metrics backend -----------------------------------------------------
    METRICS_ENABLED: bool = True
    METRICS_WINDOW_SIZE: int = 500  # in-memory ring buffer per model
    MONGO_URL: Optional[str] = None
    MONGO_DB_NAME: str = "cv_detection"
    MONGO_METRICS_COLLECTION: str = "inference_metrics"

    # -- Authentication ------------------------------------------------------
    AUTH_ENABLED: bool = False
    # Comma-separated list of accepted API keys.  When AUTH_ENABLED is False
    # the header is not checked at all - the demo remains open.
    API_KEYS: str = "demo-key-12345"
    RATE_LIMIT_PER_MINUTE: int = 60

    # -- Convenience derived values -----------------------------------------
    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_device() -> str:
    """Resolve ``settings.DEVICE`` to a concrete torch device string."""
    settings = get_settings()
    if settings.DEVICE != "auto":
        return settings.DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


