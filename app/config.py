"""
Application configuration management.
Handles environment variables, model paths, and runtime settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App Settings
    APP_NAME: str = "CV Detection Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", str(Path.home() / ".cache" / "cv_models"))
    DEVICE: str = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"

    # YOLO Settings
    YOLO_MODEL_NAME: str = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")  # nano for speed, swap to yolov8m.pt for accuracy
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

    # Detectron2 Settings
    DETECTRON2_CONFIG: str = os.getenv(
        "DETECTRON2_CONFIG",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    DETECTRON2_CONFIDENCE: float = float(os.getenv("DETECTRON2_CONFIDENCE", "0.5"))

    # Grounding DINO Settings
    GROUNDING_DINO_MODEL: str = os.getenv(
        "GROUNDING_DINO_MODEL",
        "IDEA-Research/grounding-dino-tiny"
    )
    GROUNDING_DINO_BOX_THRESHOLD: float = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.35"))
    GROUNDING_DINO_TEXT_THRESHOLD: float = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.25"))

    # SAM Settings
    SAM_MODEL_TYPE: str = os.getenv("SAM_MODEL_TYPE", "vit_b")  # vit_b, vit_l, vit_h
    SAM_CHECKPOINT: str = os.getenv("SAM_CHECKPOINT", "")  # Auto-downloaded if empty

    # Upload Settings
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def get_device() -> str:
    """Determine the best available device (CUDA or CPU)."""
    import torch

    settings = get_settings()
    if settings.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.DEVICE
