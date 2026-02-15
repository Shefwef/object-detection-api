"""
Image processing utilities.
Handles image loading, validation, and format conversions.
"""

import cv2
import numpy as np
from io import BytesIO
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
from PIL import Image
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Load and validate an uploaded image file.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Image as BGR numpy array (OpenCV format)
        
    Raises:
        HTTPException: If file is invalid, too large, or wrong format
    """
    settings = get_settings()

    # Validate file extension
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content)} bytes. Max: {settings.MAX_IMAGE_SIZE} bytes"
        )

    # Decode image
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        logger.info(f"Loaded image: {filename}, shape: {image.shape}")
        return image

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}"
        )


def image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    """Convert numpy image to bytes for API response."""
    if format.upper() == "JPEG":
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elif format.upper() == "PNG":
        _, buffer = cv2.imencode(".png", image)
    else:
        _, buffer = cv2.imencode(".jpg", image)

    return buffer.tobytes()


def resize_image(
    image: np.ndarray,
    max_size: int = 1024,
    keep_aspect: bool = True,
) -> np.ndarray:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    Useful for limiting input size to models for faster inference.
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    if keep_aspect:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = new_h = max_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
