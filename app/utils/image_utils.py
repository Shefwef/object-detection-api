"""
Image processing utilities.

Two-tier API:

* :func:`read_upload_bytes` + :func:`decode_image_bytes` - split so callers
  can pass the raw bytes to the cache-key hash without decoding twice.
* :func:`load_image_from_upload` - convenience wrapper kept for backwards
  compatibility with code paths that only need the numpy array.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─── Public helpers ───────────────────────────────────────────────────────


async def read_upload_bytes(file: UploadFile) -> bytes:
    """Read + validate an uploaded file, returning the raw bytes."""
    settings = get_settings()
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large: {len(content)} bytes. "
                f"Max: {settings.MAX_IMAGE_SIZE} bytes"
            ),
        )
    return content


def decode_image_bytes(content: bytes, filename: Optional[str] = None) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array (OpenCV format)."""
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None (unsupported image)")
        logger.debug("Decoded image: %s shape=%s", filename or "<upload>", image.shape)
        return image
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Convenience wrapper: read + decode in one step."""
    content = await read_upload_bytes(file)
    return decode_image_bytes(content, filename=file.filename)


def image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    if format.upper() == "PNG":
        _, buffer = cv2.imencode(".png", image)
    else:
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def resize_image(
    image: np.ndarray,
    max_size: int = 1024,
    keep_aspect: bool = True,
) -> np.ndarray:
    """Shrink an image to fit within ``max_size`` on the longest edge."""
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
