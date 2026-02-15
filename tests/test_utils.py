"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
import cv2

from app.utils.visualization import draw_detections, draw_masks, get_color
from app.utils.image_utils import image_to_bytes, resize_image


# ─── Visualization Tests ──────────────────────────────────

def test_get_color_cycles():
    """Colors cycle through the palette without crashing."""
    for i in range(100):
        color = get_color(i)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_draw_detections_returns_annotated_image():
    """draw_detections should return an image of same shape."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        {"id": 0, "bbox": [10, 10, 50, 50], "class_name": "cat", "confidence": 0.9},
        {"id": 1, "bbox": [60, 60, 120, 120], "label": "dog", "confidence": 0.8},
    ]

    result = draw_detections(image, detections)

    assert result.shape == image.shape
    # Annotated image should differ from blank
    assert not np.array_equal(result, image)


def test_draw_detections_handles_empty():
    """draw_detections with empty list should return copy of original."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = draw_detections(image, [])
    assert np.array_equal(result, image)


def test_draw_masks_overlay():
    """draw_masks should overlay colored regions."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    result = draw_masks(image, [mask], alpha=0.5)

    assert result.shape == image.shape
    # The masked region should have color (not black)
    assert result[50, 50].sum() > 0


# ─── Image Utils Tests ───────────────────────────────────

def test_image_to_bytes_jpeg():
    """Convert numpy image to JPEG bytes."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = image_to_bytes(image, format="JPEG")

    assert isinstance(result, bytes)
    assert len(result) > 0
    # JPEG magic bytes
    assert result[:2] == b'\xff\xd8'


def test_image_to_bytes_png():
    """Convert numpy image to PNG bytes."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = image_to_bytes(image, format="PNG")

    assert isinstance(result, bytes)
    # PNG magic bytes
    assert result[:4] == b'\x89PNG'


def test_resize_image_downscales():
    """resize_image should downscale large images."""
    image = np.zeros((2000, 3000, 3), dtype=np.uint8)
    result = resize_image(image, max_size=1024)

    assert max(result.shape[:2]) <= 1024
    # Aspect ratio should be preserved
    original_ratio = 2000 / 3000
    result_ratio = result.shape[0] / result.shape[1]
    assert abs(original_ratio - result_ratio) < 0.01


def test_resize_image_no_change_if_small():
    """resize_image should not change small images."""
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    result = resize_image(image, max_size=1024)

    assert result.shape == image.shape
