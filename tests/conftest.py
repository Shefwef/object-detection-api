"""Shared pytest fixtures."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image_bytes() -> bytes:
    """A tiny valid JPEG containing a coloured square."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [0, 255, 0]
    ok, buffer = cv2.imencode(".jpg", img)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def sample_image_array() -> np.ndarray:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [0, 255, 0]
    return img


@pytest.fixture(autouse=True)
def _reset_di_cache():
    """Reset cached DI singletons between tests so fixtures don't leak state."""
    from app.dependencies import reset_dependency_cache

    reset_dependency_cache()
    yield
    reset_dependency_cache()
