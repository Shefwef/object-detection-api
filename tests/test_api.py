"""
Tests for the FastAPI application endpoints.

Uses httpx AsyncClient to test API endpoints without running the server.
Tests are structured to verify the API contract (routes, validation, responses)
independently of model weights (using mocks where needed).
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import numpy as np
import json

from app.main import app


# ─── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image for testing."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [0, 255, 0]  # Green square
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def mock_yolo_result():
    """Mock YOLO detection result."""
    return {
        "model": "yolov8",
        "detections": [
            {
                "id": 0,
                "bbox": [30.0, 30.0, 70.0, 70.0],
                "confidence": 0.92,
                "class_id": 0,
                "class_name": "person",
            }
        ],
        "count": 1,
        "image_shape": [100, 100],
        "metadata": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "device": "cpu",
            "model_variant": "yolov8n.pt",
        },
    }


# ─── Root & Health Tests ──────────────────────────────────

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint returns API info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "models" in data
    assert len(data["models"]) == 4


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check returns model status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "yolov8" in data["models"]
    assert "detectron2" in data["models"]
    assert "grounding_dino" in data["models"]
    assert "sam" in data["models"]


# ─── YOLO Endpoint Tests ─────────────────────────────────

@pytest.mark.asyncio
async def test_yolo_detect_rejects_invalid_file():
    """Test that uploading a non-image file returns 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/yolo/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"confidence": "0.25"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_yolo_info():
    """Test YOLO model info endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/yolo/info")

    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "is_loaded" in data


# ─── Detectron2 Endpoint Tests ───────────────────────────

@pytest.mark.asyncio
async def test_detectron2_info():
    """Test Detectron2 model info endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/detectron2/info")

    assert response.status_code == 200
    data = response.json()
    assert "Detectron2" in data["model_name"]


# ─── Grounding DINO Endpoint Tests ───────────────────────

@pytest.mark.asyncio
async def test_grounding_dino_detect_requires_text_prompt(sample_image_bytes):
    """Test that Grounding DINO requires a text_prompt parameter."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Missing text_prompt should fail with 422 (validation error)
        response = await client.post(
            "/api/v1/grounding-dino/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_grounding_dino_info():
    """Test Grounding DINO model info endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/grounding-dino/info")

    assert response.status_code == 200


# ─── SAM Endpoint Tests ──────────────────────────────────

@pytest.mark.asyncio
async def test_sam_segment_points_validates_input(sample_image_bytes):
    """Test that SAM segment-points validates JSON input."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sam/segment-points",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={
                "points": "invalid json",
                "labels": "[1]",
            },
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sam_segment_points_mismatched_lengths(sample_image_bytes):
    """Test that SAM rejects mismatched points and labels arrays."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sam/segment-points",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={
                "points": "[[100,200],[300,400]]",     # 2 points
                "labels": "[1]",                        # 1 label — mismatch!
            },
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sam_info():
    """Test SAM model info endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/sam/info")

    assert response.status_code == 200


# ─── Pipeline Endpoint Tests ─────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_requires_text_prompt(sample_image_bytes):
    """Test that the pipeline endpoint requires a text_prompt."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/pipeline/detect-and-segment",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            # Missing text_prompt
        )

    assert response.status_code == 422
