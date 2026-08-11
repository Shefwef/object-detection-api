"""Endpoint tests for the routers introduced by the v2 refactor."""

from __future__ import annotations

import base64

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


# ─── Metrics endpoints ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_summary_returns_object():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v1/metrics/summary")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


@pytest.mark.asyncio
async def test_metrics_recent_accepts_limit():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v1/metrics/recent?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_metrics_reset_clears_records():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/metrics/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"


# ─── Base64 detect ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_base64_detect_rejects_bad_payload():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/yolo/detect-base64",
            json={"image": "not-real-base64!!!", "confidence": 0.25, "iou_threshold": 0.45},
        )
    assert r.status_code in (400, 422)


# ─── Auth middleware ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_docs_endpoint_stays_public_even_with_auth_middleware():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/openapi.json")
    assert r.status_code == 200


# ─── Root endpoint exposes feature flags ─────────────────────────────────


@pytest.mark.asyncio
async def test_root_exposes_features():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body
    assert "auth_enabled" in body["features"]
    assert "cache_enabled" in body["features"]
    assert "metrics_enabled" in body["features"]
