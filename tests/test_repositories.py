"""Unit tests for the repository layer (no model weights required)."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.repositories.inference_repository import (
    InMemoryInferenceRepository,
    compute_image_hash,
)
from app.repositories.metrics_repository import (
    InMemoryMetricsRepository,
    MetricRecord,
)


# ─── Inference cache ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inference_cache_hit_and_miss():
    repo = InMemoryInferenceRepository(ttl_seconds=60, max_entries=10)
    key = compute_image_hash(b"image-bytes")

    assert await repo.get(key, "yolov8") is None

    await repo.set(key, "yolov8", {"detections": [{"id": 1}]})
    assert (await repo.get(key, "yolov8")) == {"detections": [{"id": 1}]}
    assert await repo.size() == 1


@pytest.mark.asyncio
async def test_inference_cache_respects_ttl():
    repo = InMemoryInferenceRepository(ttl_seconds=0, max_entries=10)
    await repo.set("h", "m", {"k": 1})
    time.sleep(0.01)  # ensure clock moves past zero-second TTL
    assert await repo.get("h", "m") is None


@pytest.mark.asyncio
async def test_inference_cache_evicts_lru():
    repo = InMemoryInferenceRepository(ttl_seconds=60, max_entries=2)
    await repo.set("a", "m", {"v": 1})
    await repo.set("b", "m", {"v": 2})
    await repo.set("c", "m", {"v": 3})  # evicts "a"
    assert await repo.get("a", "m") is None
    assert await repo.get("b", "m") == {"v": 2}
    assert await repo.get("c", "m") == {"v": 3}


def test_image_hash_is_stable():
    assert compute_image_hash(b"abc") == compute_image_hash(b"abc")
    assert compute_image_hash(b"abc") != compute_image_hash(b"abcd")


# ─── Metrics repository ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_repo_aggregates_averages():
    repo = InMemoryMetricsRepository(window_size=100)

    await repo.record(MetricRecord.now("yolov8", 100.0, 5, cached=False))
    await repo.record(MetricRecord.now("yolov8", 200.0, 3, cached=True))
    await repo.record(MetricRecord.now("detectron2", 800.0, 4, cached=False))

    summary = await repo.summary()
    assert set(summary.keys()) == {"yolov8", "detectron2"}
    assert summary["yolov8"]["total_requests"] == 2
    assert summary["yolov8"]["avg_latency_ms"] == 150.0
    assert summary["yolov8"]["cache_hit_rate"] == 0.5
    assert summary["detectron2"]["total_requests"] == 1


@pytest.mark.asyncio
async def test_metrics_repo_recent_filters_by_model():
    repo = InMemoryMetricsRepository(window_size=100)
    await repo.record(MetricRecord.now("yolov8", 10.0, 1, cached=False))
    await repo.record(MetricRecord.now("sam", 20.0, 2, cached=False))

    only_yolo = await repo.recent(model="yolov8", limit=10)
    assert len(only_yolo) == 1
    assert only_yolo[0]["model"] == "yolov8"


@pytest.mark.asyncio
async def test_metrics_percentiles_are_ordered():
    repo = InMemoryMetricsRepository(window_size=100)
    for latency in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        await repo.record(MetricRecord.now("yolov8", float(latency), 1, cached=False))
    summary = await repo.summary()
    row = summary["yolov8"]
    assert row["p50_latency_ms"] <= row["p95_latency_ms"]
    assert row["p95_latency_ms"] <= 100
