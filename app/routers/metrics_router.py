"""
Metrics endpoints - powers the benchmark dashboard.

Every ``/api/v1/**/detect*`` call records a :class:`MetricRecord`.  These
endpoints aggregate those records to expose real-time per-model latency,
throughput, cache-hit rate, and last-seen timestamps.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_metrics_service
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/summary", summary="Per-model performance summary (live)")
async def summary(service: MetricsService = Depends(get_metrics_service)):
    """Aggregate latency + throughput per model since process start."""
    return await service.summary()


@router.get("/recent", summary="Recent inference records")
async def recent(
    model: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(50, ge=1, le=500),
    service: MetricsService = Depends(get_metrics_service),
):
    """Return the last N inference records (optionally filtered by model)."""
    return await service.recent(model=model, limit=limit)


@router.post("/reset", summary="Clear the in-memory metrics ring buffer")
async def reset(service: MetricsService = Depends(get_metrics_service)):
    """Wipe stored metric records - useful for benchmarks and demos."""
    await service.reset()
    return {"status": "cleared"}
