"""
Repository for per-model inference metrics.

Backing the metrics dashboard: every inference call records (model, latency,
detection_count, timestamp).  Aggregation on read produces averages, p50 /
p95 latency, request counts, and last-seen timestamps.

Two implementations:

* :class:`InMemoryMetricsRepository` - bounded deque per model, no external
  dependencies. Default for local development.
* :class:`MongoMetricsRepository` - persists every record and aggregates
  server-side. Enabled when ``MONGO_URL`` is configured and ``motor`` is
  installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Deque, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Data class ────────────────────────────────────────────────────────────


@dataclass
class MetricRecord:
    model: str
    latency_ms: float
    detection_count: int
    cached: bool
    timestamp: str  # ISO 8601 UTC

    @classmethod
    def now(cls, model: str, latency_ms: float, detection_count: int, cached: bool) -> "MetricRecord":
        return cls(
            model=model,
            latency_ms=float(latency_ms),
            detection_count=int(detection_count),
            cached=bool(cached),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ─── Interface ─────────────────────────────────────────────────────────────


class IMetricsRepository(ABC):
    """Contract for anything that stores per-model inference metrics."""

    @abstractmethod
    async def record(self, record: MetricRecord) -> None: ...

    @abstractmethod
    async def summary(self) -> Dict[str, Dict[str, Any]]: ...

    @abstractmethod
    async def recent(self, model: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def clear(self) -> None: ...


# ─── Aggregation helper ────────────────────────────────────────────────────


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[k], 2)


def _aggregate(records: List[MetricRecord]) -> Dict[str, Any]:
    if not records:
        return {
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_detections": 0.0,
            "last_seen": None,
        }
    latencies = [r.latency_ms for r in records]
    cache_hits = sum(1 for r in records if r.cached)
    return {
        "total_requests": len(records),
        "cache_hit_rate": round(cache_hits / len(records), 3),
        "avg_latency_ms": round(mean(latencies), 2),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "avg_detections": round(mean(r.detection_count for r in records), 2),
        "last_seen": max(r.timestamp for r in records),
    }


# ─── In-memory implementation ──────────────────────────────────────────────


class InMemoryMetricsRepository(IMetricsRepository):
    def __init__(self, window_size: int = 500) -> None:
        self._window = window_size
        self._records: Dict[str, Deque[MetricRecord]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    async def record(self, record: MetricRecord) -> None:
        self._records[record.model].append(record)

    async def summary(self) -> Dict[str, Dict[str, Any]]:
        return {model: _aggregate(list(bucket)) for model, bucket in self._records.items()}

    async def recent(self, model: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if model:
            bucket = list(self._records.get(model, ()))
        else:
            bucket = [r for records in self._records.values() for r in records]
            bucket.sort(key=lambda r: r.timestamp, reverse=True)
        return [asdict(r) for r in bucket[-limit:][::-1]]

    async def clear(self) -> None:
        self._records.clear()


# ─── MongoDB implementation (optional) ─────────────────────────────────────


class MongoMetricsRepository(IMetricsRepository):
    def __init__(self, collection: Any) -> None:
        self._col = collection

    async def record(self, record: MetricRecord) -> None:
        try:
            await self._col.insert_one(asdict(record))
        except Exception:  # pragma: no cover
            logger.exception("Mongo insert failed - metric dropped")

    async def summary(self) -> Dict[str, Dict[str, Any]]:
        pipeline = [
            {
                "$group": {
                    "_id": "$model",
                    "total_requests": {"$sum": 1},
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                    "avg_detections": {"$avg": "$detection_count"},
                    "cache_hits": {"$sum": {"$cond": ["$cached", 1, 0]}},
                    "last_seen": {"$max": "$timestamp"},
                }
            }
        ]
        try:
            docs = await self._col.aggregate(pipeline).to_list(length=None)
        except Exception:  # pragma: no cover
            logger.exception("Mongo aggregate failed")
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            total = int(doc["total_requests"])
            out[doc["_id"]] = {
                "total_requests": total,
                "cache_hit_rate": round((doc.get("cache_hits") or 0) / total, 3) if total else 0.0,
                "avg_latency_ms": round(float(doc.get("avg_latency_ms") or 0), 2),
                "avg_detections": round(float(doc.get("avg_detections") or 0), 2),
                "last_seen": doc.get("last_seen"),
                # p50/p95 require a second pass and are omitted for the Mongo
                # backend to keep this endpoint fast; the in-memory backend
                # exposes them for local benchmarking.
                "p50_latency_ms": None,
                "p95_latency_ms": None,
            }
        return out

    async def recent(self, model: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = {"model": model} if model else {}
        try:
            cursor = self._col.find(query).sort("timestamp", -1).limit(limit)
            return [
                {k: v for k, v in doc.items() if k != "_id"}
                async for doc in cursor
            ]
        except Exception:  # pragma: no cover
            logger.exception("Mongo find failed")
            return []

    async def clear(self) -> None:
        try:
            await self._col.delete_many({})
        except Exception:  # pragma: no cover
            logger.exception("Mongo clear failed")
