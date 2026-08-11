"""
FastAPI dependency providers.

Everything the routers need arrives through :func:`fastapi.Depends`.  The
providers pick the correct concrete backend (in-memory vs Redis vs Mongo)
based on ``Settings`` at process start and return a cached singleton to
subsequent calls.

    router  ->  Depends(get_detection_service)
        DetectionService(IInferenceRepository, IMetricsRepository, ModelFactory)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends

from app.config import Settings, get_settings
from app.repositories.inference_repository import (
    IInferenceRepository,
    InMemoryInferenceRepository,
    RedisInferenceRepository,
)
from app.repositories.metrics_repository import (
    IMetricsRepository,
    InMemoryMetricsRepository,
    MongoMetricsRepository,
)
from app.services.detection_service import DetectionService
from app.services.explainability_service import ExplainabilityService
from app.services.metrics_service import MetricsService
from app.services.pipeline_service import PipelineService


# ─── Cached backend singletons ─────────────────────────────────────────────
#
# Repositories may hold long-lived network connections (Redis, Mongo), so we
# construct them exactly once per process.


@lru_cache
def _inference_repository() -> IInferenceRepository:
    settings = get_settings()
    if settings.REDIS_URL:
        try:
            import redis.asyncio as redis  # type: ignore
            client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            return RedisInferenceRepository(client, ttl_seconds=settings.INFERENCE_CACHE_TTL)
        except Exception:
            # If redis isn't installed or the URL is unreachable we degrade
            # gracefully to in-memory - the API remains functional.
            pass
    return InMemoryInferenceRepository(
        ttl_seconds=settings.INFERENCE_CACHE_TTL,
        max_entries=settings.INFERENCE_CACHE_MAX_ENTRIES,
    )


@lru_cache
def _metrics_repository() -> IMetricsRepository:
    settings = get_settings()
    if settings.MONGO_URL:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
            client = AsyncIOMotorClient(settings.MONGO_URL)
            col = client[settings.MONGO_DB_NAME][settings.MONGO_METRICS_COLLECTION]
            return MongoMetricsRepository(col)
        except Exception:
            pass
    return InMemoryMetricsRepository(window_size=settings.METRICS_WINDOW_SIZE)


# ─── Public FastAPI dependencies ───────────────────────────────────────────


def get_inference_repository() -> IInferenceRepository:
    return _inference_repository()


def get_metrics_repository() -> IMetricsRepository:
    return _metrics_repository()


def get_detection_service(
    inference_repo: IInferenceRepository = Depends(get_inference_repository),
    metrics_repo: IMetricsRepository = Depends(get_metrics_repository),
) -> DetectionService:
    return DetectionService(inference_repo=inference_repo, metrics_repo=metrics_repo)


def get_pipeline_service(
    metrics_repo: IMetricsRepository = Depends(get_metrics_repository),
) -> PipelineService:
    return PipelineService(metrics_repo=metrics_repo)


@lru_cache
def _explainability_service() -> ExplainabilityService:
    return ExplainabilityService()


def get_explainability_service() -> ExplainabilityService:
    return _explainability_service()


def get_metrics_service(
    metrics_repo: IMetricsRepository = Depends(get_metrics_repository),
) -> MetricsService:
    return MetricsService(repo=metrics_repo)


# Convenient re-export for callers that want raw settings
def app_settings() -> Settings:
    return get_settings()


def reset_dependency_cache() -> None:
    """Drop cached singletons - intended for tests."""
    _inference_repository.cache_clear()
    _metrics_repository.cache_clear()
    _explainability_service.cache_clear()
