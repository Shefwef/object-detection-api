"""
Application entry point.

Wires together:

* Structured logging (:mod:`app.core.logging`)
* CORS + API-key + rate-limit middleware (:mod:`app.middleware.auth`)
* Feature routers (YOLO, Detectron2, Grounding DINO, SAM, pipeline,
  Grad-CAM explainability, metrics)
* Lifespan hooks (startup / shutdown)

The application factory pattern is used so tests can construct isolated
apps without re-importing this module.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.logging import configure_logging
from app.middleware.auth import register_middleware
from app.models.model_factory import ModelFactory
from app.routers import (
    detectron2_router,
    explain_router,
    grounding_dino_router,
    metrics_router,
    pipeline_router,
    sam_router,
    yolo_router,
)


settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "starting %s v%s (auth=%s, cache=%s, metrics=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.AUTH_ENABLED,
        settings.CACHE_ENABLED,
        settings.METRICS_ENABLED,
    )
    logger.info("models registered: %s", [t.value for t in ModelFactory.registered_types()])
    logger.info("models will lazy-load on first request")
    yield
    logger.info("shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ─── CORS ──────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Auth + rate limit ─────────────────────────────
    register_middleware(app)

    # ─── Feature routers ───────────────────────────────
    app.include_router(yolo_router.router, prefix="/api/v1")
    app.include_router(detectron2_router.router, prefix="/api/v1")
    app.include_router(grounding_dino_router.router, prefix="/api/v1")
    app.include_router(sam_router.router, prefix="/api/v1")
    app.include_router(pipeline_router.router, prefix="/api/v1")
    app.include_router(explain_router.router, prefix="/api/v1")
    app.include_router(metrics_router.router, prefix="/api/v1")

    _install_root_endpoints(app)
    return app


def _install_root_endpoints(app: FastAPI) -> None:
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "models": [t.value for t in ModelFactory.registered_types()],
            "features": {
                "auth_enabled": settings.AUTH_ENABLED,
                "cache_enabled": settings.CACHE_ENABLED,
                "metrics_enabled": settings.METRICS_ENABLED,
                "structured_logging": settings.LOG_JSON,
            },
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Report the process + per-model status. Never triggers weight loading."""
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "models": {
                model_type.value: ModelFactory.get_or_create(model_type).get_model_info()
                for model_type in ModelFactory.registered_types()
            },
        }


_DESCRIPTION = """
## Unified Object Detection & Segmentation API

A production-oriented FastAPI service that unifies four state-of-the-art
computer vision models behind a single REST surface:

* **YOLOv8** - real-time single-stage detection (80 COCO classes)
* **Detectron2 (Mask R-CNN)** - two-stage instance segmentation
* **Grounding DINO** - open-set, text-prompted detection
* **SAM** - foundation model for universal segmentation

### Highlights
* **Combined pipeline** `POST /api/v1/pipeline/detect-and-segment` chains
  Grounding DINO into SAM for zero-shot instance segmentation from text.
* **Explainability** `POST /api/v1/explain/gradcam` returns a Grad-CAM
  heatmap for a chosen YOLO detection.
* **Metrics** `GET /api/v1/metrics/summary` powers a live benchmark
  dashboard (per-model latency, cache-hit rate, throughput).
* **Layered architecture** - routers depend on services, services depend
  on repositories.  Cache and metrics backends switch between in-memory
  and Redis / MongoDB via environment variables alone.
"""


# ASGI entry point (`uvicorn app.main:app`)
app = create_app()
