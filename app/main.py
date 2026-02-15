"""
Multi-Model Object Detection & Segmentation Platform.

FastAPI application that unifies YOLOv8, Detectron2, Grounding DINO,
and SAM behind a single REST API with Swagger documentation.

Author: Shefayat E Shams Adib
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from app.config import get_settings
from app.routers import (
    yolo_router,
    detectron2_router,
    grounding_dino_router,
    sam_router,
    pipeline_router,
)

# ─── Logging Setup ─────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ───────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Multi-Model Computer Vision API

A production-ready platform that unifies four state-of-the-art CV models:

- **YOLOv8**: Real-time object detection (80 COCO classes)
- **Detectron2**: Instance segmentation with Mask R-CNN
- **Grounding DINO**: Open-set detection with text prompts
- **SAM**: Segment Anything from points/boxes/auto

### Combined Pipeline
Use **Grounding DINO + SAM** for open-vocabulary instance segmentation:
describe any object in text → get pixel-perfect segmentation masks.

### Key Features
- Lazy model loading (conserves memory)
- GPU/CPU auto-detection
- Production-ready with health checks
- Dockerized with AWS deployment support
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Middleware ───────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ─────────────────────────────────

app.include_router(yolo_router.router, prefix="/api/v1")
app.include_router(detectron2_router.router, prefix="/api/v1")
app.include_router(grounding_dino_router.router, prefix="/api/v1")
app.include_router(sam_router.router, prefix="/api/v1")
app.include_router(pipeline_router.router, prefix="/api/v1")

# ─── Root & Health Endpoints ──────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "models": ["yolov8", "detectron2", "grounding_dino", "sam"],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Reports the status of each model (loaded or not).
    """
    from app.routers.yolo_router import get_model as get_yolo
    from app.routers.detectron2_router import get_model as get_detectron2
    from app.routers.grounding_dino_router import get_model as get_gdino
    from app.routers.sam_router import get_model as get_sam

    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "models": {
            "yolov8": get_yolo().get_model_info(),
            "detectron2": get_detectron2().get_model_info(),
            "grounding_dino": get_gdino().get_model_info(),
            "sam": get_sam().get_model_info(),
        },
    }


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info("Models will be loaded lazily on first request")
    logger.info(f"API docs available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CV Detection Platform")
