# ════════════════════════════════════════════════════════════
#  Multi-Model CV Detection Platform — CPU Dockerfile
# ════════════════════════════════════════════════════════════
#  Works locally, on AWS ECS Fargate, on Kubernetes, and on
#  Hugging Face Spaces. The container listens on ${PORT:-8000}
#  so HF Spaces (which mandates 7860) is a one-liner override.
#
#  Local:   docker build -t cv-detection-api .
#           docker run -p 8000:8000 cv-detection-api
#  HF:      the platform sets PORT=7860 automatically.
# ════════════════════════════════════════════════════════════

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HF_HOME=/home/appuser/.cache/huggingface \
    XDG_CACHE_HOME=/home/appuser/.cache

# System dependencies for OpenCV + ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CPU wheels (cached separately in its own layer)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Python deps
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    python-multipart==0.0.19 \
    pydantic==2.10.4 \
    pydantic-settings==2.7.1 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.24.3 \
    Pillow>=10.0.0 \
    ultralytics>=8.3.0 \
    transformers>=4.42.0 \
    structlog>=24.4.0 \
    git+https://github.com/facebookresearch/segment-anything.git

# ── Application
WORKDIR /app
COPY app/ ./app/

# ── Non-root user (also HF Spaces requirement: uid 1000)
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser
USER appuser

# YOLOv8n weights auto-download on first inference call to
# /home/appuser/.cache/... — no need to bake them into the image.

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000
EXPOSE 7860

# Honor $PORT so HF Spaces (7860) and everything else (8000) work
# from the same image.
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
