# ════════════════════════════════════════════════════════════
#  Multi-Model CV Detection Platform — CPU-Only Dockerfile
# ════════════════════════════════════════════════════════════
#  Simple single-stage build optimized for CPU inference
#  Build: docker build -t cv-detection-api .
#  Run:   docker run -p 8000:8000 cv-detection-api
# ════════════════════════════════════════════════════════════

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and ML libraries
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

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU-only FIRST (large but cached separately)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .

# Install remaining packages (skip torch/torchvision already installed)
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
    git+https://github.com/facebookresearch/segment-anything.git

# Create application directory
WORKDIR /app

# Copy application code and model weights
COPY app/ ./app/
COPY yolov8n.pt ./yolov8n.pt

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
