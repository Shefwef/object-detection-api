# Unified Object Detection & Segmentation API — Complete Improvement Roadmap

> Drop this file into the project root. Every section is an actionable task.

---

## Why This Project Matters on Your CV

This is currently the strongest technical project in your portfolio for computer vision and MLOps roles. It already has the right architecture (FastAPI, 4 models, Docker, AWS IaC, K8s). What it is missing is a **visual frontend** and a **live demo URL** — without these, recruiters cannot experience the project. With them, it becomes an instant conversation starter.

---

## Current State Assessment

| Area | Status | Priority |
|---|---|---|
| FastAPI backend (4 models) | Strong | — |
| Swagger docs | Good | — |
| Docker multi-stage CUDA build | Strong | — |
| AWS ECS Fargate IaC | Strong | — |
| Kubernetes with HPA | Strong | — |
| Strategy Pattern abstraction | Good | — |
| Frontend / UI | Missing | CRITICAL |
| Live demo URL | Missing | CRITICAL |
| CI/CD pipeline | Missing | HIGH |
| Model comparison view | Missing | HIGH |
| Real-time webcam detection | Missing | HIGH |
| Video file inference | Missing | MEDIUM |
| API authentication | Missing | MEDIUM |
| Test suite | Missing | MEDIUM |
| Batch processing | Missing | LOW |

---

## System Architecture

### Application Architecture

```mermaid
graph TB
    subgraph Client["Frontend — React / Next.js (TypeScript + Tailwind)"]
        Upload["Image Upload\n& Webcam Stream"]
        Results["Results View\n(Annotated Canvas)"]
        Compare["Model Comparison\n(Side-by-side)"]
        Explain["Explainability\n(Grad-CAM Heatmap)"]
        Bench["Benchmark\nDashboard"]
    end

    subgraph API["FastAPI Backend — Python"]
        AuthMW["API Key Auth\n& Rate Limiting\n(slowapi)"]
        YOLOr["YOLOv8 Router\n/api/v1/yolo"]
        D2r["Detectron2 Router\n/api/v1/detectron2"]
        GDINOr["Grounding DINO Router\n/api/v1/grounding-dino"]
        SAMr["SAM Router\n/api/v1/sam"]
        Pipeliner["Pipeline Router\n/api/v1/pipeline"]
        ExplainR["Explainability Router\n/api/v1/explain"]
        MetricsR["Metrics Router\n/api/v1/metrics"]
        HealthR["/health"]
    end

    subgraph SvcLayer["Service Layer"]
        DetSvc["DetectionService\n(Orchestration + Cache)"]
        ExpSvc["ExplainabilityService\n(Grad-CAM)"]
        MetSvc["MetricsService\n(Perf tracking)"]
        PipeSvc["PipelineService\n(G-DINO → SAM)"]
    end

    subgraph ModelLayer["Model Layer — Strategy + Factory"]
        Factory["ModelFactory\n(Registry + DI)"]
        YOLOm["YOLOv8Model\n(Single-stage, 100+ FPS GPU)"]
        D2m["Detectron2Model\n(Two-stage, Mask R-CNN)"]
        GDINOm["GroundingDINOModel\n(Open-set, text-driven)"]
        SAMm["SAMModel\n(Foundation, universal seg)"]
        GradCAM["GradCAM Engine\n(Pixel-level XAI)"]
    end

    subgraph DataLayer["Data Layer"]
        Redis["Redis\n(Inference result cache\nTTL = 1h per image hash)"]
        Mongo["MongoDB\n(Request metrics,\nlatency history)"]
    end

    subgraph Deploy["Deployment"]
        Docker["Docker\n(Multi-stage CUDA build)"]
        HF["Hugging Face Spaces\n(Free GPU — T4)"]
        K8s["Kubernetes\n(HPA + rolling updates)"]
        ECS["AWS ECS Fargate\n(CloudFormation IaC)"]
    end

    Upload <-->|"REST / API Key"| AuthMW
    Compare <-->|"REST"| MetricsR
    Explain <-->|"REST"| ExplainR
    Bench <-->|"REST"| MetricsR

    AuthMW --> YOLOr
    AuthMW --> D2r
    AuthMW --> GDINOr
    AuthMW --> SAMr
    AuthMW --> Pipeliner
    AuthMW --> ExplainR

    YOLOr --> DetSvc
    D2r --> DetSvc
    GDINOr --> DetSvc
    SAMr --> DetSvc
    Pipeliner --> PipeSvc
    ExplainR --> ExpSvc
    MetricsR --> MetSvc

    DetSvc --> Factory
    PipeSvc --> Factory
    Factory --> YOLOm
    Factory --> D2m
    Factory --> GDINOm
    Factory --> SAMm
    ExpSvc --> GradCAM
    GradCAM --> YOLOm

    DetSvc <--> Redis
    MetSvc --> Mongo

    Docker --> HF
    Docker --> K8s
    Docker --> ECS
```

### Deployment Architecture

```mermaid
graph LR
    Dev["Developer"] -->|"git push main"| GH["GitHub"]

    GH -->|"GitHub Actions\n(pytest + ruff + mypy + docker build)"| CI["CI Pipeline"]

    CI -->|"Push image"| GHCR["GitHub Container\nRegistry (GHCR)"]
    CI -->|"git push hf main"| HFSpaces["Hugging Face Spaces\n(Free T4 GPU)"]

    subgraph Optional["Optional Production Targets"]
        ECS["AWS ECS Fargate\n(CloudFormation IaC)"]
        K8s["Kubernetes\n(HPA autoscaling)"]
    end

    GHCR --> ECS
    GHCR --> K8s

    subgraph Services["Backend Services"]
        Redis2["Redis (Upstash)\n(Inference cache)"]
        Mongo2["MongoDB Atlas\n(Metrics)"]
    end

    HFSpaces --> Redis2
    HFSpaces --> Mongo2
```

### Request Flow — Image Detection (Standard)

```
1. User uploads image via drag-and-drop → frontend
2. FormData POST → /api/v1/yolo/detect  (header: X-API-Key)
3. AuthMiddleware validates API key + checks rate limit (Redis)
4. DetectionService.detect(image, ModelType.YOLO, use_cache=True):
   a. image_hash = SHA-256(image bytes)
   b. Redis.get(f"inference:yolo:{image_hash}") → cache hit? return immediately
   c. Cache miss → ModelFactory.create(YOLO) → lazy-load YOLOv8 weights
   d. YOLOv8Model.infer(image) → list[Detection]
   e. Redis.setex(key, 3600, result.json()) → cache for 1 hour
   f. MetricsService.record(model, latency_ms, detection_count) → MongoDB
5. InferenceResponse (Pydantic validated) returned as JSON
6. Frontend draws bounding boxes on <canvas> element
7. structlog emits: { model, ms, detections, cached, image_hash }
```

### Request Flow — Open-Vocabulary Pipeline (Grounding DINO + SAM)

```
1. User enters text prompt: "person wearing a helmet"
2. POST /api/v1/pipeline/detect-and-segment  { text_prompt, confidence_threshold }
3. PipelineService.detect_and_segment():
   a. GroundingDINOModel.infer(image, text_prompt) → bounding boxes
   b. SAMModel.infer(image, boxes=boxes) → pixel-level masks per box
   c. Combine: each detection has { label, confidence, bbox, mask }
4. Response includes base64-encoded annotated image + detection array
5. Frontend renders mask overlay on canvas (colored fill per object)
```

### Request Flow — Grad-CAM Explainability

```
1. User clicks "Explain" on a YOLO detection result
2. POST /api/v1/explain/gradcam  { image, detection_index }
3. ExplainabilityService.generate_gradcam(image, target_class):
   a. Loads YOLOv8 backbone (already in memory via lazy cache)
   b. Runs forward pass → hooks capture gradients at target conv layer
   c. GradCAM computes weighted average of gradient maps
   d. Heatmap superimposed on original image (jet colormap)
4. Returns base64 heatmap image
5. Frontend switches "Results" view to heatmap overlay
6. Caption: "Red = high model attention | Blue = ignored"
```

---

## Phase 1 — Build the Frontend (Highest Impact)

A visual interface where you can upload an image and see annotated results from all four models side by side is the single most impressive thing you can add. Recruiters who cannot run a GPU model themselves can now interact with your work in a browser.

### Tech stack for frontend

```bash
# New Next.js frontend (or React — either works)
npx create-next-app@latest frontend --typescript --tailwind --app

npm install @radix-ui/react-tabs @radix-ui/react-progress
npm install react-dropzone       # drag-and-drop image upload
npm install framer-motion        # animations
npm install recharts             # latency/confidence charts
```

### Frontend layout

#### Home / Upload page

```
┌────────────────────────────────────────────────────────────────────┐
│  Object Detection Studio                    [API Docs] [GitHub]    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│           Detect anything. Compare every model.                   │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │              📁  Drop an image here                         │ │
│  │                  or click to browse                         │ │
│  │                                                              │ │
│  │                  Supports: JPG, PNG, WEBP (max 10MB)        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  Model Selection                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ ☑ YOLOv8 │  │☑Detectron│  │☑ Ground.DINO │  │   ☑ SAM     │  │
│  │ Fast Det.│  │Instance  │  │ Text Prompt  │  │  Segment    │  │
│  │          │  │  Seg.    │  │              │  │ Anything    │  │
│  └──────────┘  └──────────┘  └──────────────┘  └─────────────┘  │
│                                                                    │
│  For Grounding DINO:  [Text prompt: "person wearing helmet"...]   │
│                                                                    │
│                        [  Run Detection  ]                         │
└────────────────────────────────────────────────────────────────────┘
```

#### Results page (after inference)

```
┌────────────────────────────────────────────────────────────────────┐
│  Results                                    [← New Image] [Export]│
├──────────────┬─────────────────────────────────────────────────────┤
│              │   [YOLOv8] [Detectron2] [G-DINO] [SAM] [Compare]   │
│  Original    ├─────────────────────────────────────────────────────┤
│  ┌────────┐  │                                                     │
│  │        │  │   ┌──────────────────────────────┐                 │
│  │ [img]  │  │   │  [Annotated image with boxes]│                 │
│  │        │  │   │  Bounding boxes drawn on      │                 │
│  └────────┘  │   │  canvas with label + %conf   │                 │
│              │   └──────────────────────────────┘                 │
│  Detections  │                                                     │
│  ──────────  │   Detected Objects (8)                              │
│  person  3   │   ┌──────┬────────────────┬──────────┬──────────┐  │
│  car     2   │   │ #    │ Label          │ Conf     │ Box      │  │
│  helmet  1   │   ├──────┼────────────────┼──────────┼──────────┤  │
│              │   │ 1    │ person         │  94.2%   │ [coords] │  │
│  Inference   │   │ 2    │ person         │  87.1%   │ [coords] │  │
│  Time        │   │ 3    │ car            │  91.8%   │ [coords] │  │
│  ──────────  │   └──────┴────────────────┴──────────┴──────────┘  │
│  128ms       │                                                     │
└──────────────┴─────────────────────────────────────────────────────┘
```

#### Model Comparison tab

```
┌────────────────────────────────────────────────────────────────────┐
│  Model Comparison                                                  │
├───────────────────┬───────────────────┬────────────────────────────┤
│     YOLOv8        │    Detectron2      │   Grounding DINO + SAM     │
│   [img w/ boxes]  │  [img w/ masks]   │   [img w/ pixel masks]     │
│   8 detections    │   5 instances     │   3 objects (text-driven)  │
│   128ms           │   843ms           │   1200ms                   │
├───────────────────┴───────────────────┴────────────────────────────┤
│  Latency comparison                                                 │
│  YOLOv8   ██░░░░░░░░░░░░░░░  128ms   (fastest)                    │
│  Detectron ████████░░░░░░░░  843ms                                 │
│  G-DINO   █████████░░░░░░░░  1200ms  (most flexible)              │
└────────────────────────────────────────────────────────────────────┘
```

### Canvas annotation rendering

```typescript
// src/components/AnnotatedImage.tsx

export function AnnotatedImage({ imageUrl, detections }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const img = new Image()
    img.src = imageUrl
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      detections.forEach((det, i) => {
        const color = COLORS[i % COLORS.length]
        const [x1, y1, x2, y2] = det.bbox

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        // Draw label background
        ctx.fillStyle = color
        ctx.fillRect(x1, y1 - 24, (x2 - x1), 24)

        // Draw label text
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 14px Inter'
        ctx.fillText(`${det.label} ${(det.confidence * 100).toFixed(0)}%`, x1 + 4, y1 - 6)
      })
    }
  }, [imageUrl, detections])

  return <canvas ref={canvasRef} className="max-w-full rounded-lg" />
}
```

---

## Phase 2 — Real-Time Webcam Detection

This is the most visually stunning feature for demos. Opening the webcam and watching boxes appear in real time is unforgettable.

### How it works

- Browser captures webcam frames (every 200ms → 5 FPS is smooth enough)
- Sends frame as base64 to FastAPI `/api/v1/yolo/detect` (fastest model)
- Receives detections → draws on canvas overlay
- Loop continues

### Frontend component

```typescript
// src/components/WebcamDetection.tsx
import Webcam from 'react-webcam'

export function WebcamDetection() {
  const webcamRef = useRef<Webcam>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [fps, setFps] = useState(0)

  useEffect(() => {
    const interval = setInterval(async () => {
      const imageSrc = webcamRef.current?.getScreenshot()
      if (!imageSrc) return

      const t0 = performance.now()
      const blob = await (await fetch(imageSrc)).blob()
      const formData = new FormData()
      formData.append('file', blob, 'frame.jpg')

      const res = await fetch('/api/proxy/yolo/detect', { method: 'POST', body: formData })
      const { detections } = await res.json()

      drawDetections(canvasRef.current!, detections)
      setFps(Math.round(1000 / (performance.now() - t0)))
    }, 200)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="relative">
      <Webcam ref={webcamRef} screenshotFormat="image/jpeg" className="rounded-xl" />
      <canvas ref={canvasRef} className="absolute inset-0 rounded-xl" />
      <span className="absolute top-2 right-2 bg-black/60 text-white px-2 py-1 rounded text-sm">
        {fps} FPS • YOLOv8
      </span>
    </div>
  )
}
```

```bash
npm install react-webcam
```

### FastAPI change needed

Add a `/api/v1/yolo/detect-base64` endpoint that accepts base64 image strings (faster for webcam streaming than multipart form):

```python
@router.post("/detect-base64")
async def detect_base64(payload: Base64ImageRequest):
    image_data = base64.b64decode(payload.image.split(",")[1])
    image = Image.open(BytesIO(image_data))
    return await yolo_service.detect(image)
```

---

## Phase 3 — Make It Live

### Best option: Hugging Face Spaces (Free GPU)

Hugging Face Spaces supports Docker deployments with free T4 GPU access. This is perfect for this project.

```
Repository structure for HF Spaces:
├── Dockerfile           (modify to meet HF requirements)
├── app/                 (your FastAPI app)
├── frontend/            (your Next.js app, or serve static build)
└── README.md            (HF Spaces config in frontmatter)
```

**HF Spaces README frontmatter:**
```yaml
---
title: Object Detection Studio
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
---
```

**HF Spaces Dockerfile adjustments:**
- Change `EXPOSE 8000` to `EXPOSE 7860` (HF default port)
- Remove the GPU runtime requirement for HF free tier (CPU-only models will still work; YOLO runs at ~1-3 FPS on CPU which is fine for demo)
- Use `ultralytics/ultralytics` as base image for YOLO

This gives you: `https://huggingface.co/spaces/Shefwef/object-detection-studio`

### Option B: Modal.com (Serverless GPU, pay-per-use)

```python
# deploy.py
import modal

app = modal.App("object-detection-api")
image = modal.Image.from_registry("ultralytics/ultralytics:latest")

@app.function(image=image, gpu="T4", timeout=60)
@modal.web_endpoint(method="POST")
def detect(item: dict):
    # Your detection logic here
    pass
```

Cold start is ~10s but then it scales to zero. Free $30/month credit for new accounts.

### Option C: Deploy frontend on Vercel + backend on Render

```
Frontend (Next.js) → Vercel (free)
Backend (FastAPI)  → Render (free tier, CPU-only, sleeps after 15min)
```

This is CPU-only so YOLO detection will be ~2-5s, which is acceptable for a demo.

**Recommendation: Start with Render (Option C) for fast setup, then move to HF Spaces (Option A) for GPU and professional presence.**

---

## Phase 4 — CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest httpx
      - run: pytest tests/ -v --tb=short

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff mypy
      - run: ruff check app/
      - run: mypy app/ --ignore-missing-imports

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t object-detection-api:test .

  deploy-hf:
    needs: [test, lint, docker-build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Push to Hugging Face Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://Shefwef:${HF_TOKEN}@huggingface.co/spaces/Shefwef/object-detection-studio
          git push hf main --force
```

---

## Phase 5 — Test Suite

```
tests/
├── test_yolo.py              # YOLOv8 detection endpoint
├── test_detectron2.py        # Detectron2 instance seg endpoint
├── test_grounding_dino.py    # G-DINO text-based detection
├── test_sam.py               # SAM segmentation modes
├── test_pipeline.py          # Grounding DINO + SAM combined
├── test_health.py            # Health endpoint + model status
└── conftest.py               # Shared fixtures (test image, mock models)
```

```python
# tests/test_yolo.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
def test_image():
    with open("tests/fixtures/test_image.jpg", "rb") as f:
        return f.read()

@pytest.mark.asyncio
async def test_yolo_detect(test_image):
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/yolo/detect",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert "inference_time_ms" in data
```

---

## Phase 6 — API Authentication

Add an API key system so the live demo can be used without exposing unlimited free inference:

```python
# app/middleware/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = set(os.getenv("API_KEYS", "demo-key-12345").split(","))

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

The frontend uses a public demo key with rate limiting (5 requests/minute). The README shows how to request a key for production use.

---

## Phase 7 — Video File Inference

```python
# app/routers/video.py
@router.post("/yolo/detect-video")
async def detect_video(file: UploadFile = File(...), every_n_frames: int = 5):
    """Process a video file, running detection every N frames."""
    video_bytes = await file.read()
    frames = extract_frames(video_bytes, every_n_frames)
    
    results = []
    for i, frame in enumerate(frames):
        detections = yolo_model.detect(frame)
        results.append({
            "frame": i * every_n_frames,
            "detections": detections
        })
    
    return {"total_frames": len(frames), "results": results}
```

Frontend shows a timeline of detections across the video with a scrubber.

---

## Phase 8 — Benchmark Dashboard

Add a `/benchmark` page that runs all 4 models on a standard set of test images and shows:

| Metric | YOLOv8 | Detectron2 | G-DINO | SAM |
|---|---|---|---|---|
| Avg latency | 128ms | 843ms | 1200ms | 450ms |
| COCO mAP | 53.9 | 44.3 | 48.5 | N/A |
| Memory (GPU) | 1.2GB | 4.1GB | 3.8GB | 2.4GB |
| Best for | Speed | Quality | Open-set | Segmentation |

This is a visual proof of the comparative analysis claim in your CV description.

---

## CV Description (after improvements complete)

```
Unified Object Detection & Segmentation API   GitHub | Live Demo | HF Space

- FastAPI service exposing 4 CV models (YOLOv8, Detectron2/Mask R-CNN,
  Grounding DINO, SAM) via versioned REST endpoints with Swagger docs;
  Strategy Pattern abstraction, lazy loading, and per-model health checks.
- Grounding DINO + SAM open-vocabulary pipeline (text prompt to pixel
  masks, no retraining); real-time webcam inference at 5+ FPS via YOLOv8.
- React/Next.js frontend with drag-and-drop upload, canvas annotation
  rendering, side-by-side model comparison, and live webcam detection.
- Multi-stage CUDA-ready Docker build; deployed on Hugging Face Spaces
  (free GPU); GitHub Actions CI/CD with pytest, ruff, mypy, and auto-push
  to HF Spaces on main merge.
```

---

## Phase 9 — Clean Architecture & Software Engineering Principles

### Current Architecture (what you have)

The project already uses a Strategy Pattern for model abstraction — that is genuinely good design. This section formalizes and extends it so every layer is clearly defensible in an interview.

### Formal Class Hierarchy (Abstract Base Class)

```python
# app/models/base_model.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: list[list[int]] | None = None  # pixel mask for segmentation models

@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_time_ms: float
    model_name: str
    image_width: int
    image_height: int

class BaseDetectionModel(ABC):
    """
    Strategy interface for all detection/segmentation models.
    All concrete models implement this contract — the API layer
    never knows which model it is calling.
    """

    @abstractmethod
    def load(self) -> None:
        """Lazy-load model weights. Called on first request, not at startup."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model weights are in memory."""

    @abstractmethod
    async def infer(self, image: Image.Image, **kwargs) -> InferenceResult:
        """Run inference. kwargs carries model-specific params (e.g. text_prompt for G-DINO)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @property
    @abstractmethod
    def supports_segmentation(self) -> bool:
        """True if model outputs pixel masks in addition to bounding boxes."""
```

### Factory Pattern — Model Registry

```python
# app/models/model_factory.py

from enum import Enum
from app.models.base_model import BaseDetectionModel
from app.models.yolo_model import YOLOv8Model
from app.models.detectron2_model import Detectron2Model
from app.models.grounding_dino import GroundingDINOModel
from app.models.sam_model import SAMModel

class ModelType(str, Enum):
    YOLO = "yolo"
    DETECTRON2 = "detectron2"
    GROUNDING_DINO = "grounding_dino"
    SAM = "sam"

class ModelFactory:
    _registry: dict[ModelType, type[BaseDetectionModel]] = {
        ModelType.YOLO: YOLOv8Model,
        ModelType.DETECTRON2: Detectron2Model,
        ModelType.GROUNDING_DINO: GroundingDINOModel,
        ModelType.SAM: SAMModel,
    }

    @classmethod
    def create(cls, model_type: ModelType) -> BaseDetectionModel:
        model_class = cls._registry.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class()

    @classmethod
    def register(cls, model_type: ModelType, model_class: type[BaseDetectionModel]) -> None:
        """Open for extension — new models can be registered without modifying existing code."""
        cls._registry[model_type] = model_class
```

### Repository Pattern — Results Cache

```python
# app/repositories/inference_repository.py

from abc import ABC, abstractmethod
from app.models.base_model import InferenceResult

class IInferenceRepository(ABC):
    @abstractmethod
    async def save(self, image_hash: str, model: str, result: InferenceResult) -> None: ...

    @abstractmethod
    async def get_cached(self, image_hash: str, model: str) -> InferenceResult | None: ...

    @abstractmethod
    async def get_history(self, limit: int = 50) -> list[dict]: ...

class RedisInferenceRepository(IInferenceRepository):
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # Cache results for 1 hour

    async def save(self, image_hash: str, model: str, result: InferenceResult) -> None:
        key = f"inference:{model}:{image_hash}"
        await self.redis.setex(key, self.ttl, result.model_dump_json())

    async def get_cached(self, image_hash: str, model: str) -> InferenceResult | None:
        key = f"inference:{model}:{image_hash}"
        cached = await self.redis.get(key)
        return InferenceResult.model_validate_json(cached) if cached else None
```

### Service Layer — Orchestration

```python
# app/services/detection_service.py

class DetectionService:
    def __init__(
        self,
        model_factory: ModelFactory,
        inference_repo: IInferenceRepository,
        logger: Logger,
    ):
        self._factory = model_factory
        self._repo = inference_repo
        self._logger = logger
        self._models: dict[ModelType, BaseDetectionModel] = {}

    def _get_or_load(self, model_type: ModelType) -> BaseDetectionModel:
        if model_type not in self._models:
            self._models[model_type] = self._factory.create(model_type)
        model = self._models[model_type]
        if not model.is_loaded():
            self._logger.info(f"Loading model: {model.model_name}")
            model.load()
        return model

    async def detect(
        self,
        image: Image.Image,
        model_type: ModelType,
        use_cache: bool = True,
        **kwargs,
    ) -> InferenceResult:
        image_hash = compute_hash(image)

        if use_cache:
            cached = await self._repo.get_cached(image_hash, model_type)
            if cached:
                self._logger.debug(f"Cache hit: {model_type} / {image_hash}")
                return cached

        model = self._get_or_load(model_type)
        result = await model.infer(image, **kwargs)

        await self._repo.save(image_hash, model_type, result)
        self._logger.info(
            f"Inference complete",
            extra={"model": model_type, "detections": len(result.detections), "ms": result.inference_time_ms}
        )
        return result
```

### Pydantic Schemas — Strict Request/Response Contracts

```python
# app/schemas/detection.py

from pydantic import BaseModel, Field, field_validator

class DetectionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int]
    mask: list[list[int]] | None = None

class InferenceResponse(BaseModel):
    model_name: str
    inference_time_ms: float
    detections: list[DetectionResponse]
    total_detections: int
    image_dimensions: tuple[int, int]

class GroundingDINORequest(BaseModel):
    text_prompt: str = Field(min_length=1, max_length=500)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator('text_prompt')
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        return v.strip().lower()
```

### Dependency Injection via FastAPI

```python
# app/dependencies.py

from functools import lru_cache
import redis.asyncio as redis
from app.services.detection_service import DetectionService
from app.repositories.inference_repository import RedisInferenceRepository

@lru_cache
def get_redis():
    return redis.from_url(settings.REDIS_URL)

def get_inference_repo(r=Depends(get_redis)) -> IInferenceRepository:
    return RedisInferenceRepository(r)

def get_detection_service(
    repo: IInferenceRepository = Depends(get_inference_repo),
) -> DetectionService:
    return DetectionService(ModelFactory(), repo, logger)

# Router usage — clean, no globals:
@router.post("/yolo/detect")
async def detect_yolo(
    file: UploadFile,
    service: DetectionService = Depends(get_detection_service),
):
    image = Image.open(BytesIO(await file.read()))
    return await service.detect(image, ModelType.YOLO)
```

### Structured Async Logging

```python
# app/core/logging.py

import structlog

logger = structlog.get_logger()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# Usage — every field searchable in Datadog/CloudWatch:
logger.info("inference_complete", model="yolo", ms=128.4, detections=7, cached=False)
logger.error("inference_failed", model="detectron2", error=str(e), image_hash=h)
```

---

## Phase 10 — Grad-CAM / Saliency Explainability

This is the single feature that most directly maps to the "XAI" trend (Explainable AI) that is dominant in enterprise hiring right now. SHAP is to tabular ML what Grad-CAM is to computer vision.

### What Grad-CAM does

Grad-CAM generates a heatmap showing **which pixels in the image most influenced a detection**. Instead of just saying "I detected a person with 94% confidence," you can show *why* — which region of the image triggered the classification.

### Implementation (YOLOv8 + Grad-CAM)

```python
# app/services/explainability_service.py

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplainabilityService:
    def __init__(self, yolo_model):
        self.model = yolo_model

    def generate_gradcam(self, image_np: np.ndarray, target_layer=None) -> np.ndarray:
        """Returns heatmap overlay as numpy array (H x W x 3, uint8)."""
        rgb_img = image_np.astype(np.float32) / 255.0
        target_layer = target_layer or self.model.model.model[-2]

        cam = GradCAM(model=self.model.model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=self._to_tensor(rgb_img))
        return show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

# New endpoint:
# POST /api/v1/yolo/detect-explain
# Returns both detections AND a Grad-CAM heatmap as base64 image
```

```bash
pip install grad-cam
```

### Frontend rendering

Add an "Explain" toggle on the Results page. When enabled, the annotated image is replaced with the Grad-CAM heatmap overlay — hot areas (red) are where the model "looked" to make its decision.

```
┌──────────────────────────────────────────┐
│  [Detections] [Segmentation] [Explain ●] │
├──────────────────────────────────────────┤
│                                          │
│  [Grad-CAM heatmap image]                │
│  Red = high attention   Blue = ignored   │
│                                          │
│  The model focused on the upper-left     │
│  region (person's torso + helmet) to     │
│  make the "person wearing helmet"        │
│  classification.                         │
│                                          │
└──────────────────────────────────────────┘
```

---

## Phase 11 — Model Auto-Selection & Performance Tracking (MLOps Pattern)

Inspired by the churn prediction project's "auto-promote best model by ROC-AUC" pattern.

### What to build

Track inference metrics per model across all requests:

```python
# app/repositories/model_metrics_repository.py

class ModelMetricsRepository:
    async def record(self, model: str, latency_ms: float, detection_count: int) -> None:
        await self.mongo.model_metrics.insert_one({
            "model": model,
            "latency_ms": latency_ms,
            "detection_count": detection_count,
            "timestamp": datetime.utcnow(),
        })

    async def get_summary(self) -> list[ModelMetricsSummary]:
        return await self.mongo.model_metrics.aggregate([
            { "$group": {
                "_id": "$model",
                "avg_latency_ms": { "$avg": "$latency_ms" },
                "total_requests": { "$sum": 1 },
                "avg_detections": { "$avg": "$detection_count" },
            }}
        ]).to_list()
```

### Admin / Benchmark endpoint

```
GET /api/v1/admin/model-metrics

Returns:
{
  "yolo":          { "avg_latency_ms": 128, "total_requests": 4521, "avg_detections": 6.2 },
  "detectron2":    { "avg_latency_ms": 843, "total_requests": 891,  "avg_detections": 5.8 },
  "grounding_dino":{ "avg_latency_ms": 1200,"total_requests": 234,  "avg_detections": 3.1 },
  "sam":           { "avg_latency_ms": 450, "total_requests": 312,  "avg_detections": 8.4 },
}
```

Display this on the Benchmark Dashboard page (Phase 8) — live stats, not hardcoded numbers.

---

## Priority Order

1. **Build the frontend** (Phase 1) — image upload + annotated results → biggest demo impact
2. **Clean Architecture refactor** (Phase 9) — Factory + Strategy + Repository + DI before adding features
3. **Deploy on HF Spaces / Render** (Phase 3) — live URL changes everything
4. **Webcam detection** (Phase 2) — most visually impressive single feature
5. **Grad-CAM explainability** (Phase 10) — the feature that places this in XAI territory, highest AI recruiter signal
6. **CI/CD** (Phase 4) — one YAML file, high signal for engineers
7. **Model metrics tracking** (Phase 11) — makes the "comparative analysis" claim backed by live data
8. **Test suite** (Phase 5) — target the Service layer and Repository specifically
9. **API authentication** (Phase 6) — needed before live deployment handles real traffic
10. **Video inference** (Phase 7) — good differentiator once basics are solid

