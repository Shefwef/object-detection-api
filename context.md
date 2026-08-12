# Context — Unified Object Detection & Segmentation API

> A single, self-contained brain-dump of what this repository is, why it is
> structured the way it is, and how every piece of it fits together.
> Written to be picked up cold by a future contributor (human or AI) with
> zero prior conversation history.

---

## 1. What this project is

A production-oriented **FastAPI** service that unifies four state-of-the-art
computer-vision models — **YOLOv8**, **Detectron2 (Mask R-CNN)**,
**Grounding DINO**, and **SAM** — behind a single versioned REST API.

The project delivers three interlocking things:

1. **A unified inference layer.**  Every model is wrapped in a common
   `BaseDetectionModel` contract, so the API layer never needs to know
   which model it is calling.
2. **A production wrapper around that layer.**  Layered clean
   architecture, pluggable inference cache (in-memory ↔ Redis), pluggable
   metrics store (in-memory ↔ MongoDB), opt-in API-key auth + rate
   limiting, structured JSON logging, and a Grad-CAM explainability
   endpoint.
3. **A deployment story.**  Multi-stage CUDA-ready Dockerfile, AWS ECS
   Fargate CloudFormation, and Kubernetes manifests with HPA.

The single most impressive endpoint is the **Grounding DINO + SAM
pipeline**: send an image plus a free-form text prompt (`"person wearing
helmet"`) and receive pixel-perfect segmentation masks — open-vocabulary
instance segmentation without any retraining.

---

## 2. Repository layout (annotated)

```
object-detection-api/
├── README.md                     # Public-facing README with diagrams
├── context.md                    # ← you are here
├── Improvements.md               # Full roadmap (Phases 1-11)
├── TESTING_GUIDE.md              # Manual QA notes
├── Dockerfile                    # Multi-stage CUDA build
├── docker-compose.yml            # GPU + CPU profiles
├── requirements.txt              # Runtime deps (see §7 for optional ones)
├── .env.example                  # All settings documented + defaults
├── pytest.ini
│
├── app/                          # The FastAPI application
│   ├── main.py                   # ASGI entry + lifespan + middleware wiring
│   ├── config.py                 # Pydantic Settings
│   ├── dependencies.py           # FastAPI DI providers (services, repos)
│   ├── core/logging.py           # structlog / stdlib bootstrap
│   ├── middleware/auth.py        # API-key + rate-limit middleware
│   ├── models/                   # Strategy pattern
│   │   ├── base_model.py         #   BaseDetectionModel ABC + dataclasses
│   │   ├── model_factory.py      #   Registry + singleton cache
│   │   ├── yolo_model.py         #   Concrete strategies (unchanged interface)
│   │   ├── detectron2_model.py
│   │   ├── grounding_dino.py
│   │   └── sam_model.py
│   ├── repositories/             # Persistence
│   │   ├── inference_repository.py
│   │   └── metrics_repository.py
│   ├── services/                 # Orchestration
│   │   ├── detection_service.py
│   │   ├── pipeline_service.py
│   │   ├── explainability_service.py
│   │   └── metrics_service.py
│   ├── routers/                  # Thin HTTP adapters
│   │   ├── yolo_router.py        # + /detect-base64 for webcam
│   │   ├── detectron2_router.py
│   │   ├── grounding_dino_router.py
│   │   ├── sam_router.py
│   │   ├── pipeline_router.py
│   │   ├── explain_router.py     # NEW · Grad-CAM
│   │   └── metrics_router.py     # NEW · live benchmark
│   ├── schemas/detection.py      # Pydantic request/response contracts
│   └── utils/
│       ├── image_utils.py
│       ├── visualization.py
│       └── annotation_utils.py
│
├── deployment/
│   ├── aws/                      # ECR push + CloudFormation stack
│   └── k8s/                      # Deployment, Service, Ingress, HPA
│
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_api.py               # Legacy endpoint tests
│   ├── test_repositories.py      # NEW
│   ├── test_detection_service.py # NEW (uses StubModel)
│   ├── test_new_endpoints.py     # NEW (metrics, base64, openapi)
│   ├── batch_test_models.py      # Manual batch checker
│   └── test_utils.py
│
├── sample_images/                # Small demo images
└── test_results/                 # Manual test artefacts (safe to gitignore)
```

---

## 3. Architecture — layer by layer

The project follows a strict **routers → services → repositories → models**
dependency direction.  A layer never imports upward.

```
Client ─▶ Middleware (CORS · API key · Rate limit)
        ─▶ Router (parses HTTP, decodes image bytes)
            ─▶ Service (orchestrates: cache lookup → model → cache write → metrics)
                ─▶ Repository (persists cache entries and metric records)
                ─▶ ModelFactory ─▶ BaseDetectionModel ─▶ Concrete model
```

### 3.1 API layer — `app/routers/**`

Every router is a **thin HTTP adapter**. Concretely, a router:

1. Reads the multipart / JSON payload.
2. Decodes the image bytes into a NumPy array.
3. Calls the appropriate service via `fastapi.Depends`.
4. Returns the service result as JSON.

No caching, no model wiring, no metrics logic lives in routers.

**Files & endpoints**

| File | Endpoints |
|---|---|
| `yolo_router.py` | `/detect`, `/detect-base64` (webcam-ready), `/detect-visualize`, `/info` |
| `detectron2_router.py` | `/detect`, `/detect-visualize`, `/info` |
| `grounding_dino_router.py` | `/detect`, `/detect-visualize`, `/info` |
| `sam_router.py` | `/segment-auto`, `/segment-points`, `/segment-boxes`, `/info` |
| `pipeline_router.py` | `/detect-and-segment`, `/detect-and-segment-visualize` |
| `explain_router.py` | `/gradcam` |
| `metrics_router.py` | `/summary`, `/recent`, `/reset` |

### 3.2 Service layer — `app/services/**`

Services own **orchestration and cross-cutting behaviour** (caching, timing,
metrics recording, pipeline composition).

- **`DetectionService`** — the entry point for every single-model endpoint.
  Steps: hash image bytes → check cache → call the model → write cache →
  record metric → return `InferenceResult`.
- **`PipelineService`** — Grounding DINO → SAM composition. Records
  metrics under the synthetic model name `grounding_dino+sam` so the
  benchmark dashboard treats the pipeline as its own row.
- **`ExplainabilityService`** — produces a Grad-CAM heatmap for a YOLO
  detection. Uses `pytorch-grad-cam` when installed and gracefully falls
  back to a **Sobel + Gaussian saliency map** so the endpoint always
  works.
- **`MetricsService`** — thin façade over `IMetricsRepository` so
  routers depend on a service, not a repo.

### 3.3 Repository layer — `app/repositories/**`

Persistence is fully abstracted so the choice of backend is invisible to
callers.

- **`IInferenceRepository`** — cache API keyed by
  `image_hash + model_name`.
  - `InMemoryInferenceRepository`: LRU dict with TTL. Zero dependencies.
    Default.
  - `RedisInferenceRepository`: swap in by setting `REDIS_URL`.
    Degrades to in-memory if Redis is unreachable at startup.

- **`IMetricsRepository`** — per-model record store with server-side
  aggregation.
  - `InMemoryMetricsRepository`: bounded ring buffer per model.
    Includes p50 / p95 latency, cache-hit rate, throughput, last-seen.
  - `MongoMetricsRepository`: swap in by setting `MONGO_URL`.

`compute_image_hash(bytes) → SHA-256` in `inference_repository.py` is the
canonical key generator.

### 3.4 Model layer — `app/models/**`

Every model implements **`BaseDetectionModel`** (Strategy pattern):

```
BaseDetectionModel (ABC)
    load_model()                       - lazy weight loading
    is_loaded            @property     - readiness
    ensure_loaded()                    - lazy trigger
    predict(image, **kw) -> dict       - raw native output
    infer(image, **kw)   -> InferenceResult
    get_model_info()     -> dict
```

`infer()` has a default implementation that times `predict()` and
normalizes its dict output into an `InferenceResult`. This is what makes
the service layer model-agnostic — but a subclass can override `infer()`
if the model has special-purpose outputs (e.g. SAM's auto-mask generator
emits *segments*, not detections).

**`ModelFactory`** owns a `{ModelType → class}` registry and a
`{ModelType → instance}` singleton cache (`get_or_create`). New models
plug in with:

```python
ModelFactory.register(ModelType.MY_MODEL, MyModelClass)
```

No caller code changes.

### 3.5 Middleware — `app/middleware/auth.py`

Attached in `create_app()` via `register_middleware(app)`.

Two features, both opt-in:

- **API-key authentication** — when `AUTH_ENABLED=true`, requests without a
  valid `X-API-Key` header get a `401`. Health / docs / OpenAPI paths
  stay public regardless.
- **Rate limiting** — in-process fixed-window counter (60 seconds) keyed
  by API key or client IP. Returns `429 Retry-After: 60` when exceeded.
  For horizontally scaled deployments, swap in a Redis-backed limiter by
  editing `register_middleware`.

### 3.6 Configuration — `app/config.py`

Pydantic v2 `BaseSettings` hydrated from environment variables and
`.env`.  Cached with `lru_cache` so import cost is amortized.

Groups: application, server, model runtime, per-model settings, upload
limits, logging, cache backend, metrics backend, authentication. Every
knob has a documented default. See `.env.example`.

### 3.7 Logging — `app/core/logging.py`

`configure_logging()` is called once from `main.py`. When
`LOG_JSON=true` and `structlog` is installed, every line becomes a JSON
document. Otherwise a plain-text stdlib formatter is used. Because
`structlog` is imported inside a `try`, the API still runs without it.

### 3.8 Dependency injection — `app/dependencies.py`

Routers never construct their dependencies. They receive them via
`Depends(get_...)` providers that in turn wire the correct backend
(Redis / Mongo / in-memory) based on `Settings`. Each backend is
`lru_cache`-d for the process lifetime.

`reset_dependency_cache()` exists for tests.

---

## 4. Request flow examples

### 4.1 YOLOv8 single-image detection

1. Client `POST /api/v1/yolo/detect` (multipart JPEG).
2. `AuthMiddleware` checks the API key + rate limit, forwards.
3. `yolo_router.detect_objects` reads bytes, decodes to NumPy.
4. `DetectionService.detect(image, YOLO, image_bytes=bytes, **kwargs)`:
   1. Compute `image_hash = SHA-256(bytes)`.
   2. `IInferenceRepository.get(image_hash, "yolov8")` — cache hit ⇒
      short-circuit.
   3. Cache miss ⇒ `ModelFactory.get_or_create(YOLO).infer(image, ...)`.
   4. Cache the result (`setex` with TTL).
   5. Record a `MetricRecord`.
5. Router serializes the `InferenceResult` and returns JSON with
   `inference_time_ms` and `cached` alongside detections.

### 4.2 Grounding DINO + SAM pipeline

1. Client `POST /api/v1/pipeline/detect-and-segment { file, text_prompt }`.
2. `PipelineService.detect_and_segment(...)`:
   1. `GroundingDINO.get_boxes_for_sam(image, prompt)` → boxes, labels,
      scores.
   2. `SAM.predict(image, mode="boxes", boxes=boxes)` → per-box masks.
   3. Record a metric under `grounding_dino+sam`.
3. Router returns `{ detections, segments, inference_time_ms, ... }` or
   an annotated JPEG.

### 4.3 Grad-CAM explanation

1. Client `POST /api/v1/explain/gradcam { file, detection_index? }`.
2. `ExplainabilityService.explain(image, detection_index, confidence)`:
   1. Runs YOLO to get detections; picks the target box (highest
      confidence by default).
   2. **If `pytorch-grad-cam` is installed:** true Grad-CAM against the
      YOLO backbone's penultimate conv layer.
   3. **Otherwise:** dependency-free fallback that combines a Sobel edge
      map with a Gaussian centred on the detection box, blended with the
      original image using a jet colormap.
3. Returns `{ method, heatmap_base64, caption, detections,
   target_detection }`.

### 4.4 Live metrics

1. Every call to `DetectionService.detect` / `PipelineService` records a
   `MetricRecord`.
2. `GET /api/v1/metrics/summary` returns `{ model: { total_requests,
   avg_latency_ms, p50, p95, cache_hit_rate, avg_detections, last_seen } }`.
3. `GET /api/v1/metrics/recent?model=&limit=` returns recent records.
4. `POST /api/v1/metrics/reset` wipes the buffer — handy for benchmark
   runs.

---

## 5. Data contracts

All request / response shapes are Pydantic models under
`app/schemas/detection.py`. The key domain dataclasses live in
`app/models/base_model.py`:

- **`Detection`** — `id, label, confidence, bbox=(x1,y1,x2,y2), class_id?, mask_shape?, mask_rle?, extra`.
- **`InferenceResult`** — `model_name, detections[], inference_time_ms, image_shape, cached, metadata, raw`.
- **`MetricRecord`** — `model, latency_ms, detection_count, cached, timestamp (ISO 8601 UTC)`.

`InferenceResult.to_dict()` produces the exact JSON shape routers return
so clients see one consistent envelope regardless of which model was
called.

---

## 6. Extending the system

### Add a new model
1. Subclass `BaseDetectionModel` in `app/models/<name>.py`.
2. Implement `load_model()` and `predict(image, **kwargs) -> dict`.
   (Override `infer()` only if you need a non-standard result shape.)
3. Add an entry to `ModelType` and call
   `ModelFactory.register(ModelType.NEW, NewClass)` inside
   `_register_default_models()` in `app/models/model_factory.py`.
4. (Optional) Add a router in `app/routers/` and include it in
   `app/main.py::create_app`.

### Add a new persistence backend
1. Implement `IInferenceRepository` (or `IMetricsRepository`) in
   `app/repositories/`.
2. Choose it inside `_inference_repository()` / `_metrics_repository()`
   in `app/dependencies.py`.
3. Nothing else changes — services and routers already talk to the
   interface.

### Change auth policy
- Flip `AUTH_ENABLED=true`, set `API_KEYS` to a real key list, adjust
  `RATE_LIMIT_PER_MINUTE`. Routers automatically get protected.
- To require auth on a specific endpoint even when middleware is
  disabled, `Depends(require_api_key)` from `app/middleware/auth.py`.

---

## 7. Runtime dependencies

**Always required** (see `requirements.txt`)

- `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pydantic-settings`
- `torch`, `torchvision`, `opencv-python-headless`, `numpy`, `Pillow`
- `ultralytics` (YOLOv8), `transformers` (Grounding DINO),
  `segment-anything` (SAM)
- `python-dotenv`, `requests`, `aiofiles`

**Optional** — wrapped in `try/except` so absence is a silent no-op

- `structlog` — JSON logging (`LOG_JSON=true`)
- `redis` — used when `REDIS_URL` is set
- `motor` — used when `MONGO_URL` is set
- `grad-cam` (`pytorch-grad-cam`) — used by the Grad-CAM endpoint;
  otherwise the endpoint returns a Sobel + Gaussian saliency map
- `detectron2` — install from source per the note in
  `requirements.txt`; only exercised by the Detectron2 endpoints

**Testing**

- `pytest`, `pytest-asyncio`, `httpx` — the full test suite runs without
  loading real model weights thanks to `StubModel` in
  `tests/test_detection_service.py`.

---

## 8. Environment variables at a glance

Full documentation is in `.env.example`. High-signal knobs:

| Variable | Default | Effect |
|---|---|---|
| `AUTH_ENABLED` | `false` | Require `X-API-Key` |
| `API_KEYS` | `demo-key-12345` | Comma-separated valid keys |
| `RATE_LIMIT_PER_MINUTE` | `60` | Fixed-window limit |
| `CACHE_ENABLED` | `true` | Inference cache toggle |
| `REDIS_URL` | *(unset)* | Switch cache backend to Redis |
| `INFERENCE_CACHE_TTL` | `3600` | Cache TTL seconds |
| `METRICS_ENABLED` | `true` | Metrics collection toggle |
| `MONGO_URL` | *(unset)* | Persist metrics to MongoDB |
| `LOG_JSON` | `false` | Emit JSON logs |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `CORS_ORIGINS` | `*` | Comma-separated origins |
| `YOLO_MODEL_NAME` | `yolov8n.pt` | Ultralytics model file |
| `DETECTRON2_CONFIG` | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` | Detectron2 model zoo config |
| `GROUNDING_DINO_MODEL` | `IDEA-Research/grounding-dino-tiny` | HuggingFace model id |
| `SAM_MODEL_TYPE` | `vit_b` | `vit_b` / `vit_l` / `vit_h` |

---

## 9. Testing strategy

- `tests/test_repositories.py` — direct unit tests over the in-memory
  cache and metrics buffer (TTL, LRU eviction, p50 / p95 aggregation).
- `tests/test_detection_service.py` — registers a `StubModel` into
  `ModelFactory` so caching, metrics recording, and cache-hit detection
  are exercised in milliseconds without any real weights.
- `tests/test_new_endpoints.py` — HTTP-level assertions for the new
  `/metrics/*`, `/yolo/detect-base64`, `/openapi.json` routes.
- `tests/test_api.py` — existing endpoint-contract tests that still pass
  after the refactor (backwards-compatible response shapes preserved).

Run everything: `pytest -v`.

---

## 10. Deployment story

### 10.1 Current live setup

| Component | URL | Notes |
|---|---|---|
| Frontend | <https://object-detection-api-psi.vercel.app> | Vercel Hobby, always on |
| Backend | <https://unnecessarily-menispermaceous-rickey.ngrok-free.dev> | Local `uvicorn` behind an ngrok Free tunnel; live only when the maintainer's laptop is on |
| Swagger | `/docs` on the backend URL | |

**Why this shape**: PyTorch loaded in memory + a YOLO inference blows
past the 512 MB ceiling of every free serverless tier I evaluated
(Render, Fly, HF Spaces Docker CPU-basic). Running the backend on my own
machine and pushing it through an ngrok tunnel is the honest free-tier
answer that serves real inference. `frontend/src/lib/api.ts` sends
`ngrok-skip-browser-warning: true` on every request so ngrok's
interstitial never corrupts JSON responses.

**Rotation caveat**: ngrok Free hands out a new subdomain on every
restart. To keep the URL stable, claim a free static domain at
<https://dashboard.ngrok.com/domains> and run
`ngrok http --domain=<yours>.ngrok-free.dev 8000`.

### 10.2 Alternative deployment targets

- **Local** — `uvicorn app.main:app --reload`. Docs at `/docs`.
- **Docker** — GPU (`docker-compose up`) or CPU
  (`docker-compose --profile cpu up`). Multi-stage build in
  `Dockerfile`, honours `$PORT`.
- **Render.com (persistent, free-tier limited)** — `render.yaml` at the
  repo root is a Render Blueprint. Free tier serves health/docs/metrics
  but 502s on inference (OOM); Starter ($7/mo, 2 GB) fixes that.
- **Vercel (frontend)** — imports `frontend/` as a Next.js app,
  auto-deploys on every push to `main`. Single required env var:
  `NEXT_PUBLIC_API_BASE_URL`.
- **AWS ECS Fargate** — `deployment/aws/deploy.sh` builds the
  CloudFormation stack (VPC → ALB → ECS Fargate).
- **Kubernetes** — manifests in `deployment/k8s/` (Deployment, Service,
  Ingress, HPA). Readiness/liveness probes hit `/health` (never triggers
  weight loading — safe for orchestrators).

Full walk-through: [`DEPLOY.md`](DEPLOY.md).

---

## 11. Design decisions worth remembering

1. **Backward-compatible refactor.** The existing `predict()` method on
   every model was kept; the new `infer()` method wraps it. The response
   shape returned by routers is a superset of the pre-refactor shape.
   Existing tests still pass unmodified.

2. **Optional dependencies are truly optional.** Redis, Mongo,
   structlog, and `grad-cam` are all wrapped in `try/except`. Missing
   packages downgrade a feature rather than break the process.

3. **Lazy model loading everywhere.** No weights are loaded at process
   start. `/health` reports load status without triggering a load — this
   matters for K8s readiness probes.

4. **Metrics are cheap by default.** The in-memory ring buffer is
   process-local, `O(1)` per record, bounded in memory
   (`METRICS_WINDOW_SIZE`). Mongo is opt-in for persistence.

5. **Auth is opt-in.** The demo build should be openable in a browser
   with zero configuration. Production locks down with one env-var flip.

6. **Pipeline is a first-class model.** `PipelineService` records
   metrics under `grounding_dino+sam` so the benchmark dashboard treats
   the pipeline as its own comparable row.

7. **Explainability degrades gracefully.** The Sobel + Gaussian
   fallback still communicates "where the model looked" even when
   `pytorch-grad-cam` is absent — the endpoint never 500s.

---

## 12. Where to look when …

| I need to … | Look at |
|---|---|
| add a new endpoint | `app/routers/`, then include it in `app/main.py::create_app` |
| add a new model | `app/models/`, register in `model_factory.py::_register_default_models` |
| change cache TTL or size | `.env` (`INFERENCE_CACHE_TTL`, `INFERENCE_CACHE_MAX_ENTRIES`) |
| switch to Redis / Mongo | set `REDIS_URL` / `MONGO_URL` |
| require API keys | set `AUTH_ENABLED=true`, `API_KEYS=...` |
| add JSON logs | set `LOG_JSON=true`, install `structlog` |
| add a metric | `MetricRecord.now(...)` inside a service, then `IMetricsRepository.record(...)` |
| test without ML weights | use `StubModel` pattern from `tests/test_detection_service.py` |
| deploy on AWS | `deployment/aws/deploy.sh` |
| deploy on K8s | `kubectl apply -f deployment/k8s/*.yaml` |

---

*Last updated with the v2.0.0 clean-architecture refactor: `BaseDetectionModel` +
`ModelFactory`, `IInferenceRepository`, `IMetricsRepository`,
`DetectionService`, `PipelineService`, `ExplainabilityService`,
`MetricsService`, DI providers in `dependencies.py`, opt-in `AuthMiddleware`,
structured logging, `explain_router`, `metrics_router`, base64 detect
endpoint, and matching pytest coverage.*
