# Object Detection Studio — Frontend

Next.js 15 + TypeScript + Tailwind UI for the [FastAPI backend](../README.md).

Three pages:

- **`/`** — upload an image, pick any subset of models (YOLOv8, Detectron2, Grounding DINO, G-DINO + SAM pipeline), see annotated bounding boxes plus a detections table.
- **`/compare`** — run all four models on the same image in parallel with a side-by-side latency chart.
- **`/metrics`** — live per-model latency, throughput, and cache-hit-rate dashboard (auto-refreshes every 5 s).

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local     # defaults proxy /api/backend/* -> http://localhost:8000
npm run dev
```

The FastAPI backend must be running on `localhost:8000` (see the root README). Requests hit `/api/backend/api/v1/...` which the Next.js dev server rewrites to `http://localhost:8000/api/v1/...`.

## Environment variables

| Var | Purpose | Default |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Where browser calls go. Point at the deployed FastAPI URL in production. | `/api/backend` |
| `NEXT_PUBLIC_API_KEY` | Sent as `X-API-Key` if the backend has `AUTH_ENABLED=true`. | *(empty)* |
| `API_PROXY_TARGET` | Dev-only. Backend URL the rewrite forwards to. | `http://localhost:8000` |

## Production build

```bash
npm run build
npm start
```

## Deploying to Vercel

1. Push `frontend/` to GitHub (already done as part of this monorepo).
2. Import the repository in Vercel; set **Root Directory** to `frontend`.
3. Add the environment variable `NEXT_PUBLIC_API_BASE_URL` pointing at your HF Space, e.g. `https://shefwef-object-detection-studio.hf.space`.
4. Deploy. Vercel will auto-redeploy on every push to `main`.

See [`../DEPLOY.md`](../DEPLOY.md) for the end-to-end walk-through (HF Spaces backend + Vercel frontend).

## Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout + nav
│   ├── page.tsx            # Detect (upload + run + results)
│   ├── compare/page.tsx    # Side-by-side comparison
│   ├── metrics/page.tsx    # Live metrics dashboard
│   └── globals.css
├── components/
│   ├── Nav.tsx
│   ├── ImageDropzone.tsx   # react-dropzone wrapper + preview
│   ├── ModelSelector.tsx
│   ├── PromptInput.tsx     # text prompt for G-DINO / pipeline
│   ├── AnnotatedCanvas.tsx # draws bounding boxes on a <canvas>
│   ├── DetectionTable.tsx
│   └── StatChip.tsx
└── lib/
    ├── api.ts              # fetch wrapper for every endpoint
    ├── types.ts            # TS mirrors of the FastAPI schemas
    ├── colors.ts           # bounding-box palette
    └── env.ts              # centralised env-var access
```
