# Deploy — Render (backend) + Vercel (frontend)

End-to-end walk-through to ship this project as a public live demo on
free tiers only.

- **Backend** → Render.com free web service (Docker, 512 MB RAM, sleeps
  after 15 min idle, no credit card).
- **Frontend** → Vercel free Hobby tier (Next.js, auto-deploys on push).
- **CI** → Render redeploys automatically on every push to `main`;
  Vercel does the same for the frontend.

Total setup time on clean accounts: **10–15 min**. Cost: **$0/month**.

---

## 0. Prerequisites

- The repo is pushed to GitHub on `main`.
- You have a Render account: <https://render.com> (sign in with GitHub — no card required).
- You have a Vercel account: <https://vercel.com> (sign in with GitHub — no card required).

---

## 1. Deploy the backend to Render

### Option A · One-click via Blueprint (recommended)

`render.yaml` at the repo root describes the service, so Render sets
everything up for you.

1. Sign in at <https://dashboard.render.com>.
2. Click **New → Blueprint**.
3. Connect your GitHub account (first time only) and pick this repo.
4. Render reads `render.yaml`, shows a summary (**web service · Docker · free plan**), and asks you to name the Blueprint. Accept the defaults.
5. Click **Apply**.

Render will now:
- Build the Docker image from your `Dockerfile` (~5–8 min the first time).
- Start the container listening on `$PORT` (Render sets it to `10000`).
- Assign a public URL: `https://object-detection-api.onrender.com` (if the name is taken, Render appends a random suffix).

### Option B · Manual web service (if you prefer clicking through)

If Blueprint doesn't detect `render.yaml`, do it by hand:

1. **New → Web Service** → pick your GitHub repo.
2. Fill in:
   - **Name**: `object-detection-api`
   - **Region**: Oregon (or nearest to you)
   - **Branch**: `main`
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: **Free**
   - **Health Check Path**: `/health`
3. Under **Environment**, add these variables (same list as `render.yaml`):

   | Key | Value |
   |---|---|
   | `DEVICE` | `cpu` |
   | `LOG_JSON` | `true` |
   | `LOG_LEVEL` | `INFO` |
   | `CACHE_ENABLED` | `true` |
   | `INFERENCE_CACHE_TTL` | `3600` |
   | `METRICS_ENABLED` | `true` |
   | `AUTH_ENABLED` | `false` |
   | `CORS_ORIGINS` | `*` *(narrow later)* |
   | `YOLO_MODEL_NAME` | `yolov8n.pt` |

4. **Create Web Service**.

## 2. Verify the backend

Once the Render dashboard shows **Live** (green dot), open:

- `https://<your-service>.onrender.com/` — should return API metadata + feature flags.
- `https://<your-service>.onrender.com/health` — should return `{"status":"healthy", ...}`.
- `https://<your-service>.onrender.com/docs` — Swagger UI loads.

**First `/api/v1/yolo/detect` call downloads YOLOv8n weights (~6 MB)** —
expect 5–15 s. Subsequent calls are hot.

### What works on free tier (512 MB RAM)

| Endpoint | Free-tier status |
|---|---|
| `/api/v1/yolo/detect` (all variants) | ✅ works |
| `/api/v1/yolo/detect-base64` | ✅ works |
| `/api/v1/metrics/summary` and `/recent` | ✅ works |
| `/api/v1/explain/gradcam` | ✅ works (uses saliency fallback) |
| `/api/v1/grounding-dino/detect` | ⚠️ may OOM (needs ~1 GB) |
| `/api/v1/detectron2/detect` | ❌ Detectron2 not installed in the image |
| `/api/v1/sam/segment-*` | ⚠️ may OOM (needs ~1 GB) |
| `/api/v1/pipeline/detect-and-segment` | ⚠️ depends on both DINO + SAM |

If you need everything, upgrade Render to **Starter ($7/mo, 2 GB RAM)** —
no code changes required.

---

## 3. Deploy the frontend to Vercel

1. Sign in at <https://vercel.com>.
2. **Add New → Project → Import Git Repository** → pick this repo.
3. **Root Directory**: set to `frontend` (very important — the repo isn't a Next.js app at its root).
4. Framework: Vercel auto-detects **Next.js**.
5. **Environment Variables** — add one:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | `https://<your-render-service>.onrender.com` |

6. Click **Deploy**.

Vercel gives you a URL like `https://object-detection-studio.vercel.app`.
Every subsequent push to `main` auto-redeploys.

---

## 4. Lock down CORS (recommended)

Now that you know your Vercel URL, tighten the backend so it only accepts
requests from your frontend.

1. Render dashboard → your service → **Environment** tab.
2. Edit `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=https://object-detection-studio.vercel.app
   ```
3. Click **Save, rebuild, and deploy**.

Render redeploys in ~2 min. Verify from the browser that your frontend
still talks to the API.

---

## 5. Optional — turn on API-key auth for production

1. Render env:
   ```
   AUTH_ENABLED=true
   API_KEYS=demo-key-12345,another-prod-key
   RATE_LIMIT_PER_MINUTE=30
   ```
2. Vercel env: `NEXT_PUBLIC_API_KEY=demo-key-12345`. Redeploy.
3. The middleware still keeps `/health`, `/docs`, `/redoc`, `/openapi.json`, and `/` open, so probes + Swagger keep working.

---

## Environment variable summary

### Render (backend)

Required — set via `render.yaml` or Render dashboard:

| Var | Value | Notes |
|---|---|---|
| `DEVICE` | `cpu` | Skip CUDA probe |
| `LOG_JSON` | `true` | Structured logs |
| `LOG_LEVEL` | `INFO` | |
| `CACHE_ENABLED` | `true` | |
| `INFERENCE_CACHE_TTL` | `3600` | Cache TTL seconds |
| `METRICS_ENABLED` | `true` | |
| `AUTH_ENABLED` | `false` | Set `true` for prod |
| `CORS_ORIGINS` | `*` → then Vercel URL | Lock down after step 4 |
| `YOLO_MODEL_NAME` | `yolov8n.pt` | Smallest to fit free tier |

Optional (only if you attach managed backends):

| Var | Value |
|---|---|
| `API_KEYS` | Comma-separated valid keys |
| `RATE_LIMIT_PER_MINUTE` | `60` |
| `REDIS_URL` | `redis://…` from Upstash / Render Redis |
| `MONGO_URL` | `mongodb+srv://…` from Atlas |

Render sets `PORT` automatically — the Dockerfile honours `$PORT`, so
you never need to set it manually.

### Vercel (frontend)

| Var | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Your Render URL, e.g. `https://object-detection-api.onrender.com` |
| `NEXT_PUBLIC_API_KEY` | *(only if `AUTH_ENABLED=true` on backend)* |

### GitHub secrets

**None required.** Render and Vercel both auto-deploy from the GitHub
integration — no tokens live in the repo.

### What you can delete from GitHub secrets (if you had them)

- `HF_TOKEN`, `HF_USERNAME`, `HF_SPACE_NAME` — no longer used.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails on Render with `Cannot find requirements` | Wrong branch selected | Confirm the service tracks `main` |
| Container starts then exits with SIGTERM | OOM on 512 MB — heavy model got called | Only call `/api/v1/yolo/*` on free tier; upgrade to Starter for others |
| `502 Bad Gateway` for ~30 s after 15 min idle | Free tier sleeps — first request wakes it | Wait and retry; upgrade to remove sleep |
| CORS error in browser after step 4 | `CORS_ORIGINS` mistyped (trailing slash, http vs https) | Copy the exact Vercel origin — no trailing slash |
| Frontend fetches fail with `404` at `/api/backend/...` | Missing `NEXT_PUBLIC_API_BASE_URL` on Vercel | Set it to the absolute Render URL and redeploy |
| First model call takes 15 s | YOLOv8 weight download | Normal on first call; cached thereafter |
| `429 Too Many Requests` | Rate limit hit | Raise `RATE_LIMIT_PER_MINUTE` or turn off auth |

---

## Cost summary

- **Render web service (free)**: $0. Sleeps after 15 min idle, ~30 s cold start.
- **Render web service (Starter)**: $7/mo — 2 GB RAM, no sleep, all models fit.
- **Vercel Hobby**: $0.
- **Upstash Redis free**: 10 k commands/day.
- **MongoDB Atlas M0**: 512 MB shared cluster, $0.

Permanent free demo: **$0/month** with cold starts.
Always-on with every model working: **$7/month** (Render Starter upgrade).
