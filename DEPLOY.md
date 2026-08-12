# Deploy

End-to-end walk-through to ship this project as a public demo on **free
tiers only**. Two supported paths — the recommended one (currently live)
plus a persistent alternative when you need it.

- **Current live setup**: Vercel (frontend) + local FastAPI exposed via
  **ngrok** (backend). Serves full-fat inference. Requires your laptop
  to be on.
- **Alternative**: Vercel (frontend) + **Render.com** (backend). Always
  on. Free tier can serve health/docs/metrics but OOMs on real inference
  (`/api/v1/yolo/detect` etc.) — upgrade to Render Starter ($7/mo, 2 GB
  RAM) for the full stack.

Total setup time from a clean machine: **~15 min** for the ngrok path,
**~10 min** for the Render path.

---

## Live URLs (current deployment)

| Component | URL |
|---|---|
| Frontend | <https://object-detection-api-psi.vercel.app> |
| Backend | <https://unnecessarily-menispermaceous-rickey.ngrok-free.dev> |
| Swagger docs | <https://unnecessarily-menispermaceous-rickey.ngrok-free.dev/docs> |
| Health probe | <https://unnecessarily-menispermaceous-rickey.ngrok-free.dev/health> |
| Metrics | <https://unnecessarily-menispermaceous-rickey.ngrok-free.dev/api/v1/metrics/summary> |

Because ngrok Free hands out a new subdomain on every restart, the
backend URL above may rotate. If the site can't reach the backend, refer
to *Path A · Step 5* below for how to update the Vercel env after
restarting the tunnel.

---

## Path A · Vercel + local backend via ngrok (recommended, always free)

### Prerequisites

- Python 3.11+ (`py --version`)
- Node 20+ (only needed if you'll deploy the frontend yourself; not
  needed to point the existing Vercel deployment at your ngrok URL)
- A free ngrok account: <https://dashboard.ngrok.com/signup>
- A free Vercel account: <https://vercel.com>

### Step 1 · Install backend dependencies

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.19 `
            pydantic==2.10.4 pydantic-settings==2.7.1 opencv-python-headless==4.10.0.84 `
            numpy==1.24.3 Pillow structlog
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

Skips Detectron2 (source build, ~20 min) and Grounding DINO/SAM (large
downloads). YOLO alone is enough for the primary demo; add the others
later with:

```powershell
pip install transformers                                          # Grounding DINO
pip install "git+https://github.com/facebookresearch/segment-anything.git"   # SAM
```

If PowerShell refuses to run `Activate.ps1`, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 2 · Configure `.env` and start the backend

Copy the example, then edit:

```powershell
Copy-Item .env.example .env
```

Set at minimum:

```
DEVICE=cpu
CORS_ORIGINS=https://object-detection-api-psi.vercel.app
AUTH_ENABLED=false
LOG_JSON=false
```

Replace `CORS_ORIGINS` with your own Vercel URL. Start the backend:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Leave this terminal running. In a **second** PowerShell window, verify:

```powershell
curl.exe http://localhost:8000/health
```

Expect `{"status":"healthy",...}`.

### Step 3 · Install ngrok and add your token

```powershell
# Any one of:
winget install ngrok.ngrok
choco install ngrok
# …or download from https://ngrok.com/download and add to PATH
```

Then paste your token from
<https://dashboard.ngrok.com/get-started/your-authtoken>:

```powershell
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### Step 4 · Start the tunnel

In a **third** PowerShell window:

```powershell
ngrok http 8000
```

ngrok prints something like:

```
Forwarding  https://<something>.ngrok-free.dev -> http://localhost:8000
```

Copy that HTTPS URL. Verify from any terminal:

```powershell
curl.exe -H "ngrok-skip-browser-warning: true" https://<something>.ngrok-free.dev/health
```

Should return the health JSON.

> ⚠️ ngrok Free's URL changes on every restart. If you don't want to
> update Vercel every time, get a **static domain** free from
> <https://dashboard.ngrok.com/domains> (one per account) and run:
> `ngrok http --domain=my-static-domain.ngrok-free.dev 8000`.

### Step 5 · Point Vercel at the tunnel

1. <https://vercel.com/dashboard> → your project → **Settings →
   Environment Variables**.
2. Edit `NEXT_PUBLIC_API_BASE_URL` (or add it if missing):
   - Value: your ngrok HTTPS URL (no trailing slash)
   - Environments: **Production**, **Preview**, **Development** (all three)
3. Save.
4. **Deployments** tab → three-dot menu on the latest deployment →
   **Redeploy** → uncheck "Use existing build cache" → **Redeploy**.

Wait ~90 s for the new build.

### Step 6 · Turn off Vercel Deployment Protection (once)

Otherwise the site is only visible to your Vercel account.

1. Vercel project → **Settings → Deployment Protection**
2. **Vercel Authentication** → toggle **Off** (or "Only Preview
   Deployments" if you want a preview firewall)
3. Save

### Step 7 · Smoke test

Open the Vercel URL. Drop any image, tick **YOLOv8**, click **Run
detection**. First call takes 10–15 s because ultralytics downloads
`yolov8n.pt`. Subsequent calls are instant (cached).

---

## Path B · Vercel + Render.com (always-on, free-tier limited)

### Step 1 · Create the Render service

Two ways:

- **One-click Blueprint** (recommended): sign in at
  <https://dashboard.render.com> → **New → Blueprint** → connect this
  repo → **Apply**. Render reads `render.yaml` and provisions the Docker
  web service with the correct env vars.
- **Manual**: **New → Web Service** → pick this repo → runtime **Docker**,
  branch `main`, plan **Free**, health check `/health`. Add the env vars
  listed under *Environment variable summary* below.

### Step 2 · Wait for the first build (~5–8 min)

Render clones the repo, builds the image, and starts the container. The
**Live** badge appears when `/health` returns 200.

### Step 3 · Point Vercel at the Render URL

Same as *Path A · Step 5* but the value is `https://<your-service>.onrender.com`.

### Step 4 · Lock down CORS

Render dashboard → your service → **Environment** → edit
`CORS_ORIGINS` to your Vercel origin (no trailing slash). Save; Render
redeploys automatically.

### Known limitation

Render's free plan gives 512 MB RAM. With torch loaded (~400 MB) plus a
YOLO inference, the container OOMs and returns `502 Bad Gateway`.
Health, docs, root, and the metrics endpoints keep working. To serve
real inference on Render, upgrade to Starter ($7/mo, 2 GB RAM) — same
code, no changes required.

---

## Environment variable summary

### Backend

Set in your local `.env` (Path A) or the Render dashboard (Path B).

| Variable | Default / Recommended | Purpose |
|---|---|---|
| `DEVICE` | `cpu` | Force CPU on hosts without CUDA |
| `LOG_JSON` | `true` | Structured JSON logs |
| `LOG_LEVEL` | `INFO` | |
| `CACHE_ENABLED` | `true` | In-memory inference cache |
| `INFERENCE_CACHE_TTL` | `3600` | Cache TTL seconds |
| `METRICS_ENABLED` | `true` | Records per-model latency/throughput |
| `AUTH_ENABLED` | `false` | Turn on API-key auth |
| `API_KEYS` | *(unset)* | Comma-separated valid keys when auth on |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key/IP fixed window |
| `CORS_ORIGINS` | `*` → your Vercel origin | Lock down after go-live |
| `YOLO_MODEL_NAME` | `yolov8n.pt` | Smallest, works on 512 MB Render |
| `REDIS_URL` | *(unset)* | Optional — swap cache backend to Redis |
| `MONGO_URL` | *(unset)* | Optional — persist metrics to Mongo |

### Frontend (Vercel)

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Your ngrok URL (Path A) or Render URL (Path B). No trailing slash. |
| `NEXT_PUBLIC_API_KEY` | Only when the backend has `AUTH_ENABLED=true` |

### GitHub secrets

**None required.** Vercel and Render both use GitHub OAuth for deploys.
If you experimented with the earlier HF Spaces workflow, you can delete
`HF_TOKEN`, `HF_USERNAME`, `HF_SPACE_NAME` from the repo secrets.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Vercel URL bounces to a login page | Deployment Protection is on | Settings → Deployment Protection → turn off Vercel Authentication |
| Frontend can't reach backend, CORS error in devtools | Backend `CORS_ORIGINS` still `*` (some browsers block credentialled `*`) or wrong origin | Set `CORS_ORIGINS` to the exact Vercel URL, no trailing slash |
| Frontend fetch returns an HTML page | ngrok interstitial is being served | Confirm the `ngrok-skip-browser-warning` header is being sent (already patched in `frontend/src/lib/api.ts`) |
| First YOLO call takes 15 s | Ultralytics is downloading `yolov8n.pt` | Normal on first call; cached after |
| Render returns 502 on `/api/v1/*/detect` | Container OOM (512 MB tier) | Upgrade to Starter, or use Path A |
| ngrok URL changed after restart | Free tier gives ephemeral subdomains | Claim your free static domain at <https://dashboard.ngrok.com/domains> and use `ngrok http --domain=... 8000` |
| `429 Too Many Requests` | Rate limit tripped | Bump `RATE_LIMIT_PER_MINUTE` or turn off auth |

---

## Cost summary

- **Vercel Hobby**: $0/month.
- **ngrok Free**: $0/month. 1 GB egress / month, 1 static domain, 4 tunnels concurrently.
- **Render Free web service**: $0/month. 512 MB RAM, sleeps after 15 min idle.
- **Render Starter web service**: $7/month. 2 GB RAM, no sleep. Recommended if you need always-on inference.
- Upstash Redis free / MongoDB Atlas M0: both $0 for demo workloads.

Total for the current live setup: **$0/month** (Vercel + ngrok).
