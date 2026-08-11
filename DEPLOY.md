# Deploy — Hugging Face Spaces (backend) + Vercel (frontend)

End-to-end walk-through to ship this project as a public live demo.

- **Backend** → Hugging Face Space (Docker SDK, free CPU or paid T4 GPU).
- **Frontend** → Vercel (Next.js, free tier).
- **CI** → GitHub Actions auto-syncs the backend to the Space on every push to `main`; Vercel auto-deploys the frontend on every push.

Total setup time on a clean account: **15–25 min**.

---

## 1. Create the Hugging Face Space (backend)

1. Sign in at <https://huggingface.co>.
2. **New Space** → choose **Docker** SDK, **CPU basic (free)** for the demo (upgrade to a T4 later if needed).
3. Space name: `object-detection-studio` (or your preference — the deploy workflow reads it from a secret).
4. Visibility: Public.
5. Skip the wizard's file-upload step — the GitHub Action will push the container config.
6. Note your Space URL: `https://<username>-<space-name>.hf.space` (e.g. `https://shefwef-object-detection-studio.hf.space`).

## 2. Create a Hugging Face access token

1. <https://huggingface.co/settings/tokens> → **New token** → role **Write** → copy the value once.
2. This becomes the `HF_TOKEN` secret in step 3.

## 3. Add GitHub secrets

Repository → **Settings → Secrets and variables → Actions → New repository secret**.

| Secret | Value |
|---|---|
| `HF_TOKEN` | Token from step 2. |
| `HF_USERNAME` | Your Hugging Face username (e.g. `shefwef`). |
| `HF_SPACE_NAME` | The Space's slug (e.g. `object-detection-studio`). |

That's it — the `deploy-hf.yml` workflow will fire on the next push to `main` that touches `app/`, `requirements.txt`, `Dockerfile`, or `deployment/hf/**`.

Prefer to trigger manually the first time? **Actions tab → Deploy backend to Hugging Face Spaces → Run workflow**.

## 4. Set backend environment variables on the Space

Hugging Face Space → **Settings → Variables and secrets → New variable**. Everything below is optional — the Space runs with defaults if omitted.

| Variable | Recommended value | Why |
|---|---|---|
| `CORS_ORIGINS` | `https://your-vercel-url.vercel.app` | Once you know your Vercel URL, lock CORS down. During bring-up you can leave the default `*`. |
| `LOG_JSON` | `true` | Structured logs in the Space log viewer. |
| `DEVICE` | `cpu` on free tier, `auto` on GPU tier | Skips the futile CUDA probe on CPU-only Spaces. |
| `AUTH_ENABLED` | `false` | Keep the demo open. Flip to `true` + set `API_KEYS` when you're ready. |
| `INFERENCE_CACHE_TTL` | `3600` | Default is fine. |

If you attach an Upstash / Atlas free tier, add:

| Variable | Value |
|---|---|
| `REDIS_URL` | `redis://default:<pw>@<host>:6379/0` |
| `MONGO_URL` | `mongodb+srv://<user>:<pw>@<cluster>/cv_detection` |

## 5. Verify the Space

Once the Action's `push-to-space` job goes green:

- Open `https://<username>-<space-name>.hf.space/docs` — Swagger should load.
- `https://<username>-<space-name>.hf.space/health` should return `{"status":"healthy",...}`.
- The **first** call to `/api/v1/yolo/detect` will download YOLOv8 weights (~6 MB) — expect a few seconds. Subsequent calls are cached.

**Cold-start note:** free-tier Spaces sleep after inactivity; the first hit after sleep can take ~30 s while the container boots.

---

## 6. Deploy the frontend on Vercel

1. Sign in at <https://vercel.com>.
2. **Add New → Project → Import Git Repository** → pick this repo.
3. **Root Directory**: `frontend`. Vercel auto-detects Next.js.
4. **Environment Variables**:

   | Var | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | `https://<username>-<space-name>.hf.space` |
   | `NEXT_PUBLIC_API_KEY` | *(leave empty unless auth enabled)* |

5. **Deploy**.

Vercel gives you a URL like `https://object-detection-studio.vercel.app`. Every subsequent push to `main` auto-redeploys.

## 7. Lock down CORS

Now that you know the Vercel URL, go back to **HF Space → Settings → Variables** and set:

```
CORS_ORIGINS=https://object-detection-studio.vercel.app
```

Restart the Space (Settings → Restart). The backend now only accepts browser requests from your Vercel deployment.

---

## 8. Optional — turn on API-key auth for production

1. HF Space Variables:
   ```
   AUTH_ENABLED=true
   API_KEYS=demo-key-12345,another-prod-key
   RATE_LIMIT_PER_MINUTE=30
   ```
2. Vercel env: set `NEXT_PUBLIC_API_KEY=demo-key-12345`. Redeploy.
3. The middleware still keeps `/health`, `/docs`, `/redoc`, `/openapi.json`, and `/` open, so probes and Swagger keep working.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `403` on Space push | Wrong `HF_TOKEN` scope | Re-issue token with **Write** role. |
| Space builds but `/docs` returns nothing | Space is still installing torch (~2 min) | Wait, then check Space logs. |
| Frontend calls fail with CORS error | `CORS_ORIGINS` still `*` and browser blocks credentials, or set to the wrong URL | Set `CORS_ORIGINS` to the exact Vercel origin (no trailing slash). |
| Frontend fetches `/api/backend/...` in production and 404s | Missing `NEXT_PUBLIC_API_BASE_URL` on Vercel | Set it to the absolute HF Space URL and redeploy. |
| `429 Too Many Requests` | Rate limit tripped | Raise `RATE_LIMIT_PER_MINUTE` or disable auth. |
| Detectron2 endpoints 500 | Not installed on the Space | Free HF Spaces skip Detectron2 by default (heavy source install). YOLO / DINO / SAM / pipeline still work. |

---

## Cost summary

- **HF Space CPU basic**: free, sleeps after inactivity.
- **HF Space CPU upgrade** (persistent, faster): ~$0.03/hr.
- **HF Space Nvidia T4 small**: ~$0.60/hr — turn on only during demo sessions if you want real-time YOLO.
- **Vercel Hobby**: free.
- **Upstash Redis free tier**: 10k commands/day (plenty for a demo).
- **MongoDB Atlas M0**: free 512 MB shared cluster.

Total to run permanently: **$0/month** on free tiers with tolerable cold starts. Recommended when actively demoing: bump the Space to CPU upgrade for a few hours.
