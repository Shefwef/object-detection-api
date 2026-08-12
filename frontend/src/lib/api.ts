/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * Every function returns already-parsed JSON (or a Blob for visualize
 * endpoints). Errors surface as thrown Error objects with the backend's
 * `detail` string when available.
 */

import { API_BASE_URL, API_KEY } from "./env";
import type {
  DetectionResponse,
  MetricRecord,
  MetricsSummary,
  PipelineResponse,
} from "./types";

// ─── Helpers ──────────────────────────────────────────────────────────────

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    // Bypass ngrok's browser-warning interstitial when the backend is
    // exposed via ngrok. Ignored by every other server, so safe to send
    // unconditionally.
    "ngrok-skip-browser-warning": "true",
  };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  return headers;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* body may not be JSON */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

function url(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

// ─── Detection ────────────────────────────────────────────────────────────

export async function detectYolo(
  file: File,
  opts: { confidence?: number; iou_threshold?: number } = {},
): Promise<DetectionResponse> {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.confidence !== undefined) fd.append("confidence", String(opts.confidence));
  if (opts.iou_threshold !== undefined) fd.append("iou_threshold", String(opts.iou_threshold));
  const res = await fetch(url("/api/v1/yolo/detect"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handle<DetectionResponse>(res);
}

export async function detectDetectron2(
  file: File,
  opts: { confidence?: number; return_masks?: boolean } = {},
): Promise<DetectionResponse> {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.confidence !== undefined) fd.append("confidence", String(opts.confidence));
  if (opts.return_masks !== undefined) fd.append("return_masks", String(opts.return_masks));
  const res = await fetch(url("/api/v1/detectron2/detect"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handle<DetectionResponse>(res);
}

export async function detectGroundingDino(
  file: File,
  text_prompt: string,
  opts: { box_threshold?: number; text_threshold?: number } = {},
): Promise<DetectionResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("text_prompt", text_prompt);
  if (opts.box_threshold !== undefined) fd.append("box_threshold", String(opts.box_threshold));
  if (opts.text_threshold !== undefined) fd.append("text_threshold", String(opts.text_threshold));
  const res = await fetch(url("/api/v1/grounding-dino/detect"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handle<DetectionResponse>(res);
}

export async function pipelineDetectSegment(
  file: File,
  text_prompt: string,
  opts: { box_threshold?: number } = {},
): Promise<PipelineResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("text_prompt", text_prompt);
  if (opts.box_threshold !== undefined) fd.append("box_threshold", String(opts.box_threshold));
  const res = await fetch(url("/api/v1/pipeline/detect-and-segment"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handle<PipelineResponse>(res);
}

// ─── Visualize endpoints return an image blob ─────────────────────────────

export async function visualizeSam(file: File): Promise<Blob> {
  // SAM only exposes JSON responses (no server-drawn image endpoint) —
  // for now we surface auto segments via the JSON API and let the client
  // decide how to visualise. This helper is a placeholder for the future.
  throw new Error("SAM visualize is not implemented on the server.");
}

export async function detectAndVisualizeYolo(file: File, confidence = 0.25): Promise<Blob> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("confidence", String(confidence));
  const res = await fetch(url("/api/v1/yolo/detect-visualize"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`YOLO visualize failed: ${res.status}`);
  return res.blob();
}

export async function detectAndVisualizeDetectron2(file: File, confidence = 0.5): Promise<Blob> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("confidence", String(confidence));
  const res = await fetch(url("/api/v1/detectron2/detect-visualize"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Detectron2 visualize failed: ${res.status}`);
  return res.blob();
}

export async function pipelineVisualize(
  file: File,
  text_prompt: string,
  box_threshold = 0.35,
): Promise<Blob> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("text_prompt", text_prompt);
  fd.append("box_threshold", String(box_threshold));
  const res = await fetch(url("/api/v1/pipeline/detect-and-segment-visualize"), {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Pipeline visualize failed: ${res.status}`);
  return res.blob();
}

// ─── Metrics ──────────────────────────────────────────────────────────────

export async function getMetricsSummary(): Promise<MetricsSummary> {
  const res = await fetch(url("/api/v1/metrics/summary"), {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });
  return handle<MetricsSummary>(res);
}

export async function getRecentMetrics(model?: string, limit = 50): Promise<MetricRecord[]> {
  const qs = new URLSearchParams();
  if (model) qs.set("model", model);
  qs.set("limit", String(limit));
  const res = await fetch(url(`/api/v1/metrics/recent?${qs.toString()}`), {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });
  return handle<MetricRecord[]>(res);
}

// ─── Health ──────────────────────────────────────────────────────────────

export async function getHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(url("/health"), {
    method: "GET",
    cache: "no-store",
  });
  return handle<Record<string, unknown>>(res);
}
