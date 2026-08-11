/**
 * TypeScript mirrors of the FastAPI response contracts.
 * Kept intentionally loose (Number | undefined etc.) so that the frontend
 * still renders when the backend adds new optional fields.
 */

export type ModelKey =
  | "yolov8"
  | "detectron2"
  | "grounding_dino"
  | "sam"
  | "pipeline";

export interface Detection {
  id: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  confidence: number;
  class_id?: number | null;
  class_name?: string | null;
  label?: string | null;
  mask_shape?: [number, number] | null;
}

export interface DetectionResponse {
  model: string;
  detections: Detection[];
  count: number;
  image_shape: [number, number]; // [height, width]
  inference_time_ms?: number;
  cached?: boolean;
  text_prompt?: string;
  metadata?: Record<string, unknown>;
}

export interface Segment {
  id: number;
  score?: number;
  area: number;
  mask_shape: [number, number];
  bbox?: [number, number, number, number];
}

export interface PipelineResponse {
  detection_model: string;
  segmentation_model: string;
  text_prompt: string;
  detections: Detection[];
  segments: Segment[];
  detection_count: number;
  segment_count: number;
  image_shape: [number, number];
  inference_time_ms?: number;
  message?: string;
}

export interface MetricRow {
  total_requests: number;
  cache_hit_rate: number;
  avg_latency_ms: number;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  avg_detections: number;
  last_seen?: string | null;
}

export type MetricsSummary = Record<string, MetricRow>;

export interface MetricRecord {
  model: string;
  latency_ms: number;
  detection_count: number;
  cached: boolean;
  timestamp: string;
}
