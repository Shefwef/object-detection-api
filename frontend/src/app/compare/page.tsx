"use client";

import { useState } from "react";
import ImageDropzone from "@/components/ImageDropzone";
import PromptInput from "@/components/PromptInput";
import AnnotatedCanvas from "@/components/AnnotatedCanvas";
import StatChip from "@/components/StatChip";
import {
  detectYolo,
  detectDetectron2,
  detectGroundingDino,
  pipelineDetectSegment,
} from "@/lib/api";
import type { Detection, DetectionResponse, PipelineResponse } from "@/lib/types";

interface CellResult {
  key: "yolov8" | "detectron2" | "grounding_dino" | "pipeline";
  label: string;
  detections: Detection[];
  time_ms?: number;
  error?: string;
}

const CELLS: CellResult["key"][] = ["yolov8", "detectron2", "grounding_dino", "pipeline"];

export default function ComparePage() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("person . car . dog");
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<CellResult[]>([]);

  async function runAll() {
    if (!file) return;
    setBusy(true);
    setPreviewUrl(URL.createObjectURL(file));
    setResults([]);

    const cells: CellResult[] = [];
    async function safe(
      key: CellResult["key"],
      label: string,
      run: () => Promise<{ detections: Detection[]; time_ms?: number }>,
    ) {
      try {
        const r = await run();
        cells.push({ key, label, detections: r.detections, time_ms: r.time_ms });
      } catch (e) {
        cells.push({
          key,
          label,
          detections: [],
          error: e instanceof Error ? e.message : String(e),
        });
      }
      setResults([...cells]);
    }

    await Promise.all([
      safe("yolov8", "YOLOv8", async () => {
        const r: DetectionResponse = await detectYolo(file, { confidence: 0.25 });
        return { detections: r.detections, time_ms: r.inference_time_ms };
      }),
      safe("detectron2", "Detectron2", async () => {
        const r: DetectionResponse = await detectDetectron2(file, {
          confidence: 0.5,
          return_masks: false,
        });
        return { detections: r.detections, time_ms: r.inference_time_ms };
      }),
      safe("grounding_dino", "Grounding DINO", async () => {
        const r: DetectionResponse = await detectGroundingDino(file, prompt, {
          box_threshold: 0.35,
        });
        return { detections: r.detections, time_ms: r.inference_time_ms };
      }),
      safe("pipeline", "G-DINO + SAM", async () => {
        const r: PipelineResponse = await pipelineDetectSegment(file, prompt, {
          box_threshold: 0.35,
        });
        return { detections: r.detections, time_ms: r.inference_time_ms };
      }),
    ]);
    setBusy(false);
  }

  const ordered = CELLS.map((k) => results.find((r) => r.key === k)).filter(
    (r): r is CellResult => Boolean(r),
  );
  const maxMs = Math.max(1, ...ordered.map((r) => r.time_ms ?? 0));

  return (
    <div className="space-y-8">
      <section className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight text-white">Model comparison</h1>
        <p className="text-slate-400">
          Run one image through every model and see latency + detection count side-by-side.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <ImageDropzone file={file} onSelected={setFile} />
        <div className="space-y-3">
          <PromptInput value={prompt} onChange={setPrompt} disabled={busy} />
          <button
            type="button"
            className="btn-primary w-full justify-center"
            disabled={!file || !prompt.trim() || busy}
            onClick={runAll}
          >
            {busy ? "Running all models…" : "Run all models"}
          </button>
          <p className="text-xs text-slate-500">
            Requests fire in parallel; latency includes network + model.
          </p>
        </div>
      </section>

      {previewUrl && ordered.length > 0 && (
        <section className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {ordered.map((r) => (
              <div key={r.key} className="card p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">{r.label}</h3>
                  <span className="chip">{r.detections.length}</span>
                </div>
                {r.error ? (
                  <p className="text-xs text-red-300">{r.error}</p>
                ) : (
                  <AnnotatedCanvas imageUrl={previewUrl} detections={r.detections} maxWidth={360} />
                )}
                <div className="grid grid-cols-2 gap-2">
                  <StatChip
                    label="Latency"
                    value={r.time_ms !== undefined ? `${Math.round(r.time_ms)} ms` : "—"}
                  />
                  <StatChip label="Objects" value={r.detections.length} />
                </div>
              </div>
            ))}
          </div>

          <div className="card p-4">
            <h2 className="text-sm font-semibold text-white mb-2">Latency comparison</h2>
            <div className="space-y-2">
              {ordered.map((r) => (
                <div key={r.key} className="flex items-center gap-3 text-xs">
                  <span className="w-32 text-slate-300">{r.label}</span>
                  <div className="flex-1 h-2 rounded bg-white/5 overflow-hidden">
                    <div
                      className="h-full bg-accent"
                      style={{ width: `${((r.time_ms ?? 0) / maxMs) * 100}%` }}
                    />
                  </div>
                  <span className="w-16 text-right tabular-nums text-slate-400">
                    {r.time_ms !== undefined ? `${Math.round(r.time_ms)} ms` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
