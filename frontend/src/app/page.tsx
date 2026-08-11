"use client";

import { useMemo, useState } from "react";
import ImageDropzone from "@/components/ImageDropzone";
import ModelSelector, { SelectableModel } from "@/components/ModelSelector";
import PromptInput from "@/components/PromptInput";
import AnnotatedCanvas from "@/components/AnnotatedCanvas";
import DetectionTable from "@/components/DetectionTable";
import StatChip from "@/components/StatChip";
import {
  detectYolo,
  detectDetectron2,
  detectGroundingDino,
  pipelineDetectSegment,
} from "@/lib/api";
import type { Detection, DetectionResponse, PipelineResponse } from "@/lib/types";

type Result =
  | { kind: "detection"; model: SelectableModel; data: DetectionResponse }
  | { kind: "pipeline"; model: "pipeline"; data: PipelineResponse };

const DEFAULT_SELECTED: Set<SelectableModel> = new Set(["yolov8"]);

export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Set<SelectableModel>>(new Set(DEFAULT_SELECTED));
  const [prompt, setPrompt] = useState("person . car . dog");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [activeTab, setActiveTab] = useState<SelectableModel | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  const needsPrompt = selected.has("grounding_dino") || selected.has("pipeline");

  function toggleModel(m: SelectableModel) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(m)) next.delete(m);
      else next.add(m);
      return next;
    });
  }

  async function runDetection() {
    if (!file || selected.size === 0) return;
    setBusy(true);
    setError(null);
    setResults([]);

    const jobs: Promise<Result>[] = [];
    for (const m of selected) {
      if (m === "yolov8") {
        jobs.push(detectYolo(file, { confidence: 0.25 }).then((data) => ({ kind: "detection", model: m, data })));
      } else if (m === "detectron2") {
        jobs.push(
          detectDetectron2(file, { confidence: 0.5, return_masks: false }).then((data) => ({
            kind: "detection",
            model: m,
            data,
          })),
        );
      } else if (m === "grounding_dino") {
        jobs.push(
          detectGroundingDino(file, prompt, { box_threshold: 0.35 }).then((data) => ({
            kind: "detection",
            model: m,
            data,
          })),
        );
      } else if (m === "pipeline") {
        jobs.push(
          pipelineDetectSegment(file, prompt, { box_threshold: 0.35 }).then((data) => ({
            kind: "pipeline",
            model: "pipeline",
            data,
          })),
        );
      }
    }

    const settled = await Promise.allSettled(jobs);
    const collected: Result[] = [];
    const errors: string[] = [];
    for (const s of settled) {
      if (s.status === "fulfilled") collected.push(s.value);
      else errors.push(s.reason instanceof Error ? s.reason.message : String(s.reason));
    }
    setResults(collected);
    setActiveTab(collected[0]?.model ?? null);
    if (errors.length) setError(errors.join(" · "));
    setBusy(false);
  }

  return (
    <div className="space-y-8">
      <section className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Detect anything. Compare every model.
        </h1>
        <p className="text-slate-400">
          Upload an image, choose one or more models, and see them run side-by-side.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <ImageDropzone file={file} onSelected={setFile} />
        <div className="space-y-3">
          <div className="card p-4 space-y-1">
            <p className="text-xs uppercase tracking-wider text-slate-400">Run</p>
            <p className="text-sm text-slate-300">
              {selected.size} model{selected.size === 1 ? "" : "s"} selected
            </p>
            <button
              type="button"
              className="btn-primary mt-2 w-full justify-center"
              onClick={runDetection}
              disabled={!file || selected.size === 0 || (needsPrompt && !prompt.trim()) || busy}
            >
              {busy ? "Running…" : "Run detection"}
            </button>
            {needsPrompt && !prompt.trim() && (
              <p className="text-[11px] text-amber-400">
                A text prompt is required for Grounding DINO / Pipeline.
              </p>
            )}
          </div>
          <div className="card p-3 text-xs text-slate-400 space-y-1">
            <p className="text-slate-300 font-medium">Tips</p>
            <p>· Detectron2 &amp; SAM download weights on first call — expect a warm-up.</p>
            <p>· Results are cached by image hash — repeat runs are instant.</p>
          </div>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Model selection
        </h2>
        <ModelSelector selected={selected} onToggle={toggleModel} />
        {needsPrompt && (
          <PromptInput value={prompt} onChange={setPrompt} disabled={busy} />
        )}
      </section>

      {error && (
        <div className="card border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {results.length > 0 && previewUrl && (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Results
          </h2>

          <div className="flex flex-wrap gap-1">
            {results.map((r) => (
              <button
                key={r.model}
                type="button"
                onClick={() => setActiveTab(r.model)}
                className={`rounded-lg px-3 py-1.5 text-sm transition ${
                  activeTab === r.model
                    ? "bg-white/10 text-white"
                    : "text-slate-300 hover:bg-white/5"
                }`}
              >
                {labelForModel(r.model)}
                <span className="ml-2 chip">
                  {r.kind === "detection" ? r.data.count : r.data.detection_count}
                </span>
              </button>
            ))}
          </div>

          {results
            .filter((r) => r.model === activeTab)
            .map((r) => (
              <ResultView key={r.model} result={r} previewUrl={previewUrl} />
            ))}
        </section>
      )}
    </div>
  );
}

function ResultView({ result, previewUrl }: { result: Result; previewUrl: string }) {
  const detections: Detection[] =
    result.kind === "detection" ? result.data.detections : result.data.detections;
  const timeMs =
    result.kind === "detection"
      ? result.data.inference_time_ms
      : result.data.inference_time_ms;
  const cached = result.kind === "detection" ? result.data.cached : false;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <div className="space-y-3">
        <AnnotatedCanvas imageUrl={previewUrl} detections={detections} />
        <div className="grid grid-cols-3 gap-2">
          <StatChip label="Detections" value={detections.length} />
          <StatChip
            label="Latency"
            value={timeMs !== undefined ? `${Math.round(timeMs)} ms` : "—"}
          />
          <StatChip label="Cached" value={cached ? "yes" : "no"} />
        </div>
      </div>
      <div className="card p-4 space-y-2">
        <h3 className="text-sm font-semibold text-white">
          {labelForModel(result.model)} · Detections
        </h3>
        <DetectionTable detections={detections} />
      </div>
    </div>
  );
}

function labelForModel(m: SelectableModel): string {
  switch (m) {
    case "yolov8":
      return "YOLOv8";
    case "detectron2":
      return "Detectron2";
    case "grounding_dino":
      return "Grounding DINO";
    case "pipeline":
      return "G-DINO + SAM";
  }
}
