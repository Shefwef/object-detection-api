"use client";

import type { ModelKey } from "@/lib/types";

export type SelectableModel = Exclude<ModelKey, "sam"> | "pipeline";

interface Props {
  selected: Set<SelectableModel>;
  onToggle(model: SelectableModel): void;
}

const options: {
  key: SelectableModel;
  title: string;
  subtitle: string;
  badge: string;
}[] = [
  {
    key: "yolov8",
    title: "YOLOv8",
    subtitle: "Fast single-stage detection",
    badge: "Detection",
  },
  {
    key: "detectron2",
    title: "Detectron2",
    subtitle: "Two-stage Mask R-CNN",
    badge: "Segmentation",
  },
  {
    key: "grounding_dino",
    title: "Grounding DINO",
    subtitle: "Open-set text-prompted detection",
    badge: "Text prompt",
  },
  {
    key: "pipeline",
    title: "G-DINO + SAM",
    subtitle: "Open-vocabulary segmentation",
    badge: "Pipeline",
  },
];

export default function ModelSelector({ selected, onToggle }: Props) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {options.map((opt) => {
        const active = selected.has(opt.key);
        return (
          <button
            key={opt.key}
            type="button"
            onClick={() => onToggle(opt.key)}
            className={`card flex flex-col items-start gap-1 px-4 py-3 text-left transition ${
              active
                ? "border-accent/60 bg-accent/10 shadow-[0_0_0_1px_rgba(79,124,255,0.4)]"
                : "hover:border-white/20"
            }`}
          >
            <div className="flex w-full items-center justify-between">
              <span className="text-sm font-semibold text-white">{opt.title}</span>
              <span className="chip">{opt.badge}</span>
            </div>
            <span className="text-xs text-slate-400">{opt.subtitle}</span>
            <span
              className={`mt-1 inline-flex h-4 w-4 items-center justify-center rounded border ${
                active
                  ? "border-accent bg-accent text-white"
                  : "border-white/20 text-transparent"
              }`}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </span>
          </button>
        );
      })}
    </div>
  );
}
