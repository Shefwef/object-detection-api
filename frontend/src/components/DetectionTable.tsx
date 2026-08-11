"use client";

import type { Detection } from "@/lib/types";
import { pickColor } from "@/lib/colors";

interface Props {
  detections: Detection[];
}

export default function DetectionTable({ detections }: Props) {
  if (detections.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No detections above threshold.</p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th className="py-2 pr-4">#</th>
            <th className="py-2 pr-4">Label</th>
            <th className="py-2 pr-4">Confidence</th>
            <th className="py-2 pr-4">Bounding box (x1, y1, x2, y2)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {detections.map((det, idx) => (
            <tr key={idx} className="text-slate-200">
              <td className="py-1.5 pr-4">
                <span
                  className="inline-block h-3 w-3 rounded-sm align-middle"
                  style={{ backgroundColor: pickColor(idx) }}
                />
                <span className="ml-2 tabular-nums text-slate-400">{idx + 1}</span>
              </td>
              <td className="py-1.5 pr-4 font-medium">
                {det.class_name || det.label || "object"}
              </td>
              <td className="py-1.5 pr-4 tabular-nums">
                {(det.confidence * 100).toFixed(1)}%
              </td>
              <td className="py-1.5 pr-4 font-mono text-xs text-slate-400">
                [{det.bbox.map((n) => Math.round(n)).join(", ")}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
