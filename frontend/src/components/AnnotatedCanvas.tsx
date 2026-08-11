"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "@/lib/types";
import { pickColor } from "@/lib/colors";

interface Props {
  imageUrl: string;
  detections: Detection[];
  maxWidth?: number;
}

/**
 * Draws bounding boxes + labels on top of the source image.
 *
 * The canvas keeps the image's native aspect ratio; a CSS max-width caps
 * the rendered size so results stay legible on any screen.
 */
export default function AnnotatedCanvas({ imageUrl, detections, maxWidth = 720 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((det, idx) => {
        const [x1, y1, x2, y2] = det.bbox;
        const w = Math.max(1, x2 - x1);
        const h = Math.max(1, y2 - y1);
        const color = pickColor(idx);

        ctx.lineWidth = Math.max(2, Math.round(img.width / 400));
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, w, h);

        const label = `${det.class_name || det.label || "object"} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `bold ${Math.max(12, Math.round(img.width / 60))}px Inter, sans-serif`;
        const metrics = ctx.measureText(label);
        const padding = 6;
        const labelH = Math.max(16, Math.round(img.width / 50));

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - labelH - 4, metrics.width + padding * 2, labelH + 4);

        ctx.fillStyle = "#0b1020";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x1 + padding, y1 - labelH / 2 - 2);
      });
    };
    img.src = imageUrl;
  }, [imageUrl, detections]);

  return (
    <canvas
      ref={canvasRef}
      className="max-w-full h-auto rounded-lg border border-white/10 bg-black/20"
      style={{ maxWidth }}
    />
  );
}
