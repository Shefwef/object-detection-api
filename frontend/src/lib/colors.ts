/**
 * Repeating color palette used to draw bounding boxes on the canvas.
 * Chosen to remain readable against arbitrary photo backgrounds.
 */

export const BOX_COLORS = [
  "#4f7cff",
  "#22c55e",
  "#ef4444",
  "#eab308",
  "#a855f7",
  "#06b6d4",
  "#f97316",
  "#84cc16",
  "#ec4899",
  "#14b8a6",
] as const;

export function pickColor(index: number): string {
  return BOX_COLORS[index % BOX_COLORS.length];
}
