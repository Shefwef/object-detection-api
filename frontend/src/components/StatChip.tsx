"use client";

interface Props {
  label: string;
  value: string | number;
  hint?: string;
}

export default function StatChip({ label, value, hint }: Props) {
  return (
    <div className="card px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-slate-400">{label}</p>
      <p className="text-lg font-semibold text-white tabular-nums">{value}</p>
      {hint && <p className="text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}
