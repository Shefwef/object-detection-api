"use client";

interface Props {
  value: string;
  onChange(value: string): void;
  disabled?: boolean;
  placeholder?: string;
}

export default function PromptInput({
  value,
  onChange,
  disabled,
  placeholder = "e.g. person wearing helmet . red car",
}: Props) {
  return (
    <div className="card p-3">
      <label className="block text-xs uppercase tracking-wider text-slate-400 mb-1">
        Text prompt <span className="text-slate-500">· Grounding DINO / Pipeline</span>
      </label>
      <input
        type="text"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-500 disabled:opacity-50"
      />
    </div>
  );
}
