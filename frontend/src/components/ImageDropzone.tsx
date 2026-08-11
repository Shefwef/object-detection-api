"use client";

import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";

interface Props {
  onSelected(file: File): void;
  file: File | null;
}

export default function ImageDropzone({ onSelected, file }: Props) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[]) => {
      const f = accepted[0];
      if (!f) return;
      const url = URL.createObjectURL(f);
      setPreviewUrl(url);
      onSelected(f);
    },
    [onSelected],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"] },
    multiple: false,
    onDrop,
  });

  const preview = useMemo(() => {
    if (previewUrl) return previewUrl;
    if (file) return URL.createObjectURL(file);
    return null;
  }, [previewUrl, file]);

  return (
    <div
      {...getRootProps()}
      className={`card flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 p-6 text-center transition ${
        isDragActive ? "border-accent bg-white/5" : "hover:border-white/20"
      }`}
    >
      <input {...getInputProps()} />
      {preview ? (
        <>
          <img
            src={preview}
            alt="preview"
            className="max-h-48 rounded-lg border border-white/10 object-contain"
          />
          <p className="text-xs text-slate-400">
            {file?.name} · {file ? `${(file.size / 1024).toFixed(1)} KB` : ""}
            <br />
            <span className="text-slate-500">Drop another to replace</span>
          </p>
        </>
      ) : (
        <>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-slate-500"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div>
            <p className="text-sm text-slate-200">Drop an image here or click to browse</p>
            <p className="text-xs text-slate-500">JPG · PNG · WEBP · BMP · up to 10 MB</p>
          </div>
        </>
      )}
    </div>
  );
}
