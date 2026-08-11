"use client";

import { useEffect, useState } from "react";
import StatChip from "@/components/StatChip";
import { getMetricsSummary, getRecentMetrics } from "@/lib/api";
import type { MetricsSummary, MetricRecord } from "@/lib/types";

const REFRESH_MS = 5000;

export default function MetricsPage() {
  const [summary, setSummary] = useState<MetricsSummary>({});
  const [recent, setRecent] = useState<MetricRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const [s, r] = await Promise.all([getMetricsSummary(), getRecentMetrics(undefined, 20)]);
        if (cancelled) return;
        setSummary(s);
        setRecent(r);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    tick();
    const t = setInterval(tick, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const rows = Object.entries(summary);
  const maxLatency = Math.max(1, ...rows.map(([, r]) => r.avg_latency_ms));

  return (
    <div className="space-y-8">
      <section className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight text-white">Live metrics</h1>
        <p className="text-slate-400">
          Per-model latency, throughput, and cache-hit rate. Auto-refreshes every {REFRESH_MS / 1000}s.
        </p>
      </section>

      {error && (
        <div className="card border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-slate-400">Loading…</p>
      ) : rows.length === 0 ? (
        <div className="card p-6 text-center text-slate-400">
          <p>No metrics yet.</p>
          <p className="text-xs text-slate-500">
            Run detection on the home page and come back — the dashboard populates automatically.
          </p>
        </div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {rows.map(([model, row]) => (
              <div key={model} className="card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">{model}</h3>
                  <span className="chip">{row.total_requests} runs</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <StatChip
                    label="Avg latency"
                    value={`${Math.round(row.avg_latency_ms)} ms`}
                  />
                  <StatChip
                    label="p95"
                    value={row.p95_latency_ms ? `${Math.round(row.p95_latency_ms)} ms` : "—"}
                  />
                  <StatChip
                    label="Cache hit"
                    value={`${Math.round(row.cache_hit_rate * 100)}%`}
                  />
                  <StatChip label="Avg objects" value={row.avg_detections.toFixed(1)} />
                </div>
                <div className="h-2 rounded bg-white/5 overflow-hidden">
                  <div
                    className="h-full bg-accent"
                    style={{ width: `${(row.avg_latency_ms / maxLatency) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="card p-4">
            <h2 className="text-sm font-semibold text-white mb-3">Recent inferences</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="py-2 pr-4">Time (UTC)</th>
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Latency</th>
                    <th className="py-2 pr-4">Detections</th>
                    <th className="py-2 pr-4">Cached</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {recent.map((r, i) => (
                    <tr key={i} className="text-slate-200">
                      <td className="py-1.5 pr-4 font-mono text-xs text-slate-400">
                        {new Date(r.timestamp).toISOString().slice(11, 19)}
                      </td>
                      <td className="py-1.5 pr-4">{r.model}</td>
                      <td className="py-1.5 pr-4 tabular-nums">{Math.round(r.latency_ms)} ms</td>
                      <td className="py-1.5 pr-4 tabular-nums">{r.detection_count}</td>
                      <td className="py-1.5 pr-4">{r.cached ? "yes" : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
