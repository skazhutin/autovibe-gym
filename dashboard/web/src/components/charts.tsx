/** Lightweight self-contained SVG charts (no external libraries). */
import { useState } from "react";

export function Sparkline({ data, w = 120, h = 34, tone = "accent" }: { data: number[]; w?: number; h?: number; tone?: "accent" | "green" | "dim" }) {
  if (data.length < 2) return <svg width={w} height={h} />;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - 3 - ((v - min) / span) * (h - 6);
    return [x, y];
  });
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${w},${h} L0,${h} Z`;
  const color = tone === "green" ? "var(--green)" : tone === "dim" ? "var(--text-faint)" : "var(--accent-ink)";
  return (
    <svg width={w} height={h}>
      <path d={area} fill={color} opacity={0.1} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function Donut({ value, total, size = 120, percent }: { value: number; total: number; size?: number; percent?: number | null }) {
  const stroke = 12;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const rawPct = percent ?? (total > 0 ? value / total : 0);
  const pct = Math.max(0, Math.min(1, rawPct));
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--surface-2)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="var(--accent)"
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          style={{ transition: "stroke-dashoffset .6s", transform: "rotate(-90deg)", transformOrigin: "center" }}
        />
      </svg>
      <div style={{ textAlign: "center" }}>
        <div className="mono" style={{ fontSize: 22, fontWeight: 700 }}>
          {value}/{total}
        </div>
        <div className="faint" style={{ fontSize: 11 }}>
          {Math.round(pct * 100)}%
        </div>
      </div>
    </div>
  );
}

interface BarDatum {
  label: string;
  value: number;
  best?: boolean;
  sub?: string;
}
export function BarChart({ data, height = 220, fmt }: { data: BarDatum[]; height?: number; fmt?: (v: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!data.length) return null;
  const max = Math.max(...data.map((d) => d.value)) || 1;
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 14, height, position: "relative", paddingTop: 8, minWidth: 0 }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: "1 1 0", minWidth: 0, display: "flex", flexDirection: "column", alignItems: "center", height: "100%", justifyContent: "flex-end", gap: 8 }}
          onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
          <div className="mono" style={{ fontSize: 12, fontWeight: 700, color: d.best ? "var(--accent-ink)" : "var(--text-dim)" }}>
            {fmt ? fmt(d.value) : d.value.toFixed(3)}
          </div>
          <div
            style={{
              width: "100%",
              maxWidth: 64,
              height: `${(d.value / max) * (height - 60)}px`,
              background: d.best ? "var(--accent)" : "var(--surface-2)",
              borderRadius: "8px 8px 0 0",
              border: d.best ? "none" : "1px solid var(--border)",
              transition: "height .5s cubic-bezier(.4,0,.2,1), filter .15s",
              filter: hover === i ? "brightness(0.96)" : "none",
            }}
          />
          <div className="faint" style={{ fontSize: 11, textAlign: "center", maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", width: "100%" }}>
            {d.label}
          </div>
        </div>
      ))}
    </div>
  );
}

interface ScatterPoint {
  x: number;
  y: number;
  label: string;
  highlight?: boolean;
}
export function Scatter({ points, w = 440, h = 240, xLabel, yLabel }: { points: ScatterPoint[]; w?: number; h?: number; xLabel: string; yLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const pad = 36;
  if (!points.length) return null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs, 0), xMax = Math.max(...xs) || 1;
  const yMin = Math.min(...ys, 0), yMax = Math.max(...ys) || 1;
  const sx = (x: number) => pad + ((x - xMin) / (xMax - xMin || 1)) * (w - pad - 12);
  const sy = (y: number) => h - pad - ((y - yMin) / (yMax - yMin || 1)) * (h - pad - 12);
  return (
    <div style={{ position: "relative" }}>
      <svg width={w} height={h} style={{ maxWidth: "100%" }}>
        <line x1={pad} y1={h - pad} x2={w - 8} y2={h - pad} stroke="var(--border)" />
        <line x1={pad} y1={8} x2={pad} y2={h - pad} stroke="var(--border)" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={hover === i ? 8 : 6}
            fill={p.highlight ? "var(--accent)" : "var(--text-faint)"}
            stroke={p.highlight ? "var(--accent-ink)" : "transparent"}
            style={{ cursor: "pointer", transition: "r .12s" }}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        <text x={(w + pad) / 2} y={h - 6} textAnchor="middle" fontSize={11} fill="var(--text-faint)">{xLabel}</text>
        <text x={12} y={h / 2} textAnchor="middle" fontSize={11} fill="var(--text-faint)" transform={`rotate(-90 12 ${h / 2})`}>{yLabel}</text>
      </svg>
      {hover !== null && (
        <div className="chart-tip" style={{ left: sx(points[hover].x) + 10, top: sy(points[hover].y) - 8 }}>
          <strong>{points[hover].label}</strong>
          <div>{xLabel}: {points[hover].x.toLocaleString()}</div>
          <div>{yLabel}: {points[hover].y.toFixed(1)}%</div>
        </div>
      )}
    </div>
  );
}

export function MiniHist({ data, w = 120, h = 36 }: { data: number[]; w?: number; h?: number }) {
  if (!data.length) return null;
  const max = Math.max(...data) || 1;
  const bw = w / data.length;
  return (
    <svg width={w} height={h}>
      {data.map((v, i) => (
        <rect key={i} x={i * bw + 1} y={h - (v / max) * h} width={Math.max(bw - 2, 1)} height={(v / max) * h} fill="var(--accent-ink)" opacity={0.55} rx={1} />
      ))}
    </svg>
  );
}
