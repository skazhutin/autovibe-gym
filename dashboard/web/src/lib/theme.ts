export interface Appearance {
  theme: "light" | "dark";
  accent: string;
  radius: number;
  animations: "on" | "off";
  overlayOpacity: number;
}

const KEY = "autovibe.appearance";

export const DEFAULT_APPEARANCE: Appearance = {
  theme: "light",
  accent: "#FFDD2D",
  radius: 18,
  animations: "on",
  overlayOpacity: 50,
};

export function loadAppearance(): Appearance {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_APPEARANCE, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return { ...DEFAULT_APPEARANCE };
}

/** Darken a hex color by a ratio toward black (for --accent-ink in light). */
function shade(hex: string, ratio: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/(.)/g, "$1$1") : h, 16);
  const r = (n >> 16) & 255,
    g = (n >> 8) & 255,
    b = n & 255;
  const f = (c: number) =>
    Math.round(ratio < 0 ? c * (1 + ratio) : c + (255 - c) * ratio)
      .toString(16)
      .padStart(2, "0");
  return `#${f(r)}${f(g)}${f(b)}`;
}

function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/(.)/g, "$1$1") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

export function applyAppearance(a: Appearance): void {
  const root = document.documentElement;
  root.setAttribute("data-theme", a.theme);
  root.setAttribute("data-motion", a.animations);
  root.style.setProperty("--accent", a.accent);
  root.style.setProperty(
    "--accent-ink",
    a.theme === "dark" ? shade(a.accent, 0.18) : shade(a.accent, -0.42)
  );
  root.style.setProperty("--accent-soft", withAlpha(a.accent, 0.22));
  root.style.setProperty("--accent-wash", withAlpha(a.accent, 0.1));
  root.style.setProperty("--radius", `${a.radius}px`);
  root.style.setProperty("--overlay-pct", `${a.overlayOpacity}%`);
  const blurPx = ((100 - a.overlayOpacity) / 100) * 10;
  root.style.setProperty("--overlay-blur", blurPx > 0 ? `blur(${blurPx.toFixed(1)}px)` : "none");
  localStorage.setItem(KEY, JSON.stringify(a));
}
