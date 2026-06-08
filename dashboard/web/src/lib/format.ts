import type { Run, RunMode, RunStatus } from "./api";

export const MODE_LABELS: Record<RunMode, string> = {
  single: "Single-shot",
  repeated: "Multi-shot",
  iterative: "Free gym",
  gym: "Directive gym",
  fixed: "Fixed gym",
  batch: "Набор режимов",
};

export const MODE_SHORT: Record<RunMode, string> = {
  single: "Single-shot",
  repeated: "Multi-shot",
  iterative: "Free gym",
  gym: "Directive gym",
  fixed: "Fixed gym",
  batch: "Batch",
};

export const STATUS_LABELS: Record<RunStatus, string> = {
  success: "Успех",
  failed: "Ошибка",
  null: "Без сабмита",
  running: "Идёт",
};

/** Score precision follows the metric (RMSE 1dp, RMSLE 4dp, else 3dp). */
export function formatScore(score: number | null | undefined, metric?: string | null): string {
  if (score === null || score === undefined) return "—";
  const m = (metric ?? "").toLowerCase();
  if (m.includes("rmsle")) return score.toFixed(4);
  if (m.includes("rmse") || m.includes("mae") || m.includes("mse")) return score.toFixed(1);
  return score.toFixed(3);
}

/** Improvement over baseline as %, respecting metric direction. */
export function improvementPct(run: Run): number | null {
  if (run.score === null || run.score === undefined || !run.baseline) return null;
  const goalMin = (run.metric ?? "").toLowerCase().match(/rmse|rmsle|mae|mse|logloss/) && !(run.metric ?? "").startsWith("neg_");
  const delta = goalMin
    ? (run.baseline - run.score) / Math.abs(run.baseline)
    : (run.score - run.baseline) / Math.abs(run.baseline);
  return delta * 100;
}

export function timeAgo(ms: number | undefined): string {
  if (!ms) return "—";
  const diff = Date.now() - ms;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "только что";
  if (min < 60) return `${min} мин назад`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h} ч назад`;
  const d = Math.floor(h / 24);
  return `${d} дн назад`;
}

export function formatDuration(sec: number | null | undefined): string {
  if (sec === null || sec === undefined) return "—";
  if (sec < 60) return `${Math.round(sec)} с`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return s ? `${m} мин ${s} с` : `${m} мин`;
}

export function formatTokens(n: number | undefined): string {
  if (!n) return "0";
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(n);
}
