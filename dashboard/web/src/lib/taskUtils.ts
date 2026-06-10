import type { Task, TaskStatus } from "./api";

export const STATUS_TONE: Record<TaskStatus, "green" | "blue" | "red"> = {
  prepared: "green",
  partial: "blue",
  unprepared: "red",
};

export const STATUS_LABEL: Record<string, string> = {
  prepared: "подготовлен",
  partial: "частичный",
  unprepared: "не подготовлен",
};

export const TASK_LABEL: Record<string, string> = {
  auto: "auto",
  classification: "classification",
  regression: "regression",
  unknown: "unknown",
};

export const METRICS = {
  classification: ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "logloss"],
  regression: ["neg_rmse", "rmse", "mae", "r2"],
  auto: ["f1_macro", "f1_weighted", "neg_rmse", "rmse"],
};

export function inferGoal(metric: string): "max" | "min" {
  const m = metric.toLowerCase();
  if (m.startsWith("neg_")) return "max";
  return ["rmse", "rmsle", "mae", "mse", "logloss"].includes(m) ? "min" : "max";
}

export function statusOf(d: Task): TaskStatus {
  return d.status ?? (d.prepared ? "prepared" : d.hasTrain ? "partial" : "unprepared");
}

export function sourceText(d: Task): string {
  const value = d.source && d.source !== "-" ? d.source : d.sources?.[0]?.name || d.sources?.[0]?.url || "-";
  return !value || value === "source" ? "-" : value;
}
