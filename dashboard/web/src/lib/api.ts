/** Typed client for the dashboard backend (FastAPI on /api, proxied in dev). */

export type RunMode = "single" | "repeated" | "iterative" | "gym";
export type RunStatus = "success" | "failed" | "null" | "running";

export interface Run {
  id: string;
  shortId: string;
  runName?: string;
  model: string;
  modelId?: string;
  mode: RunMode;
  dataset: string;
  datasetDir?: string;
  status: RunStatus;
  score: number | null;
  metric?: string | null;
  baseline?: number | null;
  checklist: number;
  checklistTotal: number;
  checklistCoverage?: number | null;
  errors: number;
  step: number;
  steps: number | null;
  tokIn: number;
  tokOut: number;
  startedMs: number;
  endedMs?: number;
  dur: number | null;
  seed?: number | string | null;
  temp?: number | string | null;
  source?: "mlflow" | "live";
  mlflowId?: string | null;
  failReason?: string;
  command?: string;
}

export interface ModelRec {
  id: string;
  name: string;
  provider: string;
  baseUrl: string;
  ctx: number;
  temp?: number;
  maxTokens?: number;
  online: boolean | null;
  hasApiKey?: boolean;
  apiKeyEnv?: string;
}

export interface Dataset {
  id: string;
  name: string;
  task: string;
  metric: string;
  metricGoal: "min" | "max";
  rows: number;
  cols: number;
  target: string;
  source: string;
  desc: string;
  prepared: boolean;
  datasetDir: string;
  seed?: number;
  suite?: string | null;
}

export interface NotebookCell {
  type: "code" | "markdown";
  n?: number;
  code?: string;
  text?: string;
  outputs?: CellOutput[];
}
export interface CellOutput {
  type: "stdout" | "table" | "error" | "submit";
  text?: string;
  html?: string;
  ename?: string;
}

export type FeedbackChannel =
  | "runtime"
  | "contract"
  | "checklist"
  | "checklist-hint"
  | "terminal";

export interface TrajectoryStep {
  step: number;
  action: "code" | "validate" | "submit";
  title: string;
  code: string;
  budgetRemaining?: number;
  feedback: { ch: FeedbackChannel; text: string }[];
}

export interface ChecklistItem {
  id: string;
  label: string;
  desc: string;
  closed: boolean;
  closedStep: number | null;
}
export interface ChecklistData {
  items: ChecklistItem[];
  coverage: number | null;
  closed: number;
  total: number;
}

export interface RunError {
  step: number;
  cellId?: string;
  type: string;
  value: string;
  traceback: string;
  stderr: string;
}

export interface LogMessage {
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  action?: string;
}
export interface LogsData {
  messages: LogMessage[];
  processLog?: string;
}

export interface Health {
  status: string;
  mlflow_store_present: boolean;
  datasets_dir_present: boolean;
  python_bin_present: boolean;
}

export interface Settings {
  mlflow_tracking_uri: string;
  datasets_dir: string;
  default_mode: string;
  default_episode: string;
  theme: string;
  accent: string;
  radius: number;
  remote_enabled: boolean;
  remote_ssh: string;
  remote_ssh_opts: string;
  remote_repo: string;
  remote_python: string;
  remote_runs_dir: string;
  remote_password?: string;
  remote_has_password?: boolean;
}

export interface LaunchPayload {
  modelId?: string;
  model?: string;
  mode: RunMode;
  datasetId: string;
  budgetMode: "local" | "cloud";
  maxSteps?: number;
  maxTokens?: number;
  shots?: number;
  temp?: number;
  seed?: number;
}

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface ServerHealth {
  online: boolean;
  configured: boolean;
  servers: { baseUrl: string; online: boolean; status?: number; error?: string }[];
}

export const api = {
  health: () => req<Health>("/health"),
  serverHealth: () => req<ServerHealth>("/server-health"),

  getSettings: () => req<Settings>("/settings"),
  saveSettings: (s: Partial<Settings>) =>
    req<Settings>("/settings", { method: "PUT", body: JSON.stringify(s) }),
  remoteCheck: () =>
    req<{ ok: boolean; repo?: boolean; gym?: boolean; output?: string; error?: string }>(
      "/settings/remote-check",
      { method: "POST" }
    ),

  listRuns: () => req<Run[]>("/runs"),
  getRun: (id: string) => req<Run>(`/runs/${id}`),
  launchRun: (p: LaunchPayload) =>
    req<Run>("/runs", { method: "POST", body: JSON.stringify(p) }),
  stopRun: (id: string) => req<{ stopped: string }>(`/runs/${id}/stop`, { method: "POST" }),
  notebook: (id: string) => req<{ cells: NotebookCell[] }>(`/runs/${id}/notebook`),
  trajectory: (id: string) => req<TrajectoryStep[]>(`/runs/${id}/trajectory`),
  checklist: (id: string) => req<ChecklistData>(`/runs/${id}/checklist`),
  errors: (id: string) => req<RunError[]>(`/runs/${id}/errors`),
  logs: (id: string) => req<LogsData>(`/runs/${id}/logs`),

  listDatasets: () => req<Dataset[]>("/datasets"),
  getDataset: (id: string) => req<Dataset>(`/datasets/${id}`),
  datasetPreview: (id: string, split = "train", limit = 50) =>
    req<{ columns: string[]; rows: unknown[][]; total: number; shown: number }>(
      `/datasets/${id}/preview?split=${split}&limit=${limit}`
    ),
  datasetColumns: (id: string, split = "train") =>
    req<
      {
        name: string;
        dtype: string;
        kind: "numeric" | "categorical";
        missingPct: number;
        unique: number;
        hist: number[];
      }[]
    >(`/datasets/${id}/columns?split=${split}`),
  updateDataset: (id: string, body: Record<string, unknown>) =>
    req<Dataset>(`/datasets/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteDataset: (id: string) =>
    req<{ deleted: string }>(`/datasets/${id}`, { method: "DELETE" }),
  uploadDataset: (form: FormData) =>
    fetch(BASE + "/datasets", { method: "POST", body: form }).then((r) => {
      if (!r.ok) throw new Error("upload failed");
      return r.json() as Promise<Dataset>;
    }),

  listModels: () => req<ModelRec[]>("/models"),
  providers: () => req<string[]>("/models/providers"),
  createModel: (m: Partial<ModelRec> & { apiKey?: string }) =>
    req<ModelRec>("/models", { method: "POST", body: JSON.stringify(m) }),
  updateModel: (id: string, m: Partial<ModelRec> & { apiKey?: string }) =>
    req<ModelRec>(`/models/${id}`, { method: "PUT", body: JSON.stringify(m) }),
  deleteModel: (id: string) =>
    req<{ deleted: string }>(`/models/${id}`, { method: "DELETE" }),
  checkModel: (id: string) =>
    req<{ online: boolean; status?: number; error?: string }>(`/models/${id}/health`, {
      method: "POST",
    }),
};
