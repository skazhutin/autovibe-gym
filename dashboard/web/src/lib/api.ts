/** Typed client for the dashboard backend (FastAPI on /api, proxied in dev). */

export type RunMode = "single" | "repeated" | "free" | "directive" | "fixed" | "batch";
export type LaunchRunMode = "single" | "repeated" | "free" | "directive" | "fixed";
export type RunStatus = "success" | "failed" | "null" | "running";

export interface Run {
  id: string;
  shortId: string;
  runName?: string;
  model: string;
  modelId?: string;
  mode: RunMode;
  requestedMode?: string | null;
  batchId?: string | null;
  productMode?: string | null;
  modeLabel?: string | null;
  modeOrder?: number | null;
  task: string;
  taskDir?: string;
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
  thoughtsEnabled?: boolean;
  hasSummary?: boolean;
  currentStage?: string | null;
  failReason?: string;
  finalStatus?: string | null;
  command?: string;
}

export interface ModelRec {
  id: string;
  name: string;
  provider: string;
  baseUrl: string;
  ctx: number;
  maxTokens?: number;
  online: boolean | null;
  hasApiKey?: boolean;
  apiKeyEnv?: string;
  createdAt?: string | null;
}

export interface Task {
  id: string;
  name: string;
  task: string;
  taskType?: "classification" | "regression" | "auto" | "unknown";
  metric: string;
  metricGoal: "min" | "max";
  rows: number;
  cols: number;
  target: string;
  source: string;
  desc: string;
  prepared: boolean;
  status?: TaskStatus;
  taskDir: string;
  seed?: number;
  tags?: string[];
  createdAt?: string | null;
  updatedAt?: string | null;
  hasTrain?: boolean;
  hasVal?: boolean;
  hasTest?: boolean;
  rawFiles?: UploadedFileNode[];
  warnings?: string[];
  warningsCount?: number;
  sources?: TaskSource[];
  splits?: Record<string, TaskSplitFile | null>;
}

export type TaskStatus = "prepared" | "partial" | "unprepared";

export interface TaskSource {
  name?: string;
  url?: string;
  license?: string;
  citation?: string;
  author?: string;
  organization?: string;
  original_download_date?: string;
  upload_date?: string;
  notes?: string;
}

export interface AgentNotes {
  task_description: string;
  data_structure: string;
  column_descriptions: Record<string, string>;
  additional_comments: string;
  leakage_warning: string;
  visible_to_agent: boolean;
}

export interface TaskTypeConfig {
  task_type: "auto" | "classification" | "regression";
  target_col: string;
  metric_name: string;
  metric_goal: "min" | "max";
  positive_label?: string | null;
  class_labels?: string[];
  id_columns?: string[];
  ignore_columns?: string[];
  sample_weight_col?: string | null;
  group_col?: string | null;
  time_col?: string | null;
  max_runtime?: number | null;
  max_steps?: number | null;
  allowed_libraries?: string[];
  constraints?: string;
}

export interface TaskSplitFile {
  path: string;
  source_path?: string | null;
  rows: number;
  cols: number;
}

export interface TaskSplitConfig {
  mode: "raw_split" | "prepared_files";
  train?: TaskSplitFile | null;
  val?: TaskSplitFile | null;
  test?: TaskSplitFile | null;
  raw_path?: string;
  mapping?: Record<string, string>;
  ratios?: { train: number; val: number; test: number } | null;
  seed: number;
  shuffle?: boolean;
  stratify?: "auto" | "on" | "off";
  create_val_from_train?: boolean;
  val_ratio?: number | null;
}

export interface UploadedFileNode {
  id: string;
  path: string;
  name: string;
  size: number;
  format: string;
  kind: "file" | "dir";
  readable: boolean;
  rows?: number | null;
  cols?: number | null;
  status: string;
  warnings: string[];
  children?: UploadedFileNode[];
  original_name?: string;
}

export interface TaskConfig {
  id: string;
  name: string;
  created_at?: string | null;
  updated_at?: string | null;
  version: number;
  status: TaskStatus;
  task: TaskTypeConfig;
  splits: TaskSplitConfig;
  raw_files: UploadedFileNode[];
  agent_notes: AgentNotes;
  sources: TaskSource[];
  tags: string[];
  warnings: string[];
}

export interface TaskCreatePayload {
  id: string;
  name: string;
  uploadId?: string | null;
  task: TaskTypeConfig;
  splits: TaskSplitConfig;
  agentNotes: AgentNotes;
  sources: TaskSource[];
  tags: string[];
  warnings?: string[];
  desc?: string;
}

export interface TaskPreview {
  columns: string[];
  rows: unknown[][];
  total: number | null;
  shown: number;
  dtypes?: Record<string, string>;
  missing?: Record<string, number>;
  warnings?: string[];
}

export interface TaskColumnStats {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical";
  missingPct: number;
  unique: number;
  hist: number[];
  target?: boolean;
  ignored?: boolean;
  idColumn?: boolean;
}

export interface TaskUpload {
  upload_id: string;
  file?: UploadedFileNode;
  files: UploadedFileNode[];
  flat: UploadedFileNode[];
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
  type: string;
  stage?: string | null;
  thoughts?: string;
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
  type?: string;
}
export interface LogsData {
  messages: LogMessage[];
  processLog?: string;
}

export interface RunSummary {
  summary?: string;
  model?: string | null;
  generatedAt?: string | null;
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
  default_episode: string;
  date_format: "mdy" | "dmy";
  language: "ru" | "en";
  theme: string;
  accent: string;
  radius: number;
  animations: "on" | "off";
  overlay_opacity?: number;
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
  mode: LaunchRunMode | "batch";
  modes?: LaunchRunMode[];
  taskId: string;

  maxSteps?: number;
  maxTokens?: number;
  shots?: number;
  temp?: number;
  seed?: number;
  execution?: "server" | "local";
  enableThoughts?: boolean;
  hintCooldown?: number;
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

async function reqForm<T>(path: string, body: FormData): Promise<T> {
  const res = await fetch(BASE + path, { method: "POST", body });
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
    req<{ ok: boolean; repo?: boolean; runtime?: boolean; output?: string; error?: string }>(
      "/settings/remote-check",
      { method: "POST" }
    ),

  listRuns: () => req<Run[]>("/runs"),
  listArchivedRuns: () => req<Run[]>("/runs/archived"),
  archiveRuns: (ids: string[]) => req<{ archived: string[] }>("/runs/archive", { method: "POST", body: JSON.stringify({ ids }) }),
  unarchiveRuns: (ids: string[]) => req<{ unarchived: string[] }>("/runs/unarchive", { method: "POST", body: JSON.stringify({ ids }) }),
  getRun: (id: string) => req<Run>(`/runs/${id}`),
  launchRun: (p: LaunchPayload) =>
    req<Run>("/runs", { method: "POST", body: JSON.stringify(p) }),
  stopRun: (id: string) => req<{ stopped: string }>(`/runs/${id}/stop`, { method: "POST" }),
  notebook: (id: string) => req<{ cells: NotebookCell[] }>(`/runs/${id}/notebook`),
  trajectory: (id: string) => req<TrajectoryStep[]>(`/runs/${id}/trajectory`),
  thoughts: (id: string) =>
    req<{ step: number; type: string; stage?: string; thoughts: string; timestamp?: string }[]>(
      `/runs/${id}/thoughts`
    ),
  runSummary: (id: string) =>
    req<RunSummary>(`/runs/${id}/summary`),
  checklist: (id: string) => req<ChecklistData>(`/runs/${id}/checklist`),
  errors: (id: string) => req<RunError[]>(`/runs/${id}/errors`),
  logs: (id: string) => req<LogsData>(`/runs/${id}/logs`),

  listTasks: () => req<Task[]>("/tasks"),
  listArchivedTasks: () => req<Task[]>("/tasks/archived"),
  archiveTasks: (ids: string[]) => req<{ archived: string[] }>("/tasks/archive", { method: "POST", body: JSON.stringify({ ids }) }),
  unarchiveTasks: (ids: string[]) => req<{ unarchived: string[] }>("/tasks/unarchive", { method: "POST", body: JSON.stringify({ ids }) }),
  getTask: (id: string) => req<Task>(`/tasks/${id}`),
  getTaskConfig: (id: string) => req<TaskConfig>(`/tasks/${id}/config`),
  updateTaskConfig: (id: string, body: Partial<TaskConfig>) =>
    req<TaskConfig>(`/tasks/${id}/config`, { method: "PUT", body: JSON.stringify(body) }),
  uploadTaskFile: (file: File, uploadId?: string | null) => {
    const form = new FormData();
    form.append("file", file);
    if (uploadId) form.append("upload_id", uploadId);
    return reqForm<TaskUpload>("/tasks/uploads", form);
  },
  uploadTaskFromUrl: (url: string, uploadId?: string | null) =>
    req<TaskUpload>("/tasks/uploads/from-url", {
      method: "POST",
      body: JSON.stringify({ url, uploadId }),
    }),
  listUploadedFiles: (uploadId: string) => req<TaskUpload>(`/tasks/uploads/${uploadId}/files`),
  extractUploadedArchive: (uploadId: string, path?: string) =>
    req<TaskUpload>(`/tasks/uploads/${uploadId}/extract`, {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  previewUploadedTable: (uploadId: string, path: string, limit = 50) =>
    req<TaskPreview>(
      `/tasks/uploads/${uploadId}/preview?path=${encodeURIComponent(path)}&limit=${limit}`
    ),
  createTaskFromConfig: (body: TaskCreatePayload) =>
    req<Task>("/tasks/create-from-config", { method: "POST", body: JSON.stringify(body) }),
  prepareTask: (id: string) =>
    req<Task>(`/tasks/${id}/prepare`, { method: "POST", body: JSON.stringify({}) }),
  taskPreview: (id: string, split = "train", limit = 50) =>
    req<TaskPreview>(
      `/tasks/${id}/preview?split=${split}&limit=${limit}`
    ),
  taskColumns: (id: string, split = "train") =>
    req<TaskColumnStats[]>(`/tasks/${id}/columns?split=${split}`),
  updateTask: (id: string, body: Record<string, unknown>) =>
    req<Task>(`/tasks/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteTask: (id: string) =>
    req<{ deleted: string }>(`/tasks/${id}`, { method: "DELETE" }),
  uploadTask: (form: FormData) =>
    reqForm<Task>("/tasks", form),

  listModels: () => req<ModelRec[]>("/models"),
  listArchivedModels: () => req<ModelRec[]>("/models/archived"),
  archiveModels: (ids: string[]) => req<{ archived: string[] }>("/models/archive", { method: "POST", body: JSON.stringify({ ids }) }),
  unarchiveModels: (ids: string[]) => req<{ unarchived: string[] }>("/models/unarchive", { method: "POST", body: JSON.stringify({ ids }) }),
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
