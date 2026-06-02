import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  api,
  type AgentNotes,
  type Dataset,
  type DatasetCreatePayload,
  type DatasetPreview,
  type DatasetSource,
  type DatasetTaskConfig,
  type UploadedFileNode,
} from "../../lib/api";
import { Button, Card, Field, Modal, Spinner, Tag } from "../ui";
import { Icon } from "../Icon";

const STEPS = [
  "Основное",
  "Файлы",
  "Сплиты",
  "Конфиг",
  "Заметки",
  "Источники",
  "Проверка",
];

const METRICS = {
  classification: ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "logloss"],
  regression: ["neg_rmse", "rmse", "mae", "r2"],
  auto: ["f1_macro", "f1_weighted", "neg_rmse", "rmse"],
};

type TaskType = "auto" | "classification" | "regression";
type MetricGoal = "max" | "min";
type SplitMode = "raw_split" | "prepared_files";
type StratifyMode = "auto" | "on" | "off";

const TASK_LABEL: Record<TaskType, string> = {
  auto: "auto",
  classification: "classification",
  regression: "regression",
};
const MODE_LABEL: Record<SplitMode, string> = {
  raw_split: "Raw-режим",
  prepared_files: "Prepared режим",
};
const NAME_RE = /^[a-z0-9][a-z0-9_]*$/;

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]/g, "")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}


function inferGoal(metric: string): MetricGoal {
  const m = metric.toLowerCase();
  if (m.startsWith("neg_")) return "max";
  return ["rmse", "rmsle", "mae", "mse", "logloss"].includes(m) ? "min" : "max";
}

function formatBytes(size: number) {
  if (!size) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let n = size;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function flatten(nodes: UploadedFileNode[]): UploadedFileNode[] {
  return nodes.flatMap((node) => (node.children ? [node, ...flatten(node.children)] : [node]));
}

/** Portal-based tooltip — renders in document.body, always on top */
function Info({ text }: { text: string }) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const dotRef = useRef<HTMLSpanElement>(null);

  function show() {
    const rect = dotRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ top: rect.top, left: rect.left + rect.width / 2 });
    setVisible(true);
  }

  return (
    <>
      <span
        ref={dotRef}
        className="info-dot"
        aria-label={text}
        onMouseEnter={show}
        onMouseLeave={() => setVisible(false)}
      >?</span>
      {visible && createPortal(
        <div className="tooltip-portal" style={{ top: pos.top, left: pos.left }}>
          {text}
        </div>,
        document.body
      )}
    </>
  );
}

function FieldInfo({
  label,
  info,
  hint,
  children,
  required,
}: {
  label: ReactNode;
  info: string;
  hint?: string;
  children: ReactNode;
  required?: boolean;
}) {
  return (
    <Field
      label={<span className="field-info-label">{label}<Info text={info} /></span>}
      hint={hint}
      required={required}
    >
      {children}
    </Field>
  );
}

function FileRows({
  files, selected, onSelect, onPreview, onExtract, onDelete,
}: {
  files: UploadedFileNode[];
  selected?: string | null;
  onSelect: (path: string) => void;
  onPreview: (path: string) => void;
  onExtract: (path: string) => void;
  onDelete: (path: string) => void;
}) {
  const rows = files.filter((f) => f.kind === "file");
  if (!rows.length) return <div className="empty-inline">Файлы ещё не загружены.</div>;
  return (
    <div className="table-wrap">
      <table className="data dataset-file-table">
        <thead>
          <tr>
            <th>Файл</th><th>Формат</th><th>Размер</th><th>Форма</th><th />
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => (
            <tr key={f.path} className={selected === f.path ? "row-selected" : ""}>
              <td>
                <button className="linkish mono" type="button" onClick={() => onSelect(f.path)}>{f.path}</button>
                {f.warnings?.length > 0 && <div className="warn-text">{f.warnings.join(" ")}</div>}
              </td>
              <td><Tag mono>{f.format}</Tag></td>
              <td className="mono faint">{formatBytes(f.size)}</td>
              <td className="mono faint">{f.rows != null && f.cols != null ? `${f.rows} x ${f.cols}` : "-"}</td>
              <td>
                <div className="row" style={{ justifyContent: "flex-end" }}>
                  {f.readable && <Button size="sm" variant="ghost" icon="table" onClick={() => onPreview(f.path)}>Превью</Button>}
                  {f.status === "archive" && <Button size="sm" variant="ghost" icon="layers" onClick={() => onExtract(f.path)}>Распаковать</Button>}
                  <Button size="sm" variant="ghost" icon="trash" onClick={() => onDelete(f.path)} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PreviewBox({ preview, target }: { preview: DatasetPreview | null; target: string }) {
  if (!preview) return null;
  return (
    <Card style={{ padding: 0 }}>
      <div className="table-wrap wizard-preview">
        <table className="data">
          <thead>
            <tr>{preview.columns.map((c) => <th key={c} className={c === target ? "target-col" : undefined}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {preview.rows.slice(0, 8).map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j} className={`mono${preview.columns[j] === target ? " target-col" : ""}`}>
                    {v === null ? <span className="faint">пусто</span> : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="faint" style={{ padding: "10px 14px", fontSize: 12 }}>
        показано {preview.shown} из {preview.total ?? "?"} строк
      </div>
    </Card>
  );
}

function SourceEditor({ sources, onChange }: { sources: DatasetSource[]; onChange: (v: DatasetSource[]) => void }) {
  const update = (idx: number, patch: DatasetSource) => onChange(sources.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  return (
    <div className="stack" style={{ gap: 12 }}>
      {sources.map((source, idx) => (
        <Card key={idx} className="compact-card">
          <div className="spread" style={{ marginBottom: 12 }}>
            <strong>Источник {idx + 1}</strong>
            <Button size="sm" variant="ghost" icon="trash" onClick={() => onChange(sources.filter((_, i) => i !== idx))}>Удалить</Button>
          </div>
          <div className="grid-2">
            <Field label="Название"><input className="input" value={source.name ?? ""} onChange={(e) => update(idx, { name: e.target.value })} /></Field>
            <Field label="URL"><input className="input" value={source.url ?? ""} onChange={(e) => update(idx, { url: e.target.value })} /></Field>
            <Field label="Лицензия"><input className="input" value={source.license ?? ""} onChange={(e) => update(idx, { license: e.target.value })} /></Field>
            <Field label="Цитирование"><input className="input" value={source.citation ?? ""} onChange={(e) => update(idx, { citation: e.target.value })} /></Field>
            <Field label="Автор"><input className="input" value={source.author ?? ""} onChange={(e) => update(idx, { author: e.target.value })} /></Field>
            <Field label="Организация"><input className="input" value={source.organization ?? ""} onChange={(e) => update(idx, { organization: e.target.value })} /></Field>
          </div>
          <Field label="Заметки"><textarea className="input" rows={2} value={source.notes ?? ""} onChange={(e) => update(idx, { notes: e.target.value })} /></Field>
        </Card>
      ))}
      <Button icon="plus" onClick={() => onChange([...sources, { upload_date: new Date().toISOString().slice(0, 10) }])}>
        Добавить источник
      </Button>
    </div>
  );
}

// ── Draft persistence ──────────────────────────────────────────────
const DRAFT_KEY = "autovibe_problem_draft";
function saveDraft(data: object) {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify(data)); } catch {}
}
function loadDraft(): Record<string, unknown> | null {
  try { const s = localStorage.getItem(DRAFT_KEY); return s ? JSON.parse(s) : null; }
  catch { return null; }
}
function clearDraft() {
  try { localStorage.removeItem(DRAFT_KEY); } catch {}
}

export function DatasetWizard({ onClose, onCreated }: { onClose: () => void; onCreated: (dataset: Dataset) => void }) {
  const fileInput = useRef<HTMLInputElement>(null);

  // Restore from draft on first mount
  const d = useMemo(loadDraft, []);

  const [step, setStep] = useState<number>((d?.step as number) ?? 0);

  // Step 0 — Основное
  const [name, setName] = useState<string>((d?.name as string) ?? "");
  const [nameError, setNameError] = useState<string | null>(null);
  const [taskType, setTaskType] = useState<TaskType>((d?.taskType as TaskType) ?? "auto");
  const [target, setTarget] = useState<string>((d?.target as string) ?? "");
  const [metric, setMetric] = useState<string>((d?.metric as string) ?? "f1_macro");
  const [metricGoal, setMetricGoal] = useState<MetricGoal>((d?.metricGoal as MetricGoal) ?? "max");
  const [tags, setTags] = useState<string>((d?.tags as string) ?? "");
  const [desc, setDesc] = useState<string>((d?.desc as string) ?? "");

  // Step 1 — Files (transient server state — not persisted)
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [files, setFiles] = useState<UploadedFileNode[]>([]);
  const [deletedPaths, setDeletedPaths] = useState<Set<string>>(new Set());
  const [uploadUrl, setUploadUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);

  // Step 2 — Splits
  const [splitMode, setSplitMode] = useState<SplitMode>((d?.splitMode as SplitMode) ?? "raw_split");
  const [ratios, setRatios] = useState<{ train: number; val: number; test: number }>(
    (d?.ratios as { train: number; val: number; test: number }) ?? { train: 0.7, val: 0.15, test: 0.15 }
  );
  const [shuffle, setShuffle] = useState<boolean>((d?.shuffle as boolean) ?? true);
  const [stratify, setStratify] = useState<StratifyMode>((d?.stratify as StratifyMode) ?? "auto");
  const [seed, setSeed] = useState<number>((d?.seed as number) ?? 42);
  const [mapping, setMapping] = useState<{ train: string; val: string; test: string }>(
    (d?.mapping as { train: string; val: string; test: string }) ?? { train: "", val: "", test: "" }
  );
  const [createValFromTrain, setCreateValFromTrain] = useState<boolean>((d?.createValFromTrain as boolean) ?? true);
  const [valRatio, setValRatio] = useState<number>((d?.valRatio as number) ?? 0.15);
  const [useTimeSplits, setUseTimeSplits] = useState<boolean>((d?.useTimeSplits as boolean) ?? false);
  const [timeCol, setTimeCol] = useState<string>((d?.timeCol as string) ?? "");

  // Step 3 — Config (advanced)
  const [idColumns, setIdColumns] = useState<string>((d?.idColumns as string) ?? "");
  const [ignoreColumns, setIgnoreColumns] = useState<string>((d?.ignoreColumns as string) ?? "");
  const [allowedLibraries, setAllowedLibraries] = useState<string>((d?.allowedLibraries as string) ?? "");

  // Step 4 — Agent notes
  const [agentNotes, setAgentNotes] = useState<AgentNotes>(
    (d?.agentNotes as AgentNotes) ?? {
      task_description: "",
      data_structure: "",
      column_descriptions: {},
      additional_comments: "",
      leakage_warning: "",
      visible_to_agent: true,
    }
  );

  // Step 5 — Sources
  const [sources, setSources] = useState<DatasetSource[]>((d?.sources as DatasetSource[]) ?? []);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  // Persist draft on every change (files/upload state excluded — server-side transient)
  useEffect(() => {
    saveDraft({ step, name, taskType, target, metric, metricGoal, tags, desc, splitMode, ratios, shuffle, stratify, seed, mapping, createValFromTrain, valRatio, useTimeSplits, timeCol, idColumns, ignoreColumns, allowedLibraries, agentNotes, sources });
  }, [step, name, taskType, target, metric, metricGoal, tags, desc, splitMode, ratios, shuffle, stratify, seed, mapping, createValFromTrain, valRatio, useTimeSplits, timeCol, idColumns, ignoreColumns, allowedLibraries, agentNotes, sources]);

  const flatFiles = useMemo(() => flatten(files).filter((f) => f.kind === "file" && !deletedPaths.has(f.path)), [files, deletedPaths]);
  const readableFiles = flatFiles.filter((f) => f.readable);
  const finalSlug = slugify(name);

  // Auto-select first readable file when none is selected
  useEffect(() => {
    if (!selectedFile && readableFiles.length) setSelectedFile(readableFiles[0].path);
  }, [readableFiles, selectedFile]);
  const warnings = useMemo(() => {
    const out: string[] = [];
    if (splitMode === "prepared_files" && !mapping.val && !createValFromTrain) out.push("Validation split отсутствует.");
    if (splitMode === "prepared_files" && !mapping.test) out.push("Без test-сплита финальная оценка будет невозможна.");
    if (agentNotes.visible_to_agent && (agentNotes.task_description || agentNotes.additional_comments)) {
      out.push("Заметки агента попадут в промпт — не добавляйте утечки.");
    }
    return out;
  }, [agentNotes, createValFromTrain, mapping.test, mapping.val, splitMode]);

  function handleNameChange(raw: string) {
    // Normalize: lowercase, spaces→underscores, strip other invalid chars
    const normalized = raw.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    setName(normalized);
    if (normalized && !NAME_RE.test(normalized)) {
      setNameError("Только строчные буквы, цифры и _ (подчёркивание).");
    } else {
      setNameError(null);
    }
  }

  async function uploadFiles(picked: FileList | null) {
    if (!picked?.length) return;
    setBusy(true); setError(null);
    try {
      let current = uploadId;
      let latestFiles = files;
      const existingNames = new Set(flatten(latestFiles).flatMap((f) => [f.name, f.original_name].filter(Boolean) as string[]));
      const dupes: string[] = [];
      for (const file of Array.from(picked)) {
        if (existingNames.has(file.name)) { dupes.push(file.name); continue; }
        const res = await api.uploadDatasetFile(file, current);
        current = res.upload_id;
        latestFiles = res.files;
        existingNames.add(file.name);
      }
      setUploadId(current);
      setFiles(latestFiles);
      if (dupes.length) setError(`Уже загружено: ${dupes.join(", ")}`);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); if (fileInput.current) fileInput.current.value = ""; }
  }

  async function uploadByUrl() {
    if (!uploadUrl.trim()) return;
    setBusy(true); setError(null);
    try {
      const res = await api.uploadDatasetFromUrl(uploadUrl.trim(), uploadId);
      setUploadId(res.upload_id);
      setFiles(res.files);
      setUploadUrl("");
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function extractArchive(path: string) {
    if (!uploadId) return;
    setBusy(true); setError(null);
    try { const res = await api.extractUploadedArchive(uploadId, path); setFiles(res.files); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function loadPreview(path: string) {
    if (!uploadId) return;
    setBusy(true); setError(null);
    try { setPreviewPath(path); setPreview(await api.previewUploadedTable(uploadId, path, 50)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  function validateStep(idx = step): string | null {
    if (idx === 0) {
      if (!name.trim()) return "Укажите название датасета.";
      if (!NAME_RE.test(name)) return "Название: только строчные буквы, цифры и _ (без пробелов).";
      if (!target.trim()) return "Укажите target-колонку.";
      if (!metric.trim()) return "Укажите метрику.";
    }
    if (idx === 1 && !readableFiles.length) return "Загрузите хотя бы один табличный файл.";
    if (idx === 2) {
      if (splitMode === "raw_split") {
        const sum = ratios.train + ratios.val + ratios.test;
        if (Math.abs(sum - 1) > 0.0001 || ratios.train <= 0 || ratios.val < 0 || ratios.test < 0)
          return "Доли split'ов должны суммироваться в 1.0.";
      } else if (!mapping.train) {
        return "Нужен train-файл.";
      }
    }
    return null;
  }

  /** Can navigate to step idx only if all previous steps pass validation */
  function canNavigateTo(idx: number): boolean {
    for (let i = 0; i < idx; i++) {
      if (validateStep(i) !== null) return false;
    }
    return true;
  }

  function navigateTo(idx: number) {
    if (idx <= step) { setStep(idx); return; } // always allow going back
    if (!canNavigateTo(idx)) {
      for (let i = 0; i < idx; i++) {
        const err = validateStep(i);
        if (err) { setError(`Шаг ${i + 1}: ${err}`); return; }
      }
    }
    setError(null);
    setStep(idx);
  }

  function updateGoal(nextMetric: string) {
    setMetric(nextMetric);
    setMetricGoal(inferGoal(nextMetric));
  }

  function taskConfig(): DatasetTaskConfig {
    return {
      task_type: taskType,
      target_col: target.trim(),
      metric_name: metric.trim(),
      metric_goal: metricGoal,
      positive_label: null,
      class_labels: [],
      id_columns: idColumns.split(",").map((s) => s.trim()).filter(Boolean),
      ignore_columns: ignoreColumns.split(",").map((s) => s.trim()).filter(Boolean),
      sample_weight_col: null,
      group_col: null,
      time_col: useTimeSplits ? timeCol || null : null,
      max_runtime: null,
      max_steps: null,
      allowed_libraries: allowedLibraries.split(",").map((s) => s.trim()).filter(Boolean),
      constraints: "",
    };
  }

  function buildPayload(): DatasetCreatePayload {
    const notes = { ...agentNotes, column_descriptions: {} };
    return {
      id: finalSlug,
      name: name.trim(),
      uploadId,
      task: taskConfig(),
      splits:
        splitMode === "raw_split"
          ? { mode: "raw_split", raw_path: selectedFile ?? "", ratios, seed, shuffle, stratify }
          : {
              mode: "prepared_files",
              mapping: Object.fromEntries(Object.entries(mapping).filter(([, v]) => v)),
              seed, shuffle: true, stratify,
              create_val_from_train: createValFromTrain,
              val_ratio: valRatio,
            },
      agentNotes: notes,
      sources,
      tags: tags.split(",").map((s) => s.trim()).filter(Boolean),
      warnings,
      desc,
    };
  }

  async function createDataset() {
    const stepError = [0, 1, 2].map((idx) => validateStep(idx)).find(Boolean);
    if (stepError) { setError(stepError); return; }
    setBusy(true); setError(null);
    try {
      const created = await api.createDatasetFromConfig(buildPayload());
      clearDraft(); // success → clear saved draft
      onCreated(created);
    }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  function handleCancel() {
    clearDraft(); // explicit cancel → clear draft
    onClose();
  }

  function next() {
    const err = validateStep();
    if (err) { setError(err); return; }
    setError(null);
    setStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  const payloadPreview = (() => {
    try { return JSON.stringify(buildPayload(), null, 2); }
    catch { return "Ошибка в данных формы."; }
  })();

  return (
    <Modal
      title="Новая проблема"
      width={980}
      onClose={onClose}
      footer={
        <>
          <span className="required-legend"><span className="required-star">*</span> обязательные поля</span>
          <Button variant="ghost" onClick={handleCancel}>Отмена</Button>
          <Button variant="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0 || busy}>Назад</Button>
          {step < STEPS.length - 1 ? (
            <Button variant="primary" onClick={next} disabled={busy}>Далее</Button>
          ) : (
            <Button variant="primary" onClick={createDataset} disabled={busy}>
              {busy ? <Spinner /> : "Создать"}
            </Button>
          )}
        </>
      }
    >
      <div className="wizard">
        <div className="wizard-rail">
          {STEPS.map((label, idx) => (
            <button
              key={label}
              className={`wizard-step${idx === step ? " active" : ""}${idx < step ? " done" : ""}${idx > step && !canNavigateTo(idx) ? " locked" : ""}`}
              onClick={() => navigateTo(idx)}
              title={idx > step && !canNavigateTo(idx) ? "Сначала заполните обязательные поля предыдущих шагов" : undefined}
            >
              <span>{idx + 1}</span>
              {label}
            </button>
          ))}
        </div>

        <div className="wizard-panel">
          {/* ─── Шаг 0: Основное ─── */}
          {step === 0 && (
            <div className="stack" style={{ gap: 16 }}>
              <FieldInfo
                label="Название"
                info="Имя датасета в формате snake_case: только строчные латинские буквы, цифры и символ _ (подчёркивание). Пробелы и спецсимволы недопустимы. Пример: dry_bean_quality"
                required
              >
                <input
                  className={`input mono${nameError ? " input-error" : ""}`}
                  value={name}
                  onChange={(e) => handleNameChange(e.target.value)}
                  placeholder="dry_bean_quality"
                />
                {nameError && <div className="field-error">{nameError}</div>}
              </FieldInfo>

              <div className="grid-3">
                <FieldInfo label="Тип задачи" info="classification — предсказание категорий (классов). regression — предсказание числа. auto — тип определится автоматически по данным.">
                  <select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                    <option value="auto">auto</option>
                    <option value="classification">classification</option>
                    <option value="regression">regression</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Target column" info="Название колонки с целевой переменной — то, что агент должен научиться предсказывать. Эта колонка никогда не включается в признаки." required>
                  <input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" />
                </FieldInfo>
                <FieldInfo
                  label={<>Метрика <a className="docs-link" href="https://scikit-learn.org/stable/modules/model_evaluation.html" target="_blank" rel="noopener noreferrer">все метрики sklearn ↗</a></>}
                  info="Название метрики из sklearn.metrics. Считается на test после submit; агент видит только val. Примеры: f1_macro, accuracy, neg_rmse, roc_auc"
                  required
                >
                  <input
                    className="input mono"
                    value={metric}
                    onChange={(e) => updateGoal(e.target.value)}
                    placeholder="f1_macro"
                    list="metric-suggestions"
                  />
                  <datalist id="metric-suggestions">
                    {(METRICS[taskType as keyof typeof METRICS] ?? METRICS.auto).map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                </FieldInfo>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "var(--gap, 16px)" }}>
                <FieldInfo label="Metric goal" info="Направление оптимизации: maximize — чем больше, тем лучше (accuracy, f1); minimize — чем меньше, тем лучше (rmse, mae). Выводится автоматически из имени метрики.">
                  <select className="input" value={metricGoal} onChange={(e) => setMetricGoal(e.target.value as MetricGoal)}>
                    <option value="max">maximize</option>
                    <option value="min">minimize</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Теги" info="Ключевые слова через запятую — используются для поиска и фильтрации на странице Проблем. Пример: tabular, benchmark, uci">
                  <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} placeholder="tabular, benchmark" />
                </FieldInfo>
              </div>

              <FieldInfo label="Описание" info="Короткое описание для карточки проблемы. Не влияет ни на что — только отображается в интерфейсе.">
                <textarea className="input" rows={2} value={desc} onChange={(e) => setDesc(e.target.value)} />
              </FieldInfo>

              <div className="path-preview">Папка: <span className="mono">datasets/{finalSlug || "dataset_id"}</span></div>
            </div>
          )}

          {/* ─── Шаг 1: Raw files ─── */}
          {step === 1 && (
            <div className="stack" style={{ gap: 16 }}>
              <Card className="compact-card">
                <input ref={fileInput} type="file" multiple hidden onChange={(e) => uploadFiles(e.target.files)} />
                <div
                  className={`upload-dropzone${dragging ? " dragging" : ""}${busy ? " uploading" : ""}`}
                  onClick={() => !busy && fileInput.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFiles(e.dataTransfer.files); }}
                >
                  <Icon name="upload" size={28} strokeWidth={1.6} />
                  <div className="upload-dropzone-text">Загрузите или перетащите файлы</div>
                  <div className="upload-dropzone-hint">CSV, TSV, XLSX, Parquet, JSON, ZIP, TAR, GZ</div>
                </div>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <FieldInfo label="Загрузить по URL" info="Прямая ссылка на файл (CSV, ZIP и т.д.). Файл скачается на сервер в staging-зону загрузки.">
                    <input className="input" value={uploadUrl} onChange={(e) => setUploadUrl(e.target.value)} placeholder="https://example.com/data.csv" />
                  </FieldInfo>
                  <div style={{ display: "flex", alignItems: "end" }}>
                    <Button icon="external" onClick={uploadByUrl} disabled={busy || !uploadUrl.trim()}>Скачать</Button>
                  </div>
                </div>
              </Card>
              <FileRows
                files={flatten(files).filter((f) => !deletedPaths.has(f.path))}
                selected={selectedFile}
                onSelect={setSelectedFile}
                onPreview={loadPreview}
                onExtract={extractArchive}
                onDelete={(path) => {
                  setDeletedPaths((prev) => new Set([...prev, path]));
                  if (selectedFile === path) setSelectedFile(null);
                  if (previewPath === path) { setPreviewPath(null); setPreview(null); }
                }}
              />
              {previewPath && <div className="faint">Превью: <span className="mono">{previewPath}</span></div>}
              <PreviewBox preview={preview} target={target} />
            </div>
          )}

          {/* ─── Шаг 2: Сплиты ─── */}
          {step === 2 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="segmented-with-info">
                <div className="segmented">
                  <button className={splitMode === "raw_split" ? "active" : ""} onClick={() => setSplitMode("raw_split")}>Raw-режим</button>
                  <button className={splitMode === "prepared_files" ? "active" : ""} onClick={() => setSplitMode("prepared_files")}>Prepared режим</button>
                </div>
                <Info text="Raw-режим: один файл, автоматически разбивается на train/val/test. Prepared режим: у вас уже есть готовые train/val/test файлы — просто сопоставьте их." />
              </div>

              {splitMode === "raw_split" ? (
                <>
                  <FieldInfo label="Raw-таблица" info="Исходный файл целиком. Будет автоматически перемешан и разбит на train/val/test согласно заданным долям." required>
                    <select className="input mono" value={selectedFile ?? ""} onChange={(e) => setSelectedFile(e.target.value)}>
                      <option value="">Выберите таблицу</option>
                      {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                    </select>
                  </FieldInfo>
                  <div className="grid-3">
                    <FieldInfo label="Доля train" info="Доля строк для обучения агента. Рекомендуется 0.7–0.8.">
                      <input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.train} onChange={(e) => setRatios((r) => ({ ...r, train: Number(e.target.value) }))} />
                    </FieldInfo>
                    <FieldInfo label="Доля val" info="Доля строк для валидации — агент видит свой score на этой части после каждого submit-шага.">
                      <input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.val} onChange={(e) => setRatios((r) => ({ ...r, val: Number(e.target.value) }))} />
                    </FieldInfo>
                    <FieldInfo label="Доля test" info="Скрытая тест-выборка. Агент не имеет к ней доступа — используется только для финальной оценки после submit.">
                      <input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.test} onChange={(e) => setRatios((r) => ({ ...r, test: Number(e.target.value) }))} />
                    </FieldInfo>
                  </div>
                </>
              ) : (
                <>
                  <div className="grid-3">
                    {(["train", "val", "test"] as const).map((split) => (
                      <div key={split}>
                        {split === "val" && !mapping.val && createValFromTrain ? (
                          <Field label={<span className="field-info-label">val-файл <Info text="val-файл будет создан автоматически из части train. Отключите чекбокс ниже, чтобы указать файл вручную." /></span>}>
                            <input className="input mono" disabled value="будет создан из train" style={{ color: "var(--text-dim)", fontStyle: "italic" }} readOnly />
                          </Field>
                        ) : (
                          <FieldInfo
                            label={`${split}-файл`}
                            info={split === "train" ? "Файл с обучающими данными." : split === "val" ? "Файл для валидации — агент видит score на этом наборе." : "Скрытый тест-файл. Не виден агенту до submit."}
                            required={split === "train"}
                          >
                            <select className="input mono" value={mapping[split]} onChange={(e) => setMapping((m) => ({ ...m, [split]: e.target.value }))}>
                              <option value="">Не выбран</option>
                              {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                            </select>
                          </FieldInfo>
                        )}
                      </div>
                    ))}
                  </div>
                  {!mapping.val && (
                    <label className="check-row">
                      <input type="checkbox" checked={createValFromTrain} onChange={(e) => setCreateValFromTrain(e.target.checked)} />
                      Создать validation split из train
                    </label>
                  )}
                  {!mapping.val && createValFromTrain && (
                    <FieldInfo label="Доля val из train" info="Какую часть train отрезать для validation. Обычно 0.15–0.2.">
                      <input className="input mono" type="number" min={0.01} max={0.9} step={0.01} value={valRatio} onChange={(e) => setValRatio(Number(e.target.value))} />
                    </FieldInfo>
                  )}
                  {!mapping.test && <div className="warn-box">Без test-сплита финальная оценка будет невозможна.</div>}
                </>
              )}

              <div className="grid-3">
                <FieldInfo label="Shuffle" info="Перемешать строки перед разбиением. Рекомендуется для большинства задач, кроме временных рядов.">
                  <select className="input" value={shuffle ? "true" : "false"} onChange={(e) => setShuffle(e.target.value === "true")}>
                    <option value="true">да</option>
                    <option value="false">нет</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Stratify" info="Стратифицированный сплит: каждая часть содержит одинаковое распределение классов target. Auto — включается автоматически для классификации.">
                  <select className="input" value={stratify} onChange={(e) => setStratify(e.target.value as StratifyMode)}>
                    <option value="auto">auto</option>
                    <option value="on">on</option>
                    <option value="off">off</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Seed" info="Фиксирует случайность — одинаковый seed всегда даёт воспроизводимые train/val/test сплиты.">
                  <input className="input mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value || 42))} />
                </FieldInfo>
              </div>

              <label className="check-row">
                <input type="checkbox" checked={useTimeSplits} onChange={(e) => setUseTimeSplits(e.target.checked)} />
                Временные сплиты (time-based split)
              </label>
              {useTimeSplits && (
                <FieldInfo label="Временная колонка" info="Название колонки с датой/временем. При включённых временных сплитах данные сортируются по этой колонке: обучение — на прошлом, тест — на будущем.">
                  <input className="input mono" value={timeCol} onChange={(e) => setTimeCol(e.target.value)} placeholder="date" />
                </FieldInfo>
              )}
            </div>
          )}

          {/* ─── Шаг 3: Конфиг ─── */}
          {step === 3 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <FieldInfo label="Тип задачи" info="Тип ML-задачи. Влияет на стратификацию и доступные метрики.">
                  <select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                    <option value="auto">auto</option>
                    <option value="classification">classification</option>
                    <option value="regression">regression</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Target" info="Колонка с целевой переменной. Совпадает с шагом Основное.">
                  <input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} />
                </FieldInfo>
                <FieldInfo label="Метрика" info="Функция оценки. Можно ввести вручную любую метрику из sklearn.metrics.">
                  <input className="input mono" value={metric} onChange={(e) => updateGoal(e.target.value)} />
                </FieldInfo>
              </div>
              <details className="disclosure">
                <summary>Расширенный конфиг</summary>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <FieldInfo label="ID-колонки" info="Колонки-идентификаторы (id, row_id) — автоматически исключаются из признаков. Через запятую.">
                    <input className="input mono" value={idColumns} onChange={(e) => setIdColumns(e.target.value)} placeholder="id, row_id" />
                  </FieldInfo>
                  <FieldInfo label="Игнорируемые колонки" info="Колонки, которые точно не должны стать признаками X — например, служебные поля или дубли target.">
                    <input className="input mono" value={ignoreColumns} onChange={(e) => setIgnoreColumns(e.target.value)} />
                  </FieldInfo>
                  <FieldInfo label="Разрешённые библиотеки" info="Ограничение набора инструментов агента. Пусто = разрешено всё. Через запятую." hint="sklearn, xgboost">
                    <input className="input mono" value={allowedLibraries} onChange={(e) => setAllowedLibraries(e.target.value)} placeholder="sklearn, xgboost" />
                  </FieldInfo>
                </div>
              </details>
            </div>
          )}

          {/* ─── Шаг 4: Заметки ─── */}
          {step === 4 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="warn-box">Эти поля могут быть видны LLM-агенту. Не добавляйте тестовые метки или скрытые ответы.</div>
              <label className="check-row">
                <input type="checkbox" checked={agentNotes.visible_to_agent} onChange={(e) => setAgentNotes((n) => ({ ...n, visible_to_agent: e.target.checked }))} />
                Передавать заметки агенту
              </label>
              <FieldInfo label="Описание задачи" info="Что нужно предсказать и зачем. Если включено 'Передавать агенту' — попадает в промпт как контекст задачи.">
                <textarea className="input" rows={3} value={agentNotes.task_description} onChange={(e) => setAgentNotes((n) => ({ ...n, task_description: e.target.value }))} />
              </FieldInfo>
              <FieldInfo label="Структура данных" info="Описание колонок: типы, форматы, особенности (пропуски, выбросы). Помогает агенту быстрее разобраться с данными.">
                <textarea className="input" rows={3} value={agentNotes.data_structure} onChange={(e) => setAgentNotes((n) => ({ ...n, data_structure: e.target.value }))} />
              </FieldInfo>
              <FieldInfo label="Дополнительные комментарии" info="Любые пояснения для агента: известные проблемы в данных, советы по feature engineering, особенности задачи.">
                <textarea className="input" rows={3} value={agentNotes.additional_comments} onChange={(e) => setAgentNotes((n) => ({ ...n, additional_comments: e.target.value }))} />
              </FieldInfo>
              <FieldInfo label="Предупреждения" info="Потенциальные источники data leakage или других проблем, о которых должен знать агент. Не указывайте тестовые ответы.">
                <textarea className="input" rows={2} value={agentNotes.leakage_warning} onChange={(e) => setAgentNotes((n) => ({ ...n, leakage_warning: e.target.value }))} />
              </FieldInfo>
            </div>
          )}

          {/* ─── Шаг 5: Источники ─── */}
          {step === 5 && <SourceEditor sources={sources} onChange={setSources} />}

          {/* ─── Шаг 6: Проверка ─── */}
          {step === 6 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <Card className="compact-card">
                  <div className="metric-label">Датасет</div>
                  <div className="ds-title">{name || "-"}</div>
                  <div className="mono faint">{finalSlug || "-"}</div>
                </Card>
                <Card className="compact-card">
                  <div className="metric-label">Задача</div>
                  <div className="ds-title">{TASK_LABEL[taskType]}</div>
                  <div className="mono faint">{target || "-"} / {metric || "-"}</div>
                </Card>
                <Card className="compact-card">
                  <div className="metric-label">Режим сплита</div>
                  <div className="ds-title">{MODE_LABEL[splitMode]}</div>
                  <div className="mono faint">{splitMode === "raw_split" ? selectedFile || "-" : mapping.train || "-"}</div>
                </Card>
              </div>
              {warnings.length > 0 && <div className="warn-box">{warnings.join(" ")}</div>}
              <div className="grid-2">
                <Card className="compact-card">
                  <h4>dataset_config.json</h4>
                  <pre className="code" style={{ maxHeight: 360 }}>{payloadPreview}</pre>
                </Card>
                <Card className="compact-card">
                  <h4>prepared/meta.json</h4>
                  <pre className="code">{JSON.stringify({ name, target_col: target, metric_name: metric, task_type: taskType, seed, notes: { description: agentNotes.task_description || desc } }, null, 2)}</pre>
                </Card>
              </div>
            </div>
          )}

          {error && <div className="error-line"><Icon name="alert" size={15} /> {error}</div>}
          {busy && <div className="spinner-row"><Spinner /> Загрузка...</div>}
        </div>
      </div>
    </Modal>
  );
}
