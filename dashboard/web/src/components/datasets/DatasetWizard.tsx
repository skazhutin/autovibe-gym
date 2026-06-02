import { useMemo, useRef, useState, type ReactNode } from "react";
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
  "Raw files",
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
  auto: "авто",
  classification: "классификация",
  regression: "регрессия",
};
const MODE_LABEL: Record<SplitMode, string> = {
  raw_split: "raw split",
  prepared_files: "готовые файлы",
};
const FILE_STATUS_LABEL: Record<string, string> = {
  archive: "архив",
  error: "ошибка",
  readable: "читается",
  unsupported: "не поддерживается",
};

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9_-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "");
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
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function flatten(nodes: UploadedFileNode[]): UploadedFileNode[] {
  return nodes.flatMap((node) => (node.children ? [node, ...flatten(node.children)] : [node]));
}

function Info({ text }: { text: string }) {
  return (
    <span className="info-dot" title={text} aria-label={text}>
      ?
    </span>
  );
}

function FieldInfo({
  label,
  info,
  hint,
  children,
}: {
  label: string;
  info: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <Field
      label={<span className="field-info-label">{label}<Info text={info} /></span>}
      hint={hint}
    >
      {children}
    </Field>
  );
}

function FileRows({
  files,
  selected,
  onSelect,
  onPreview,
  onExtract,
}: {
  files: UploadedFileNode[];
  selected?: string | null;
  onSelect: (path: string) => void;
  onPreview: (path: string) => void;
  onExtract: (path: string) => void;
}) {
  const rows = files.filter((f) => f.kind === "file");
  if (!rows.length) {
    return <div className="empty-inline">Файлы еще не загружены.</div>;
  }
  return (
    <div className="table-wrap">
      <table className="data dataset-file-table">
        <thead>
          <tr>
            <th>Файл</th>
            <th>Формат</th>
            <th>Размер</th>
            <th>Форма</th>
            <th>Статус</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => (
            <tr key={f.path} className={selected === f.path ? "row-selected" : ""}>
              <td>
                <button className="linkish mono" type="button" onClick={() => onSelect(f.path)}>
                  {f.path}
                </button>
                {f.warnings?.length > 0 && <div className="warn-text">{f.warnings.join(" ")}</div>}
              </td>
              <td>
                <Tag mono>{f.format}</Tag>
              </td>
              <td className="mono faint">{formatBytes(f.size)}</td>
              <td className="mono faint">{f.rows != null && f.cols != null ? `${f.rows} x ${f.cols}` : "-"}</td>
              <td>
                <Tag tone={f.readable ? "green" : f.status === "archive" ? "blue" : f.status === "error" ? "red" : "neutral"}>
                  {FILE_STATUS_LABEL[f.status] ?? f.status}
                </Tag>
              </td>
              <td>
                <div className="row" style={{ justifyContent: "flex-end" }}>
                  {f.readable && (
                    <Button size="sm" variant="ghost" icon="table" onClick={() => onPreview(f.path)}>
                      Превью
                    </Button>
                  )}
                  {f.status === "archive" && (
                    <Button size="sm" variant="ghost" icon="layers" onClick={() => onExtract(f.path)}>
                      Распаковать
                    </Button>
                  )}
                  {f.readable && (
                    <Button size="sm" icon="check" onClick={() => onSelect(f.path)}>
                      Выбрать
                    </Button>
                  )}
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
            <tr>
              {preview.columns.map((c) => (
                <th key={c} className={c === target ? "target-col" : undefined}>
                  {c}
                </th>
              ))}
            </tr>
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
        показано {preview.shown} из {preview.total ?? "неизвестно"} строк
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
            <Button size="sm" variant="ghost" icon="trash" onClick={() => onChange(sources.filter((_, i) => i !== idx))}>
              Удалить
            </Button>
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

export function DatasetWizard({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (dataset: Dataset) => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugTouched, setSlugTouched] = useState(false);
  const [taskType, setTaskType] = useState<TaskType>("auto");
  const [target, setTarget] = useState("");
  const [metric, setMetric] = useState("f1_macro");
  const [metricGoal, setMetricGoal] = useState<MetricGoal>("max");
  const [seed, setSeed] = useState(42);
  const [tags, setTags] = useState("");
  const [desc, setDesc] = useState("");

  const [uploadId, setUploadId] = useState<string | null>(null);
  const [files, setFiles] = useState<UploadedFileNode[]>([]);
  const [uploadUrl, setUploadUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);

  const [splitMode, setSplitMode] = useState<SplitMode>("raw_split");
  const [ratios, setRatios] = useState({ train: 0.7, val: 0.15, test: 0.15 });
  const [shuffle, setShuffle] = useState(true);
  const [stratify, setStratify] = useState<StratifyMode>("auto");
  const [mapping, setMapping] = useState({ train: "", val: "", test: "" });
  const [createValFromTrain, setCreateValFromTrain] = useState(true);
  const [valRatio, setValRatio] = useState(0.15);

  const [idColumns, setIdColumns] = useState("");
  const [ignoreColumns, setIgnoreColumns] = useState("");
  const [sampleWeightCol, setSampleWeightCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [timeCol, setTimeCol] = useState("");
  const [positiveLabel, setPositiveLabel] = useState("");
  const [classLabels, setClassLabels] = useState("");
  const [maxRuntime, setMaxRuntime] = useState("");
  const [maxSteps, setMaxSteps] = useState("");
  const [allowedLibraries, setAllowedLibraries] = useState("");
  const [constraints, setConstraints] = useState("");

  const [agentNotes, setAgentNotes] = useState<AgentNotes>({
    task_description: "",
    data_structure: "",
    column_descriptions: {},
    additional_comments: "",
    leakage_warning: "",
    visible_to_agent: true,
  });
  const [columnDescriptions, setColumnDescriptions] = useState("{}");
  const [sources, setSources] = useState<DatasetSource[]>([]);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const flatFiles = useMemo(() => flatten(files).filter((f) => f.kind === "file"), [files]);
  const readableFiles = flatFiles.filter((f) => f.readable);
  const finalSlug = slug || slugify(name);
  const metricOptions = METRICS[taskType];
  const warnings = useMemo(() => {
    const out: string[] = [];
    if (splitMode === "prepared_files" && !mapping.val && !createValFromTrain) out.push("Validation split отсутствует.");
    if (splitMode === "prepared_files" && !mapping.test) out.push("Без test-сплита финальная оценка будет невозможна.");
    if (agentNotes.visible_to_agent && (agentNotes.task_description || agentNotes.additional_comments)) {
      out.push("Заметки агента могут попасть в промпт. Не добавляйте туда утечки.");
    }
    return out;
  }, [agentNotes, createValFromTrain, mapping.test, mapping.val, splitMode]);

  function setNameAndMaybeSlug(value: string) {
    setName(value);
    if (!slugTouched) setSlug(slugify(value));
  }

  async function uploadFiles(picked: FileList | null) {
    if (!picked?.length) return;
    setBusy(true);
    setError(null);
    try {
      let current = uploadId;
      let latestFiles = files;
      for (const file of Array.from(picked)) {
        const res = await api.uploadDatasetFile(file, current);
        current = res.upload_id;
        latestFiles = res.files;
      }
      setUploadId(current);
      setFiles(latestFiles);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function uploadByUrl() {
    if (!uploadUrl.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.uploadDatasetFromUrl(uploadUrl.trim(), uploadId);
      setUploadId(res.upload_id);
      setFiles(res.files);
      setUploadUrl("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function extractArchive(path: string) {
    if (!uploadId) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.extractUploadedArchive(uploadId, path);
      setFiles(res.files);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadPreview(path: string) {
    if (!uploadId) return;
    setBusy(true);
    setError(null);
    try {
      setPreviewPath(path);
      setPreview(await api.previewUploadedTable(uploadId, path, 50));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function validateStep(idx = step): string | null {
    if (idx === 0) {
      if (!name.trim()) return "Укажите имя датасета.";
      if (!finalSlug) return "Укажите id датасета.";
      if (!target.trim()) return "Укажите target-колонку.";
      if (!metric.trim()) return "Укажите метрику.";
    }
    if (idx === 1 && !readableFiles.length) return "Загрузите или распакуйте хотя бы один читаемый табличный файл.";
    if (idx === 2) {
      if (splitMode === "raw_split") {
        const sum = ratios.train + ratios.val + ratios.test;
        if (!selectedFile) return "Выберите raw-таблицу для split.";
        if (Math.abs(sum - 1) > 0.0001 || ratios.train <= 0 || ratios.val < 0 || ratios.test < 0) {
          return "Доли split'ов должны суммироваться в 1.0.";
        }
      } else if (!mapping.train) {
        return "Для режима готовых файлов нужен train-файл.";
      }
    }
    return null;
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
      positive_label: positiveLabel || null,
      class_labels: classLabels.split(",").map((s) => s.trim()).filter(Boolean),
      id_columns: idColumns.split(",").map((s) => s.trim()).filter(Boolean),
      ignore_columns: ignoreColumns.split(",").map((s) => s.trim()).filter(Boolean),
      sample_weight_col: sampleWeightCol || null,
      group_col: groupCol || null,
      time_col: timeCol || null,
      max_runtime: maxRuntime ? Number(maxRuntime) : null,
      max_steps: maxSteps ? Number(maxSteps) : null,
      allowed_libraries: allowedLibraries.split(",").map((s) => s.trim()).filter(Boolean),
      constraints,
    };
  }

  function buildPayload(): DatasetCreatePayload {
    let parsedColumns: Record<string, string> = {};
    try {
      parsedColumns = JSON.parse(columnDescriptions || "{}");
    } catch {
      throw new Error("Описания колонок должны быть валидным JSON.");
    }
    const notes = { ...agentNotes, column_descriptions: parsedColumns };
    return {
      id: finalSlug,
      name: name.trim(),
      uploadId,
      task: taskConfig(),
      splits:
        splitMode === "raw_split"
          ? {
              mode: "raw_split",
              raw_path: selectedFile ?? "",
              ratios,
              seed,
              shuffle,
              stratify,
            }
          : {
              mode: "prepared_files",
              mapping: Object.fromEntries(Object.entries(mapping).filter(([, v]) => v)),
              seed,
              shuffle: true,
              stratify,
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
    if (stepError) {
      setError(stepError);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await api.createDatasetFromConfig(buildPayload());
      onCreated(created);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function next() {
    const err = validateStep();
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  const payloadPreview = (() => {
    try {
      return JSON.stringify(buildPayload(), null, 2);
    } catch {
      return "JSON описаний колонок невалиден.";
    }
  })();

  return (
    <Modal
      title="Добавить датасет"
      width={980}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Отмена</Button>
          <Button variant="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0 || busy}>
            Назад
          </Button>
          {step < STEPS.length - 1 ? (
            <Button variant="primary" onClick={next} disabled={busy}>Далее</Button>
          ) : (
            <Button variant="primary" onClick={createDataset} disabled={busy}>
              {busy ? <Spinner /> : "Создать датасет"}
            </Button>
          )}
        </>
      }
    >
      <div className="wizard">
        <div className="wizard-rail">
          {STEPS.map((label, idx) => (
            <button key={label} className={`wizard-step${idx === step ? " active" : ""}${idx < step ? " done" : ""}`} onClick={() => setStep(idx)}>
              <span>{idx + 1}</span>
              {label}
            </button>
          ))}
        </div>
        <div className="wizard-panel">
          {step === 0 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-2">
                <FieldInfo label="Имя датасета" info="Человекочитаемое имя датасета. Пример: Dry Bean Quality.">
                  <input className="input" value={name} onChange={(e) => setNameAndMaybeSlug(e.target.value)} />
                </FieldInfo>
                <FieldInfo label="Dataset ID" info="Имя папки внутри datasets/. Используйте строчные латинские буквы, цифры, дефисы или подчеркивания.">
                  <input className="input mono" value={slug} onChange={(e) => { setSlugTouched(true); setSlug(slugify(e.target.value)); }} />
                </FieldInfo>
              </div>
              <div className="grid-3">
                <FieldInfo label="Тип задачи" info="Авто позволяет бэкенду вывести тип. Классификация/регрессия помогают стратификации и отображению.">
                  <select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                    <option value="auto">авто</option>
                    <option value="classification">классификация</option>
                    <option value="regression">регрессия</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Target column" info="Колонка с меткой, которую нужно предсказывать. Она должна быть в train и validation данных.">
                  <input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" />
                </FieldInfo>
                <FieldInfo label="Метрика" info="Основная метрика оценщика, которая отображается в прогонах.">
                  <select className="input mono" value={metric} onChange={(e) => updateGoal(e.target.value)}>
                    {metricOptions.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </FieldInfo>
              </div>
              <div className="grid-3">
                <FieldInfo label="Цель метрики" info="Что лучше: больше или меньше. Автоматически выводится из имени метрики, но можно изменить.">
                  <select className="input" value={metricGoal} onChange={(e) => setMetricGoal(e.target.value as MetricGoal)}>
                    <option value="max">больше лучше</option>
                    <option value="min">меньше лучше</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Seed" info="Random seed для split'ов и записи в метаданные.">
                  <input className="input mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value || 42))} />
                </FieldInfo>
              </div>
              <FieldInfo label="Теги" info="Теги через запятую для фильтрации в Центре датасетов." hint="пример: табличные, бенчмарк">
                <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} />
              </FieldInfo>
              <FieldInfo label="Короткое описание" info="Короткое описание без утечек для Центра датасетов.">
                <textarea className="input" rows={3} value={desc} onChange={(e) => setDesc(e.target.value)} />
              </FieldInfo>
              <div className="path-preview">Итоговый путь: <span className="mono">datasets/{finalSlug || "dataset-id"}</span></div>
            </div>
          )}

          {step === 1 && (
            <div className="stack" style={{ gap: 16 }}>
              <Card className="compact-card">
                <div className="spread">
                  <div>
                    <strong>Raw upload</strong>
                    <div className="faint" style={{ fontSize: 12 }}>CSV, TSV, JSON, JSONL, XLSX, Parquet, Feather, ZIP, TAR, TGZ и GZ с одним файлом.</div>
                  </div>
                  <div className="row">
                    <input ref={fileInput} type="file" multiple hidden onChange={(e) => uploadFiles(e.target.files)} />
                    <Button icon="upload" onClick={() => fileInput.current?.click()} disabled={busy}>Загрузить файлы</Button>
                  </div>
                </div>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <Field label="Добавить по URL">
                    <input className="input" value={uploadUrl} onChange={(e) => setUploadUrl(e.target.value)} placeholder="https://example.com/data.csv" />
                  </Field>
                  <div style={{ display: "flex", alignItems: "end" }}>
                    <Button icon="external" onClick={uploadByUrl} disabled={busy || !uploadUrl.trim()}>Скачать</Button>
                  </div>
                </div>
              </Card>
              <FileRows files={files} selected={selectedFile} onSelect={setSelectedFile} onPreview={loadPreview} onExtract={extractArchive} />
              {previewPath && <div className="faint">Превью файла <span className="mono">{previewPath}</span></div>}
              <PreviewBox preview={preview} target={target} />
            </div>
          )}

          {step === 2 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="segmented">
                <button className={splitMode === "raw_split" ? "active" : ""} onClick={() => setSplitMode("raw_split")}>Разбить raw-таблицу</button>
                <button className={splitMode === "prepared_files" ? "active" : ""} onClick={() => setSplitMode("prepared_files")}>Сопоставить готовые файлы</button>
              </div>
              {splitMode === "raw_split" ? (
                <>
                  <Field label="Raw-таблица">
                    <select className="input mono" value={selectedFile ?? ""} onChange={(e) => setSelectedFile(e.target.value)}>
                      <option value="">Выберите таблицу</option>
                      {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                    </select>
                  </Field>
                  <div className="grid-3">
                    <Field label="Доля train"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.train} onChange={(e) => setRatios((r) => ({ ...r, train: Number(e.target.value) }))} /></Field>
                    <Field label="Доля val"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.val} onChange={(e) => setRatios((r) => ({ ...r, val: Number(e.target.value) }))} /></Field>
                    <Field label="Доля test"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.test} onChange={(e) => setRatios((r) => ({ ...r, test: Number(e.target.value) }))} /></Field>
                  </div>
                </>
              ) : (
                <>
                  <div className="grid-3">
                    {(["train", "val", "test"] as const).map((split) => (
                      <Field key={split} label={`${split}-файл${split === "train" ? " *" : ""}`}>
                        <select className="input mono" value={mapping[split]} onChange={(e) => setMapping((m) => ({ ...m, [split]: e.target.value }))}>
                          <option value="">Не выбран</option>
                          {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                        </select>
                      </Field>
                    ))}
                  </div>
                  {!mapping.val && (
                    <label className="check-row">
                      <input type="checkbox" checked={createValFromTrain} onChange={(e) => setCreateValFromTrain(e.target.checked)} />
                      Создать validation split из train
                    </label>
                  )}
                  {!mapping.val && createValFromTrain && (
                    <Field label="Доля validation из train">
                      <input className="input mono" type="number" min={0.01} max={0.9} step={0.01} value={valRatio} onChange={(e) => setValRatio(Number(e.target.value))} />
                    </Field>
                  )}
                  {!mapping.test && <div className="warn-box">Без test-сплита финальная оценка будет невозможна.</div>}
                </>
              )}
              <div className="grid-3">
                <Field label="Перемешивать">
                  <select className="input" value={shuffle ? "true" : "false"} onChange={(e) => setShuffle(e.target.value === "true")}>
                    <option value="true">да</option>
                    <option value="false">нет</option>
                  </select>
                </Field>
                <Field label="Stratify">
                  <select className="input" value={stratify} onChange={(e) => setStratify(e.target.value as StratifyMode)}>
                    <option value="auto">авто</option>
                    <option value="on">вкл</option>
                    <option value="off">выкл</option>
                  </select>
                </Field>
                <Field label="Seed для split"><input className="input mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value || 42))} /></Field>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <Field label="Тип задачи"><select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}><option value="auto">авто</option><option value="classification">классификация</option><option value="regression">регрессия</option></select></Field>
                <Field label="Target"><input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} /></Field>
                <Field label="Метрика"><input className="input mono" value={metric} onChange={(e) => updateGoal(e.target.value)} /></Field>
              </div>
              <details className="disclosure">
                  <summary>Расширенный конфиг</summary>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <Field label="ID-колонки"><input className="input mono" value={idColumns} onChange={(e) => setIdColumns(e.target.value)} placeholder="id, row_id" /></Field>
                  <Field label="Игнорируемые колонки"><input className="input mono" value={ignoreColumns} onChange={(e) => setIgnoreColumns(e.target.value)} /></Field>
                  <Field label="Колонка весов"><input className="input mono" value={sampleWeightCol} onChange={(e) => setSampleWeightCol(e.target.value)} /></Field>
                  <Field label="Group column"><input className="input mono" value={groupCol} onChange={(e) => setGroupCol(e.target.value)} /></Field>
                  <Field label="Временная колонка"><input className="input mono" value={timeCol} onChange={(e) => setTimeCol(e.target.value)} /></Field>
                  <Field label="Положительный класс"><input className="input mono" value={positiveLabel} onChange={(e) => setPositiveLabel(e.target.value)} /></Field>
                  <Field label="Классы"><input className="input mono" value={classLabels} onChange={(e) => setClassLabels(e.target.value)} /></Field>
                  <Field label="Разрешенные библиотеки"><input className="input mono" value={allowedLibraries} onChange={(e) => setAllowedLibraries(e.target.value)} placeholder="sklearn, xgboost" /></Field>
                  <Field label="Макс. время"><input className="input mono" type="number" value={maxRuntime} onChange={(e) => setMaxRuntime(e.target.value)} /></Field>
                  <Field label="Макс. шагов"><input className="input mono" type="number" value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} /></Field>
                </div>
                <Field label="Ограничения"><textarea className="input" rows={3} value={constraints} onChange={(e) => setConstraints(e.target.value)} /></Field>
              </details>
            </div>
          )}

          {step === 4 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="warn-box">Эти поля могут быть видны LLM-агенту. Не добавляйте тестовые метки, скрытые ответы или утечки.</div>
              <label className="check-row">
                <input type="checkbox" checked={agentNotes.visible_to_agent} onChange={(e) => setAgentNotes((n) => ({ ...n, visible_to_agent: e.target.checked }))} />
                Видно агенту
              </label>
              <Field label="Описание задачи"><textarea className="input" rows={3} value={agentNotes.task_description} onChange={(e) => setAgentNotes((n) => ({ ...n, task_description: e.target.value }))} /></Field>
              <Field label="Структура данных"><textarea className="input" rows={3} value={agentNotes.data_structure} onChange={(e) => setAgentNotes((n) => ({ ...n, data_structure: e.target.value }))} /></Field>
              <Field label="JSON описаний колонок"><textarea className="input mono" rows={5} value={columnDescriptions} onChange={(e) => setColumnDescriptions(e.target.value)} /></Field>
              <Field label="Дополнительные комментарии"><textarea className="input" rows={3} value={agentNotes.additional_comments} onChange={(e) => setAgentNotes((n) => ({ ...n, additional_comments: e.target.value }))} /></Field>
              <Field label="Предупреждение об утечке"><textarea className="input" rows={2} value={agentNotes.leakage_warning} onChange={(e) => setAgentNotes((n) => ({ ...n, leakage_warning: e.target.value }))} /></Field>
            </div>
          )}

          {step === 5 && <SourceEditor sources={sources} onChange={setSources} />}

          {step === 6 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <Card className="compact-card"><div className="metric-label">Датасет</div><div className="ds-title">{name || "-"}</div><div className="mono faint">{finalSlug || "-"}</div></Card>
                <Card className="compact-card"><div className="metric-label">Задача</div><div className="ds-title">{TASK_LABEL[taskType]}</div><div className="mono faint">{target || "-"} / {metric || "-"}</div></Card>
                <Card className="compact-card"><div className="metric-label">План разбиения</div><div className="ds-title">{MODE_LABEL[splitMode]}</div><div className="mono faint">{splitMode === "raw_split" ? selectedFile || "-" : mapping.train || "-"}</div></Card>
              </div>
              {warnings.length > 0 && <div className="warn-box">{warnings.join(" ")}</div>}
              <div className="grid-2">
                <Card className="compact-card">
                  <h4>Будущий dataset_config.json</h4>
                  <pre className="code" style={{ maxHeight: 360 }}>{payloadPreview}</pre>
                </Card>
                <Card className="compact-card">
                  <h4>Совместимый prepared/meta.json</h4>
                  <pre className="code">{JSON.stringify({ name, target_col: target, metric_name: metric, task_type: taskType, seed, notes: { description: agentNotes.task_description || desc } }, null, 2)}</pre>
                </Card>
              </div>
            </div>
          )}

          {error && <div className="error-line"><Icon name="alert" size={15} /> {error}</div>}
          {busy && <div className="spinner-row"><Spinner /> Работаю...</div>}
        </div>
      </div>
    </Modal>
  );
}
