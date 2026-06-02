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
  "Basic info",
  "Raw files",
  "Split mapping",
  "Task config",
  "Agent notes",
  "Sources",
  "Review",
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
    return <div className="empty-inline">No uploaded files yet.</div>;
  }
  return (
    <div className="table-wrap">
      <table className="data dataset-file-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Format</th>
            <th>Size</th>
            <th>Shape</th>
            <th>Status</th>
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
                  {f.status}
                </Tag>
              </td>
              <td>
                <div className="row" style={{ justifyContent: "flex-end" }}>
                  {f.readable && (
                    <Button size="sm" variant="ghost" icon="table" onClick={() => onPreview(f.path)}>
                      Preview
                    </Button>
                  )}
                  {f.status === "archive" && (
                    <Button size="sm" variant="ghost" icon="layers" onClick={() => onExtract(f.path)}>
                      Extract
                    </Button>
                  )}
                  {f.readable && (
                    <Button size="sm" icon="check" onClick={() => onSelect(f.path)}>
                      Use
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
                    {v === null ? <span className="faint">empty</span> : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="faint" style={{ padding: "10px 14px", fontSize: 12 }}>
        shown {preview.shown} of {preview.total ?? "unknown"} rows
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
            <strong>Source {idx + 1}</strong>
            <Button size="sm" variant="ghost" icon="trash" onClick={() => onChange(sources.filter((_, i) => i !== idx))}>
              Remove
            </Button>
          </div>
          <div className="grid-2">
            <Field label="Name"><input className="input" value={source.name ?? ""} onChange={(e) => update(idx, { name: e.target.value })} /></Field>
            <Field label="URL"><input className="input" value={source.url ?? ""} onChange={(e) => update(idx, { url: e.target.value })} /></Field>
            <Field label="License"><input className="input" value={source.license ?? ""} onChange={(e) => update(idx, { license: e.target.value })} /></Field>
            <Field label="Citation"><input className="input" value={source.citation ?? ""} onChange={(e) => update(idx, { citation: e.target.value })} /></Field>
            <Field label="Author"><input className="input" value={source.author ?? ""} onChange={(e) => update(idx, { author: e.target.value })} /></Field>
            <Field label="Organization"><input className="input" value={source.organization ?? ""} onChange={(e) => update(idx, { organization: e.target.value })} /></Field>
          </div>
          <Field label="Notes"><textarea className="input" rows={2} value={source.notes ?? ""} onChange={(e) => update(idx, { notes: e.target.value })} /></Field>
        </Card>
      ))}
      <Button icon="plus" onClick={() => onChange([...sources, { upload_date: new Date().toISOString().slice(0, 10) }])}>
        Add source
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
  const [suite, setSuite] = useState("");
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
    if (splitMode === "prepared_files" && !mapping.val && !createValFromTrain) out.push("Validation split is missing.");
    if (splitMode === "prepared_files" && !mapping.test) out.push("Without a test split, final benchmark scoring will be unavailable.");
    if (agentNotes.visible_to_agent && (agentNotes.task_description || agentNotes.additional_comments)) {
      out.push("Agent notes may be included in prompts. Keep them non-leaky.");
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
      if (!name.trim()) return "Dataset name is required.";
      if (!finalSlug) return "Dataset id is required.";
      if (!target.trim()) return "Target column is required.";
      if (!metric.trim()) return "Metric is required.";
    }
    if (idx === 1 && !readableFiles.length) return "Upload or extract at least one readable table file.";
    if (idx === 2) {
      if (splitMode === "raw_split") {
        const sum = ratios.train + ratios.val + ratios.test;
        if (!selectedFile) return "Select one raw table for splitting.";
        if (Math.abs(sum - 1) > 0.0001 || ratios.train <= 0 || ratios.val < 0 || ratios.test < 0) {
          return "Split ratios must sum to 1.0.";
        }
      } else if (!mapping.train) {
        return "Prepared-files mode requires a train file.";
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
      throw new Error("Column descriptions must be valid JSON.");
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
      suite: suite || null,
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
      return "Column descriptions JSON is invalid.";
    }
  })();

  return (
    <Modal
      title="Add dataset"
      width={980}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0 || busy}>
            Back
          </Button>
          {step < STEPS.length - 1 ? (
            <Button variant="primary" onClick={next} disabled={busy}>Next</Button>
          ) : (
            <Button variant="primary" onClick={createDataset} disabled={busy}>
              {busy ? <Spinner /> : "Create dataset"}
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
                <FieldInfo label="Dataset name" info="Human-readable dataset name. Example: Dry Bean Quality.">
                  <input className="input" value={name} onChange={(e) => setNameAndMaybeSlug(e.target.value)} />
                </FieldInfo>
                <FieldInfo label="Dataset id / slug" info="Folder name under datasets/. Use lowercase letters, numbers, dashes or underscores.">
                  <input className="input mono" value={slug} onChange={(e) => { setSlugTouched(true); setSlug(slugify(e.target.value)); }} />
                </FieldInfo>
              </div>
              <div className="grid-3">
                <FieldInfo label="Task type" info="Auto lets the backend infer task labels. Classification/regression help stratification and display.">
                  <select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                    <option value="auto">auto</option>
                    <option value="classification">classification</option>
                    <option value="regression">regression</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Target column" info="Column containing the label to predict. It must exist in train and validation data.">
                  <input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" />
                </FieldInfo>
                <FieldInfo label="Metric" info="Primary scoring metric used by the evaluator and shown in runs.">
                  <select className="input mono" value={metric} onChange={(e) => updateGoal(e.target.value)}>
                    {metricOptions.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </FieldInfo>
              </div>
              <div className="grid-3">
                <FieldInfo label="Metric goal" info="Whether higher or lower metric values are better. Auto-inferred from metric name, but editable.">
                  <select className="input" value={metricGoal} onChange={(e) => setMetricGoal(e.target.value as MetricGoal)}>
                    <option value="max">max</option>
                    <option value="min">min</option>
                  </select>
                </FieldInfo>
                <FieldInfo label="Seed" info="Random seed used for split creation and recorded in metadata.">
                  <input className="input mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value || 42))} />
                </FieldInfo>
                <FieldInfo label="Suite" info="Optional dataset suite or collection name.">
                  <input className="input" value={suite} onChange={(e) => setSuite(e.target.value)} />
                </FieldInfo>
              </div>
              <FieldInfo label="Tags" info="Comma-separated tags for filtering in Dataset Center." hint="example: tabular, benchmark">
                <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} />
              </FieldInfo>
              <FieldInfo label="Short description" info="Short, non-leaky description shown in Dataset Center.">
                <textarea className="input" rows={3} value={desc} onChange={(e) => setDesc(e.target.value)} />
              </FieldInfo>
              <div className="path-preview">Final path: <span className="mono">datasets/{finalSlug || "dataset-id"}</span></div>
            </div>
          )}

          {step === 1 && (
            <div className="stack" style={{ gap: 16 }}>
              <Card className="compact-card">
                <div className="spread">
                  <div>
                    <strong>Raw upload area</strong>
                    <div className="faint" style={{ fontSize: 12 }}>CSV, TSV, JSON, JSONL, XLSX, Parquet, Feather, ZIP, TAR, TGZ and single-file GZ.</div>
                  </div>
                  <div className="row">
                    <input ref={fileInput} type="file" multiple hidden onChange={(e) => uploadFiles(e.target.files)} />
                    <Button icon="upload" onClick={() => fileInput.current?.click()} disabled={busy}>Upload files</Button>
                  </div>
                </div>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <Field label="Add by URL">
                    <input className="input" value={uploadUrl} onChange={(e) => setUploadUrl(e.target.value)} placeholder="https://example.com/data.csv" />
                  </Field>
                  <div style={{ display: "flex", alignItems: "end" }}>
                    <Button icon="external" onClick={uploadByUrl} disabled={busy || !uploadUrl.trim()}>Download</Button>
                  </div>
                </div>
              </Card>
              <FileRows files={files} selected={selectedFile} onSelect={setSelectedFile} onPreview={loadPreview} onExtract={extractArchive} />
              {previewPath && <div className="faint">Previewing <span className="mono">{previewPath}</span></div>}
              <PreviewBox preview={preview} target={target} />
            </div>
          )}

          {step === 2 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="segmented">
                <button className={splitMode === "raw_split" ? "active" : ""} onClick={() => setSplitMode("raw_split")}>Split one raw table</button>
                <button className={splitMode === "prepared_files" ? "active" : ""} onClick={() => setSplitMode("prepared_files")}>Map prepared files</button>
              </div>
              {splitMode === "raw_split" ? (
                <>
                  <Field label="Raw table">
                    <select className="input mono" value={selectedFile ?? ""} onChange={(e) => setSelectedFile(e.target.value)}>
                      <option value="">Select a table</option>
                      {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                    </select>
                  </Field>
                  <div className="grid-3">
                    <Field label="Train ratio"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.train} onChange={(e) => setRatios((r) => ({ ...r, train: Number(e.target.value) }))} /></Field>
                    <Field label="Val ratio"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.val} onChange={(e) => setRatios((r) => ({ ...r, val: Number(e.target.value) }))} /></Field>
                    <Field label="Test ratio"><input className="input mono" type="number" min={0} max={1} step={0.01} value={ratios.test} onChange={(e) => setRatios((r) => ({ ...r, test: Number(e.target.value) }))} /></Field>
                  </div>
                </>
              ) : (
                <>
                  <div className="grid-3">
                    {(["train", "val", "test"] as const).map((split) => (
                      <Field key={split} label={`${split} file${split === "train" ? " *" : ""}`}>
                        <select className="input mono" value={mapping[split]} onChange={(e) => setMapping((m) => ({ ...m, [split]: e.target.value }))}>
                          <option value="">Not mapped</option>
                          {readableFiles.map((f) => <option key={f.path} value={f.path}>{f.path}</option>)}
                        </select>
                      </Field>
                    ))}
                  </div>
                  {!mapping.val && (
                    <label className="check-row">
                      <input type="checkbox" checked={createValFromTrain} onChange={(e) => setCreateValFromTrain(e.target.checked)} />
                      Create validation split from train
                    </label>
                  )}
                  {!mapping.val && createValFromTrain && (
                    <Field label="Validation ratio from train">
                      <input className="input mono" type="number" min={0.01} max={0.9} step={0.01} value={valRatio} onChange={(e) => setValRatio(Number(e.target.value))} />
                    </Field>
                  )}
                  {!mapping.test && <div className="warn-box">Without a test split, final benchmark scoring will be unavailable.</div>}
                </>
              )}
              <div className="grid-3">
                <Field label="Shuffle">
                  <select className="input" value={shuffle ? "true" : "false"} onChange={(e) => setShuffle(e.target.value === "true")}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                </Field>
                <Field label="Stratify">
                  <select className="input" value={stratify} onChange={(e) => setStratify(e.target.value as StratifyMode)}>
                    <option value="auto">auto</option>
                    <option value="on">on</option>
                    <option value="off">off</option>
                  </select>
                </Field>
                <Field label="Split seed"><input className="input mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value || 42))} /></Field>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <Field label="Task type"><select className="input" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}><option value="auto">auto</option><option value="classification">classification</option><option value="regression">regression</option></select></Field>
                <Field label="Target"><input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} /></Field>
                <Field label="Metric"><input className="input mono" value={metric} onChange={(e) => updateGoal(e.target.value)} /></Field>
              </div>
              <details className="disclosure">
                <summary>Advanced config</summary>
                <div className="grid-2" style={{ marginTop: 14 }}>
                  <Field label="ID columns"><input className="input mono" value={idColumns} onChange={(e) => setIdColumns(e.target.value)} placeholder="id, row_id" /></Field>
                  <Field label="Ignore columns"><input className="input mono" value={ignoreColumns} onChange={(e) => setIgnoreColumns(e.target.value)} /></Field>
                  <Field label="Sample weight column"><input className="input mono" value={sampleWeightCol} onChange={(e) => setSampleWeightCol(e.target.value)} /></Field>
                  <Field label="Group column"><input className="input mono" value={groupCol} onChange={(e) => setGroupCol(e.target.value)} /></Field>
                  <Field label="Time column"><input className="input mono" value={timeCol} onChange={(e) => setTimeCol(e.target.value)} /></Field>
                  <Field label="Positive label"><input className="input mono" value={positiveLabel} onChange={(e) => setPositiveLabel(e.target.value)} /></Field>
                  <Field label="Class labels"><input className="input mono" value={classLabels} onChange={(e) => setClassLabels(e.target.value)} /></Field>
                  <Field label="Allowed libraries"><input className="input mono" value={allowedLibraries} onChange={(e) => setAllowedLibraries(e.target.value)} placeholder="sklearn, xgboost" /></Field>
                  <Field label="Max runtime"><input className="input mono" type="number" value={maxRuntime} onChange={(e) => setMaxRuntime(e.target.value)} /></Field>
                  <Field label="Max steps"><input className="input mono" type="number" value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} /></Field>
                </div>
                <Field label="Constraints"><textarea className="input" rows={3} value={constraints} onChange={(e) => setConstraints(e.target.value)} /></Field>
              </details>
            </div>
          )}

          {step === 4 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="warn-box">These fields may be visible to the LLM-agent. Do not include test labels, hidden answers, or leakage.</div>
              <label className="check-row">
                <input type="checkbox" checked={agentNotes.visible_to_agent} onChange={(e) => setAgentNotes((n) => ({ ...n, visible_to_agent: e.target.checked }))} />
                Visible to agent
              </label>
              <Field label="Task description"><textarea className="input" rows={3} value={agentNotes.task_description} onChange={(e) => setAgentNotes((n) => ({ ...n, task_description: e.target.value }))} /></Field>
              <Field label="Data structure"><textarea className="input" rows={3} value={agentNotes.data_structure} onChange={(e) => setAgentNotes((n) => ({ ...n, data_structure: e.target.value }))} /></Field>
              <Field label="Column descriptions JSON"><textarea className="input mono" rows={5} value={columnDescriptions} onChange={(e) => setColumnDescriptions(e.target.value)} /></Field>
              <Field label="Additional comments"><textarea className="input" rows={3} value={agentNotes.additional_comments} onChange={(e) => setAgentNotes((n) => ({ ...n, additional_comments: e.target.value }))} /></Field>
              <Field label="Leakage warning"><textarea className="input" rows={2} value={agentNotes.leakage_warning} onChange={(e) => setAgentNotes((n) => ({ ...n, leakage_warning: e.target.value }))} /></Field>
            </div>
          )}

          {step === 5 && <SourceEditor sources={sources} onChange={setSources} />}

          {step === 6 && (
            <div className="stack" style={{ gap: 16 }}>
              <div className="grid-3">
                <Card className="compact-card"><div className="metric-label">Dataset</div><div className="ds-title">{name || "-"}</div><div className="mono faint">{finalSlug || "-"}</div></Card>
                <Card className="compact-card"><div className="metric-label">Task</div><div className="ds-title">{taskType}</div><div className="mono faint">{target || "-"} / {metric || "-"}</div></Card>
                <Card className="compact-card"><div className="metric-label">Split plan</div><div className="ds-title">{splitMode}</div><div className="mono faint">{splitMode === "raw_split" ? selectedFile || "-" : mapping.train || "-"}</div></Card>
              </div>
              {warnings.length > 0 && <div className="warn-box">{warnings.join(" ")}</div>}
              <div className="grid-2">
                <Card className="compact-card">
                  <h4>Generated dataset_config.json</h4>
                  <pre className="code" style={{ maxHeight: 360 }}>{payloadPreview}</pre>
                </Card>
                <Card className="compact-card">
                  <h4>Compatible prepared/meta.json</h4>
                  <pre className="code">{JSON.stringify({ name, target_col: target, metric_name: metric, task_type: taskType, seed, notes: { description: agentNotes.task_description || desc } }, null, 2)}</pre>
                </Card>
              </div>
            </div>
          )}

          {error && <div className="error-line"><Icon name="alert" size={15} /> {error}</div>}
          {busy && <div className="spinner-row"><Spinner /> Working...</div>}
        </div>
      </div>
    </Modal>
  );
}
