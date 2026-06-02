import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  type DatasetConfig,
  type DatasetFileEntry,
  type DatasetPreview,
  type DatasetPrepareResult,
  type DatasetValidation,
} from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Skeleton, Spinner, Tabs, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const TABS = [
  { id: "config", label: "Конфиг", icon: "settings" },
  { id: "preview", label: "Превью", icon: "table" },
  { id: "validate", label: "Валидация", icon: "check2" },
  { id: "prepared", label: "Prepared", icon: "database" },
];

const FORMAT_OPTIONS = ["auto", "csv", "csv.gz", "tsv", "txt", "parquet", "xlsx", "xls", "json", "jsonl", "ndjson", "zip", "feather", "orc"];

function asString(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function linesToList(text: string) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function splitKeys(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.map((item) => formatValue(item)).join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function defaultRawEntry(index: number, role: "base" | "table" = "table"): DatasetFileEntry {
  return {
    logical_name: index === 0 ? "table_1" : `table_${index + 1}`,
    role,
    source_type: "upload",
    url: "",
    path: "",
    format: "auto",
    read_options: {},
    optional: false,
    archive_member: "",
  };
}

function defaultPreSplitEntries(existing: DatasetFileEntry[] = []): DatasetFileEntry[] {
  const byRole = new Map(existing.map((entry) => [entry.role, entry]));
  return ["train", "val", "test"].map((role) => {
    const current = byRole.get(role);
    if (current) return current;
    return {
      logical_name: role,
      role,
      source_type: "upload",
      url: "",
      path: "",
      format: "auto",
      read_options: {},
      optional: role === "val",
      archive_member: "",
    };
  });
}

function SplitBar({ train, val, test }: { train: number; val: number; test: number }) {
  const total = train + val + test;
  const normalize = (value: number) => (total > 0 ? `${(value / total) * 100}%` : "0%");
  return (
    <div className="splitbar-wrap">
      <div className="splitbar" aria-hidden="true">
        <span className="splitbar-segment train" style={{ width: normalize(train) }} />
        <span className="splitbar-segment val" style={{ width: normalize(val) }} />
        <span className="splitbar-segment test" style={{ width: normalize(test) }} />
      </div>
      <div className="splitbar-legend">
        <span><i className="train" /> train {train.toFixed(2)}</span>
        <span><i className="val" /> val {val.toFixed(2)}</span>
        <span><i className="test" /> test {test.toFixed(2)}</span>
      </div>
    </div>
  );
}

function IssueList({ title, items, tone }: { title: string; items: { message: string; field?: string; logical_name?: string }[]; tone: "red" | "blue" }) {
  if (!items.length) return null;
  return (
    <Card className={`issue-card ${tone}`}>
      <div className="section-title">{title}</div>
      <div className="stack" style={{ gap: 10 }}>
        {items.map((item, idx) => (
          <div key={`${item.message}-${idx}`} className="issue-row">
            <Tag tone={tone === "red" ? "red" : "blue"}>{item.logical_name || item.field || "dataset"}</Tag>
            <div>{item.message}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function DatasetDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [tab, setTab] = useState("config");
  const { data, loading, reload } = useAsync(() => api.getDataset(id), [id]);
  const [config, setConfig] = useState<DatasetConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveNote, setSaveNote] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewKey, setPreviewKey] = useState<string>("");
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [prepareResult, setPrepareResult] = useState<DatasetPrepareResult | null>(null);
  const [preparedSplit, setPreparedSplit] = useState("train");
  const [preparedVersion, setPreparedVersion] = useState(0);
  const [uploadFiles, setUploadFiles] = useState<Record<string, File | null>>({});

  const preparedPreview = useAsync(
    () => api.datasetPreview(id, preparedSplit, 25),
    [id, preparedSplit, preparedVersion]
  );

  useEffect(() => {
    if (data?.config) {
      setConfig(data.config);
      const first = data.config.ingestion.files[0];
      if (first) setPreviewKey(first.logical_name);
    }
  }, [data]);

  const rawMode = config?.ingestion.mode === "raw";
  const files = config?.ingestion.files ?? [];
  const baseTableOptions = files.map((entry) => entry.logical_name);
  const previewOptions = useMemo(() => {
    const options = files.map((entry) => ({ value: entry.logical_name, label: entry.logical_name }));
    if (rawMode) options.push({ value: "__joined__", label: "joined dataframe" });
    return options;
  }, [files, rawMode]);

  async function persist(showSuccess = true) {
    if (!config) return null;
    setSaving(true);
    setSaveNote(null);
    try {
      const updated = await api.saveDatasetConfig(id, config);
      setConfig(updated.config);
      if (showSuccess) setSaveNote("Конфиг сохранён.");
      reload();
      return updated;
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось сохранить конфиг.");
      return null;
    } finally {
      setSaving(false);
    }
  }

  function patchConfig(updater: (current: DatasetConfig) => DatasetConfig) {
    setConfig((current) => (current ? updater(current) : current));
    setSaveNote(null);
  }

  function patchFile(index: number, updater: (entry: DatasetFileEntry) => DatasetFileEntry) {
    patchConfig((current) => {
      const nextFiles = current.ingestion.files.map((entry, idx) => (idx === index ? updater(entry) : entry));
      return {
        ...current,
        ingestion: { ...current.ingestion, files: nextFiles },
        relations: normalizeRelations(current.relations, nextFiles),
      };
    });
  }

  function normalizeRelations(
    relations: DatasetConfig["relations"] | undefined,
    nextFiles: DatasetFileEntry[]
  ) {
    const baseTable = relations?.base_table && nextFiles.some((file) => file.logical_name === relations.base_table)
      ? relations.base_table
      : nextFiles[0]?.logical_name;
    return {
      base_table: baseTable,
      joins: relations?.joins ?? [],
    };
  }

  function switchMode(mode: "raw" | "pre_split") {
    patchConfig((current) => {
      const nextFiles =
        mode === "raw"
          ? (current.ingestion.files.length ? [current.ingestion.files[0], ...current.ingestion.files.slice(1)] : [defaultRawEntry(0, "base")]).map((entry, idx) => ({
              ...entry,
              logical_name: idx === 0 ? entry.logical_name || "table_1" : entry.logical_name || `table_${idx + 1}`,
              role: idx === 0 ? "base" : "table",
              optional: false,
            }))
          : defaultPreSplitEntries(current.ingestion.files);
      return {
        ...current,
        ingestion: { ...current.ingestion, mode, files: nextFiles },
        relations: mode === "raw" ? normalizeRelations(current.relations, nextFiles) : { base_table: "", joins: [] },
        split:
          mode === "pre_split"
            ? {
                ...current.split,
                strategy: "pre_split",
              }
            : {
                ...current.split,
                strategy: current.split.strategy === "pre_split" ? "stratified_random" : current.split.strategy,
              },
      };
    });
  }

  async function handleUpload(index: number) {
    if (!config) return;
    const entry = config.ingestion.files[index];
    const file = uploadFiles[entry.logical_name];
    if (!file) return;
    setBusyKey(`upload-${entry.logical_name}`);
    try {
      const form = new FormData();
      form.append("files", file);
      const [saved] = await api.uploadDatasetFiles(id, form);
      patchFile(index, (current) => ({
        ...current,
        path: saved.path,
        source_type: "upload",
      }));
      setUploadFiles((current) => ({ ...current, [entry.logical_name]: null }));
      setSaveNote(`Файл для ${entry.logical_name} загружен.`);
      reload();
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось загрузить файл.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleDownload(index: number) {
    if (!config) return;
    const entry = config.ingestion.files[index];
    if (!entry.url.trim()) {
      setSaveNote("Укажите URL для скачивания.");
      return;
    }
    setBusyKey(`download-${entry.logical_name}`);
    try {
      const [saved] = await api.downloadDatasetFiles(id, [
        {
          url: entry.url,
          suggested_name: entry.path.split("/").pop() || undefined,
        },
      ]);
      patchFile(index, (current) => ({
        ...current,
        path: saved.path,
        source_type: "url",
      }));
      setSaveNote(`Файл для ${entry.logical_name} скачан.`);
      reload();
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось скачать файл.");
    } finally {
      setBusyKey(null);
    }
  }

  async function runPreview() {
    if (!config) return;
    const saved = await persist(false);
    if (!saved) return;
    setBusyKey("preview");
    try {
      const result = previewKey === "__joined__"
        ? await api.previewDatasetSource(id, { joined: true, limit: 15 })
        : await api.previewDatasetSource(id, { logical_name: previewKey, limit: 15 });
      setPreview(result);
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось получить превью.");
    } finally {
      setBusyKey(null);
    }
  }

  async function runValidation() {
    const saved = await persist(false);
    if (!saved) return;
    setBusyKey("validate");
    try {
      setValidation(await api.validateDataset(id));
      setTab("validate");
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось провалидировать датасет.");
    } finally {
      setBusyKey(null);
    }
  }

  async function runPrepare() {
    const saved = await persist(false);
    if (!saved) return;
    setBusyKey("prepare");
    try {
      setPrepareResult(await api.prepareDataset(id));
      setPreparedVersion((value) => value + 1);
      reload();
      setTab("prepared");
    } catch (e) {
      setSaveNote(e instanceof Error ? e.message : "Не удалось подготовить датасет.");
    } finally {
      setBusyKey(null);
    }
  }

  if (loading && !data) return <Skeleton h={300} />;
  if (!data || !config) {
    return <EmptyState icon="alert" title="Датасет не найден" action={<Button onClick={() => nav("/datasets")}>К датасетам</Button>} />;
  }

  const source = (config.source ?? {}) as Record<string, unknown>;
  const datasetNotes = (config.dataset_notes ?? {}) as Record<string, unknown>;
  const task = (config.task ?? {}) as Record<string, unknown>;
  const split = (config.split ?? {}) as Record<string, unknown>;
  const preparation = (config.preparation ?? {}) as Record<string, unknown>;
  const preparedMeta = asRecord(prepareResult?.meta ?? data.preparedMeta);
  const preparedInputFormats = asRecord(preparedMeta.input_formats);
  const preparedSource = asRecord(preparedMeta.source);
  const preparedNotes = asRecord(preparedMeta.dataset_notes);
  const preparedWarnings = Array.isArray(preparedMeta.warnings) ? preparedMeta.warnings : [];
  const preparedJoinDiagnostics = Array.isArray(preparedMeta.join_diagnostics) ? preparedMeta.join_diagnostics : [];

  return (
    <div>
      <button className="back-link" onClick={() => nav("/datasets")}><Icon name="chevronLeft" size={16} /> Все датасеты</button>

      <Card style={{ marginBottom: 20 }}>
        <div className="spread dataset-hero">
          <div>
            <div className="ds-title" style={{ fontSize: 18 }}>{data.name}</div>
            <div className="run-meta-line" style={{ margin: "10px 0 0", flexWrap: "wrap" }}>
              <Tag tone={data.task === "Регрессия" ? "blue" : "neutral"}>{data.task}</Tag>
              <span className="mono faint">mode: {config.ingestion.mode}</span>
              <span className="mono faint">metric: {data.metric}</span>
              <span className="mono faint">target: {data.target}</span>
              {!data.prepared && <Tag tone="red">не подготовлен</Tag>}
            </div>
          </div>
          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <Button variant="secondary" onClick={() => persist(true)} disabled={saving}>{saving ? <Spinner /> : "Сохранить конфиг"}</Button>
            <Button variant="secondary" onClick={runValidation} disabled={!!busyKey}>{busyKey === "validate" ? <Spinner /> : "Проверить"}</Button>
            <Button variant="primary" onClick={runPrepare} disabled={!!busyKey}>{busyKey === "prepare" ? <Spinner /> : "Prepare dataset"}</Button>
          </div>
        </div>
        {saveNote && <div className="muted" style={{ marginTop: 12, color: saveNote.includes("не удалось") ? "var(--red)" : undefined }}>{saveNote}</div>}
      </Card>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === "config" && (
        <div className="stack" style={{ gap: 18 }}>
          <Card>
            <div className="section-title">Общее</div>
            <div className="grid-2">
              <Field label="Имя">
                <input className="input mono" value={config.name} onChange={(e) => patchConfig((current) => ({ ...current, name: e.target.value }))} />
              </Field>
              <Field label="Suite">
                <input className="input mono" value={config.suite} onChange={(e) => patchConfig((current) => ({ ...current, suite: e.target.value }))} />
              </Field>
            </div>
          </Card>

          <Card>
            <div className="section-title">Источник</div>
            <div className="grid-2">
              <Field label="Title"><input className="input" value={asString(source.title)} onChange={(e) => patchConfig((current) => ({ ...current, source: { ...current.source, title: e.target.value } }))} /></Field>
              <Field label="URL"><input className="input mono" value={asString(source.url)} onChange={(e) => patchConfig((current) => ({ ...current, source: { ...current.source, url: e.target.value } }))} /></Field>
              <Field label="License"><input className="input" value={asString(source.license)} onChange={(e) => patchConfig((current) => ({ ...current, source: { ...current.source, license: e.target.value } }))} /></Field>
              <Field label="Citation"><input className="input" value={asString(source.citation)} onChange={(e) => patchConfig((current) => ({ ...current, source: { ...current.source, citation: e.target.value } }))} /></Field>
            </div>
            <Field label="Описание источника">
              <textarea className="input" rows={3} value={asString(source.description)} onChange={(e) => patchConfig((current) => ({ ...current, source: { ...current.source, description: e.target.value } }))} />
            </Field>
          </Card>

          <Card>
            <div className="section-title">Комментарии и LLM-контекст</div>
            <div className="stack" style={{ gap: 14 }}>
              <Field label="Короткое описание">
                <input className="input" value={asString(datasetNotes.short_description)} onChange={(e) => patchConfig((current) => ({ ...current, dataset_notes: { ...current.dataset_notes, short_description: e.target.value } }))} />
              </Field>
              <Field label="LLM context / допкомментарии" hint="Здесь можно хранить доменный контекст, риски утечки и даже сам текст задания для модели.">
                <textarea className="input mono" rows={7} value={asString(datasetNotes.llm_context)} onChange={(e) => patchConfig((current) => ({ ...current, dataset_notes: { ...current.dataset_notes, llm_context: e.target.value } }))} />
              </Field>
              <div className="grid-2">
                <Field label="Warnings (по строке)">
                  <textarea className="input" rows={4} value={asStringList(datasetNotes.warnings).join("\n")} onChange={(e) => patchConfig((current) => ({ ...current, dataset_notes: { ...current.dataset_notes, warnings: linesToList(e.target.value) } }))} />
                </Field>
                <Field label="Known pitfalls (по строке)">
                  <textarea className="input" rows={4} value={asStringList(datasetNotes.known_pitfalls).join("\n")} onChange={(e) => patchConfig((current) => ({ ...current, dataset_notes: { ...current.dataset_notes, known_pitfalls: linesToList(e.target.value) } }))} />
                </Field>
              </div>
            </div>
          </Card>

          <Card>
            <div className="spread" style={{ marginBottom: 14 }}>
              <div className="section-title" style={{ marginBottom: 0 }}>Ingestion</div>
              <div className="row" style={{ gap: 8 }}>
                <Button variant={rawMode ? "primary" : "secondary"} size="sm" onClick={() => switchMode("raw")}>Raw mode</Button>
                <Button variant={!rawMode ? "primary" : "secondary"} size="sm" onClick={() => switchMode("pre_split")}>Already split</Button>
              </div>
            </div>

            {files.map((entry, index) => (
              <Card key={`${entry.logical_name}-${index}`} className="dataset-file-card">
                <div className="spread" style={{ marginBottom: 12 }}>
                  <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <Tag tone="dark">{entry.logical_name}</Tag>
                    <Tag mono>{entry.role}</Tag>
                    <Tag mono>{entry.format}</Tag>
                  </div>
                  {rawMode && files.length > 1 && (
                    <Button
                      size="sm"
                      variant="ghost"
                      icon="trash"
                      onClick={() =>
                        patchConfig((current) => {
                          const nextFiles = current.ingestion.files.filter((_, idx) => idx !== index);
                          return {
                            ...current,
                            ingestion: { ...current.ingestion, files: nextFiles.length ? nextFiles : [defaultRawEntry(0, "base")] },
                            relations: normalizeRelations(current.relations, nextFiles.length ? nextFiles : [defaultRawEntry(0, "base")]),
                          };
                        })
                      }
                    >
                      Удалить
                    </Button>
                  )}
                </div>

                <div className="grid-3">
                  <Field label="Logical name">
                    <input className="input mono" value={entry.logical_name} onChange={(e) => patchFile(index, (current) => ({ ...current, logical_name: e.target.value }))} />
                  </Field>
                  <Field label="Role">
                    <select className="input" value={entry.role} onChange={(e) => patchFile(index, (current) => ({ ...current, role: e.target.value }))}>
                      {rawMode ? (
                        <>
                          <option value="base">base</option>
                          <option value="table">table</option>
                        </>
                      ) : (
                        <>
                          <option value="train">train</option>
                          <option value="val">val</option>
                          <option value="test">test</option>
                        </>
                      )}
                    </select>
                  </Field>
                  <Field label="Source type">
                    <select className="input" value={entry.source_type} onChange={(e) => patchFile(index, (current) => ({ ...current, source_type: e.target.value }))}>
                      <option value="upload">upload</option>
                      <option value="url">url</option>
                      <option value="local">local</option>
                    </select>
                  </Field>
                </div>

                <div className="grid-2">
                  <Field label="Path" hint="Файл внутри datasets/<name>/raw_data/">
                    <input className="input mono" value={entry.path} onChange={(e) => patchFile(index, (current) => ({ ...current, path: e.target.value }))} placeholder="raw_data/train.parquet" />
                  </Field>
                  <Field label="Format">
                    <select className="input mono" value={entry.format} onChange={(e) => patchFile(index, (current) => ({ ...current, format: e.target.value }))}>
                      {FORMAT_OPTIONS.map((fmt) => <option key={fmt} value={fmt}>{fmt}</option>)}
                    </select>
                  </Field>
                </div>

                <div className="grid-2">
                  <Field label="Archive member">
                    <input className="input mono" value={entry.archive_member} onChange={(e) => patchFile(index, (current) => ({ ...current, archive_member: e.target.value }))} placeholder="nested/file.csv" />
                  </Field>
                  <Field label="URL">
                    <input className="input mono" value={entry.url} onChange={(e) => patchFile(index, (current) => ({ ...current, url: e.target.value }))} placeholder="https://example.com/data.zip" />
                  </Field>
                </div>

                <div className="grid-4">
                  <Field label="sep / delimiter">
                    <input className="input mono" value={asString(entry.read_options.sep ?? entry.read_options.delimiter)} onChange={(e) => patchFile(index, (current) => ({ ...current, read_options: { ...current.read_options, sep: e.target.value || undefined } }))} placeholder="; or ," />
                  </Field>
                  <Field label="encoding">
                    <input className="input mono" value={asString(entry.read_options.encoding)} onChange={(e) => patchFile(index, (current) => ({ ...current, read_options: { ...current.read_options, encoding: e.target.value || undefined } }))} placeholder="utf-8" />
                  </Field>
                  <Field label="header">
                    <input className="input mono" value={entry.read_options.header === undefined ? "" : String(entry.read_options.header)} onChange={(e) => patchFile(index, (current) => ({ ...current, read_options: { ...current.read_options, header: e.target.value === "" ? undefined : Number.isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value) } }))} placeholder="0 / none" />
                  </Field>
                  <Field label="sheet_name">
                    <input className="input mono" value={entry.read_options.sheet_name === undefined ? "" : String(entry.read_options.sheet_name)} onChange={(e) => patchFile(index, (current) => ({ ...current, read_options: { ...current.read_options, sheet_name: e.target.value === "" ? undefined : Number.isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value) } }))} placeholder="0" />
                  </Field>
                </div>

                <div className="spread" style={{ marginTop: 10, gap: 12, flexWrap: "wrap" }}>
                  <label className="checkbox-row">
                    <input type="checkbox" checked={Boolean(entry.read_options.lines)} onChange={(e) => patchFile(index, (current) => ({ ...current, read_options: { ...current.read_options, lines: e.target.checked || undefined } }))} />
                    <span>JSON Lines / NDJSON</span>
                  </label>
                  <label className="checkbox-row">
                    <input type="checkbox" checked={entry.optional} onChange={(e) => patchFile(index, (current) => ({ ...current, optional: e.target.checked }))} />
                    <span>Optional source</span>
                  </label>
                </div>

                <div className="row" style={{ marginTop: 14, gap: 10, flexWrap: "wrap" }}>
                  <label className="btn btn-secondary btn-sm" style={{ cursor: "pointer" }}>
                    <Icon name="upload" size={16} />
                    <span>{uploadFiles[entry.logical_name]?.name ?? "Выбрать файл"}</span>
                    <input hidden type="file" onChange={(e) => setUploadFiles((current) => ({ ...current, [entry.logical_name]: e.target.files?.[0] ?? null }))} />
                  </label>
                  <Button size="sm" variant="secondary" onClick={() => handleUpload(index)} disabled={busyKey !== null}>
                    {busyKey === `upload-${entry.logical_name}` ? <Spinner /> : "Загрузить"}
                  </Button>
                  <Button size="sm" variant="secondary" onClick={() => handleDownload(index)} disabled={busyKey !== null}>
                    {busyKey === `download-${entry.logical_name}` ? <Spinner /> : "Скачать по URL"}
                  </Button>
                </div>
              </Card>
            ))}

            <div className="row" style={{ marginTop: 14, gap: 10, flexWrap: "wrap" }}>
              {rawMode && (
                <Button size="sm" variant="secondary" icon="plus" onClick={() => patchConfig((current) => {
                  const nextFiles = [...current.ingestion.files, defaultRawEntry(current.ingestion.files.length)];
                  return {
                    ...current,
                    ingestion: { ...current.ingestion, files: nextFiles },
                    relations: normalizeRelations(current.relations, nextFiles),
                  };
                })}>
                  Добавить таблицу
                </Button>
              )}
              {!rawMode && (
                <Button size="sm" variant="secondary" icon="refresh" onClick={() => patchConfig((current) => ({
                  ...current,
                  ingestion: { ...current.ingestion, files: defaultPreSplitEntries(current.ingestion.files) },
                }))}>
                  Восстановить train/val/test
                </Button>
              )}
            </div>
          </Card>

          <Card>
            <div className="section-title">Файлы в raw_data</div>
            {data.files.length ? (
              <div className="stack" style={{ gap: 10 }}>
                {data.files.map((file) => (
                  <div key={file.path} className="issue-row">
                    <Tag mono>{file.format}</Tag>
                    <div className="mono" style={{ fontSize: 12.5 }}>{file.path}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="muted">Файлы ещё не загружены.</div>
            )}
          </Card>

          {rawMode && (
            <Card>
              <div className="section-title">Relations / joins</div>
              <div className="grid-2">
                <Field label="Base table">
                  <select className="input" value={asString(config.relations?.base_table)} onChange={(e) => patchConfig((current) => ({ ...current, relations: { ...current.relations, base_table: e.target.value } }))}>
                    {baseTableOptions.map((option) => <option key={option} value={option}>{option}</option>)}
                  </select>
                </Field>
                <Field label="Комментарий">
                  <div className="muted" style={{ fontSize: 13, paddingTop: 10 }}>
                    Base table задаёт grain предсказания. Остальные таблицы подтягиваются через left/right joins.
                  </div>
                </Field>
              </div>

              <div className="stack" style={{ gap: 12 }}>
                {(config.relations?.joins ?? []).map((join, index) => (
                  <Card key={`${join.left_table}-${join.right_table}-${index}`} className="join-card">
                    <div className="grid-4">
                      <Field label="Left table">
                        <select className="input" value={join.left_table} onChange={(e) => patchConfig((current) => {
                          const joins = [...(current.relations?.joins ?? [])];
                          joins[index] = { ...joins[index], left_table: e.target.value };
                          return { ...current, relations: { ...current.relations, joins } };
                        })}>
                          {baseTableOptions.map((option) => <option key={option} value={option}>{option}</option>)}
                        </select>
                      </Field>
                      <Field label="Right table">
                        <select className="input" value={join.right_table} onChange={(e) => patchConfig((current) => {
                          const joins = [...(current.relations?.joins ?? [])];
                          joins[index] = { ...joins[index], right_table: e.target.value };
                          return { ...current, relations: { ...current.relations, joins } };
                        })}>
                          {baseTableOptions.map((option) => <option key={option} value={option}>{option}</option>)}
                        </select>
                      </Field>
                      <Field label="Join type">
                        <select className="input" value={join.how} onChange={(e) => patchConfig((current) => {
                          const joins = [...(current.relations?.joins ?? [])];
                          joins[index] = { ...joins[index], how: e.target.value };
                          return { ...current, relations: { ...current.relations, joins } };
                        })}>
                          <option value="left">left</option>
                          <option value="inner">inner</option>
                          <option value="right">right</option>
                          <option value="outer">outer</option>
                        </select>
                      </Field>
                      <Field label="Удалить">
                        <Button size="sm" variant="ghost" icon="trash" onClick={() => patchConfig((current) => ({
                          ...current,
                          relations: {
                            ...current.relations,
                            joins: (current.relations?.joins ?? []).filter((_, idx) => idx !== index),
                          },
                        }))}>Удалить</Button>
                      </Field>
                    </div>
                    <div className="grid-2">
                      <Field label="left_on (через запятую)">
                        <input className="input mono" value={join.left_on.join(", ")} onChange={(e) => patchConfig((current) => {
                          const joins = [...(current.relations?.joins ?? [])];
                          joins[index] = { ...joins[index], left_on: splitKeys(e.target.value) };
                          return { ...current, relations: { ...current.relations, joins } };
                        })} />
                      </Field>
                      <Field label="right_on (через запятую)">
                        <input className="input mono" value={join.right_on.join(", ")} onChange={(e) => patchConfig((current) => {
                          const joins = [...(current.relations?.joins ?? [])];
                          joins[index] = { ...joins[index], right_on: splitKeys(e.target.value) };
                          return { ...current, relations: { ...current.relations, joins } };
                        })} />
                      </Field>
                    </div>
                  </Card>
                ))}
              </div>

              <div style={{ marginTop: 14 }}>
                <Button size="sm" variant="secondary" icon="plus" onClick={() => patchConfig((current) => ({
                  ...current,
                  relations: {
                    ...current.relations,
                    joins: [
                      ...(current.relations?.joins ?? []),
                      {
                        left_table: current.relations?.base_table || current.ingestion.files[0]?.logical_name || "",
                        right_table: current.ingestion.files[1]?.logical_name || current.ingestion.files[0]?.logical_name || "",
                        how: "left",
                        left_on: [],
                        right_on: [],
                      },
                    ],
                  },
                }))}>
                  Добавить join
                </Button>
              </div>
            </Card>
          )}

          <Card>
            <div className="section-title">Task и split</div>
            <div className="grid-3">
              <Field label="Task type">
                <select className="input" value={asString(task.type, "classification")} onChange={(e) => patchConfig((current) => ({ ...current, task: { ...current.task, type: e.target.value } }))}>
                  <option value="classification">classification</option>
                  <option value="regression">regression</option>
                </select>
              </Field>
              <Field label="Target column">
                <input className="input mono" value={asString(task.target_col)} onChange={(e) => patchConfig((current) => ({ ...current, task: { ...current.task, target_col: e.target.value } }))} />
              </Field>
              <Field label="Metric">
                <input className="input mono" value={asString(task.metric)} onChange={(e) => patchConfig((current) => ({ ...current, task: { ...current.task, metric: e.target.value } }))} placeholder="f1_macro / f1_weighted / neg_rmse" />
              </Field>
            </div>
            <div className="grid-2">
              <Field label="Forbidden columns / leakage columns (по строке)">
                <textarea className="input mono" rows={4} value={asStringList(task.forbidden_columns).join("\n")} onChange={(e) => patchConfig((current) => ({ ...current, task: { ...current.task, forbidden_columns: linesToList(e.target.value) } }))} />
              </Field>
              <Field label="Drop columns (по строке)">
                <textarea className="input mono" rows={4} value={asStringList(preparation.drop_columns).join("\n")} onChange={(e) => patchConfig((current) => ({ ...current, preparation: { ...current.preparation, drop_columns: linesToList(e.target.value) } }))} />
              </Field>
            </div>

            {rawMode ? (
              <div className="stack" style={{ gap: 14, marginTop: 14 }}>
                <div className="grid-4">
                  <Field label="Strategy">
                    <select className="input" value={asString(split.strategy, "stratified_random")} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, strategy: e.target.value } }))}>
                      <option value="stratified_random">stratified_random</option>
                      <option value="temporal">temporal</option>
                    </select>
                  </Field>
                  <Field label="Seed">
                    <input className="input mono" value={String(asNumber(split.seed, 42))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, seed: Number(e.target.value) || 0 } }))} />
                  </Field>
                  <Field label="Train fraction">
                    <input className="input mono" value={String(asNumber(split.train_fraction, 0.7))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, train_fraction: Number(e.target.value) || 0 } }))} />
                  </Field>
                  <Field label="Val fraction">
                    <input className="input mono" value={String(asNumber(split.val_fraction, 0.15))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, val_fraction: Number(e.target.value) || 0 } }))} />
                  </Field>
                </div>
                <div className="grid-2">
                  <Field label="Test fraction">
                    <input className="input mono" value={String(asNumber(split.test_fraction, 0.15))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, test_fraction: Number(e.target.value) || 0 } }))} />
                  </Field>
                  <Field label="Split preview">
                    <SplitBar
                      train={asNumber(split.train_fraction, 0.7)}
                      val={asNumber(split.val_fraction, 0.15)}
                      test={asNumber(split.test_fraction, 0.15)}
                    />
                  </Field>
                </div>
                {split.strategy === "temporal" && (
                  <div className="grid-2">
                    <Field label="Timestamp columns (через запятую)">
                      <input
                        className="input mono"
                        value={asStringList((split.timestamp as { source_columns?: string[] } | undefined)?.source_columns).join(", ")}
                        onChange={(e) => patchConfig((current) => ({
                          ...current,
                          split: {
                            ...current.split,
                            timestamp: {
                              ...((current.split.timestamp as Record<string, unknown>) || {}),
                              source_columns: splitKeys(e.target.value),
                            },
                          },
                        }))}
                      />
                    </Field>
                    <Field label="Timestamp format">
                      <input
                        className="input mono"
                        value={asString((split.timestamp as { format?: string } | undefined)?.format)}
                        onChange={(e) => patchConfig((current) => ({
                          ...current,
                          split: {
                            ...current.split,
                            timestamp: {
                              ...((current.split.timestamp as Record<string, unknown>) || {}),
                              format: e.target.value,
                            },
                          },
                        }))}
                        placeholder="%Y-%m-%d"
                      />
                    </Field>
                  </div>
                )}
              </div>
            ) : (
              <div className="grid-3" style={{ marginTop: 14 }}>
                <Field label="Seed">
                  <input className="input mono" value={String(asNumber(split.seed, 42))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, seed: Number(e.target.value) || 0 } }))} />
                </Field>
                <Field label="Create val from train">
                  <label className="checkbox-row" style={{ minHeight: 42 }}>
                    <input type="checkbox" checked={Boolean(split.create_val_from_train_if_missing)} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, create_val_from_train_if_missing: e.target.checked } }))} />
                    <span>Если val не загружен, выделить его из train</span>
                  </label>
                </Field>
                <Field label="val fraction from train">
                  <input className="input mono" value={String(asNumber(split.val_fraction_from_train, 0.15))} onChange={(e) => patchConfig((current) => ({ ...current, split: { ...current.split, val_fraction_from_train: Number(e.target.value) || 0 } }))} />
                </Field>
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "preview" && (
        <div className="stack" style={{ gap: 18 }}>
          <Card>
            <div className="spread" style={{ marginBottom: 14, gap: 10, flexWrap: "wrap" }}>
              <div className="section-title" style={{ marginBottom: 0 }}>Превью источника</div>
              <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
                <select className="input" value={previewKey} onChange={(e) => setPreviewKey(e.target.value)}>
                  {previewOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <Button variant="secondary" onClick={runPreview} disabled={!!busyKey}>{busyKey === "preview" ? <Spinner /> : "Обновить превью"}</Button>
              </div>
            </div>

            {!preview && <div className="muted">Сохраните конфиг и нажмите «Обновить превью», чтобы увидеть данные до подготовки.</div>}
            {preview && (
              <div className="stack" style={{ gap: 14 }}>
                <div className="grid-3">
                  <div className="statbox"><span className="k">format</span><span className="v mono">{preview.format}</span></div>
                  <div className="statbox"><span className="k">rows</span><span className="v mono">{preview.shape.rows === null ? "—" : preview.shape_is_approximate ? `≈${preview.shape.rows}` : preview.shape.rows}</span></div>
                  <div className="statbox"><span className="k">cols</span><span className="v mono">{preview.shape.cols ?? "—"}</span></div>
                </div>
                {preview.warnings.length > 0 && (
                  <div className="stack" style={{ gap: 8 }}>
                    {preview.warnings.map((warning, idx) => <Tag key={`${warning}-${idx}`} tone="blue">{warning}</Tag>)}
                  </div>
                )}
                {preview.archive_members && preview.archive_members.length > 0 && (
                  <Card className="dataset-inline-card">
                    <div className="section-title">ZIP contents</div>
                    <div className="stack" style={{ gap: 8 }}>
                      {preview.archive_members.map((member) => (
                        <div key={member.member} className="issue-row">
                          <Tag mono>{member.format}</Tag>
                          <span className="mono">{member.member}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
                {(Object.keys(preview.dtypes).length > 0 || Object.keys(preview.missing_counts).length > 0) && (
                  <div className="grid-2">
                    <Card className="dataset-inline-card">
                      <div className="section-title">Dtypes</div>
                      <div className="stack" style={{ gap: 8 }}>
                        {Object.entries(preview.dtypes).map(([column, dtype]) => (
                          <div key={column} className="issue-row">
                            <Tag mono>{column}</Tag>
                            <span className="mono">{dtype}</span>
                          </div>
                        ))}
                      </div>
                    </Card>
                    <Card className="dataset-inline-card">
                      <div className="section-title">Missing counts</div>
                      <div className="stack" style={{ gap: 8 }}>
                        {Object.entries(preview.missing_counts).map(([column, count]) => (
                          <div key={column} className="issue-row">
                            <Tag mono>{column}</Tag>
                            <span className="mono">{String(count)}</span>
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                )}
                {preview.target_distribution && (
                  <Card className="dataset-inline-card">
                    <div className="section-title">Target distribution</div>
                    <div className="stack" style={{ gap: 8 }}>
                      {Object.entries(preview.target_distribution).map(([label, value]) => (
                        <div key={label} className="issue-row">
                          <Tag mono>{label}</Tag>
                          <span className="mono">{formatValue(value)}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
                {preview.join_diagnostics && preview.join_diagnostics.length > 0 && (
                  <Card className="dataset-inline-card">
                    <div className="section-title">Join diagnostics</div>
                    <div className="stack" style={{ gap: 8 }}>
                      {preview.join_diagnostics.map((join, idx) => {
                        const info = asRecord(join);
                        return (
                          <div key={`${info.left_table ?? "left"}-${info.right_table ?? "right"}-${idx}`} className="issue-row">
                            <Tag tone="blue">{String(info.how ?? "join")}</Tag>
                            <div className="mono" style={{ fontSize: 12.5 }}>
                              {String(info.left_table ?? "left")} → {String(info.right_table ?? "right")} | rows {formatValue(info.left_rows)} → {formatValue(info.result_rows)} | x{formatValue(info.row_growth_ratio)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                )}
                {preview.columns.length > 0 ? (
                  <div className="table-wrap">
                    <table className="data">
                      <thead><tr>{preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                      <tbody>
                        {preview.rows.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            {row.map((value, cellIndex) => (
                              <td key={`${rowIndex}-${cellIndex}`} className="mono" style={{ fontSize: 12.5 }}>
                                {value === null ? <span className="faint">∅</span> : String(value)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <EmptyState icon="table" title="Нет строк для показа" text="Для ZIP-архива выберите archive_member или используйте joined preview после настройки таблиц." />
                )}
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "validate" && (
        <div className="stack" style={{ gap: 18 }}>
          <IssueList title="Ошибки" items={validation?.errors ?? []} tone="red" />
          <IssueList title="Предупреждения" items={validation?.warnings ?? []} tone="blue" />
          {!validation && <EmptyState icon="check2" title="Валидация ещё не запускалась" text="Сохраните конфиг и запустите проверку." />}
          {validation && !validation.errors.length && !validation.warnings.length && (
            <Card><div className="section-title">Конфиг прошёл проверку</div><div className="muted">Блокирующих ошибок и предупреждений не найдено.</div></Card>
          )}
        </div>
      )}

      {tab === "prepared" && (
        <div className="stack" style={{ gap: 18 }}>
          {Object.keys(preparedMeta).length > 0 && (
            <Card>
              <div className="section-title">Meta summary</div>
              <div className="grid-4">
                <div className="statbox"><span className="k">status</span><span className="v mono">{prepareResult?.status ?? "ready"}</span></div>
                <div className="statbox"><span className="k">train</span><span className="v mono">{formatValue(preparedMeta.n_train)}</span></div>
                <div className="statbox"><span className="k">val</span><span className="v mono">{formatValue(preparedMeta.n_val)}</span></div>
                <div className="statbox"><span className="k">test</span><span className="v mono">{formatValue(preparedMeta.n_test)}</span></div>
              </div>
              <div className="grid-4" style={{ marginTop: 12 }}>
                <div className="statbox"><span className="k">rows source</span><span className="v mono">{formatValue(preparedMeta.n_rows_source)}</span></div>
                <div className="statbox"><span className="k">rows prepared</span><span className="v mono">{formatValue(preparedMeta.n_rows_prepared)}</span></div>
                <div className="statbox"><span className="k">features</span><span className="v mono">{formatValue(preparedMeta.n_features)}</span></div>
                <div className="statbox"><span className="k">split strategy</span><span className="v mono">{formatValue(preparedMeta.split_strategy)}</span></div>
              </div>
              <div className="grid-2" style={{ marginTop: 12 }}>
                <Card className="dataset-inline-card">
                  <div className="section-title">Dataset details</div>
                  <div className="stack" style={{ gap: 8 }}>
                    <div className="issue-row"><Tag mono>target</Tag><span className="mono">{formatValue(preparedMeta.target_col)}</span></div>
                    <div className="issue-row"><Tag mono>metric</Tag><span className="mono">{formatValue(preparedMeta.metric)}</span></div>
                    <div className="issue-row"><Tag mono>seed</Tag><span className="mono">{formatValue(preparedMeta.seed)}</span></div>
                    <div className="issue-row"><Tag mono>source</Tag><span>{formatValue(preparedSource.title || preparedSource.url)}</span></div>
                    <div className="issue-row"><Tag mono>summary</Tag><span>{formatValue(preparedNotes.short_description)}</span></div>
                  </div>
                </Card>
                <Card className="dataset-inline-card">
                  <div className="section-title">Input formats</div>
                  <div className="stack" style={{ gap: 8 }}>
                    {Object.keys(preparedInputFormats).length > 0 ? Object.entries(preparedInputFormats).map(([name, fmt]) => (
                      <div key={name} className="issue-row">
                        <Tag mono>{name}</Tag>
                        <span className="mono">{formatValue(fmt)}</span>
                      </div>
                    )) : <div className="muted">Нет данных о входных форматах.</div>}
                  </div>
                </Card>
              </div>
              {preparedJoinDiagnostics.length > 0 && (
                <Card className="dataset-inline-card" style={{ marginTop: 12 }}>
                  <div className="section-title">Join diagnostics</div>
                  <div className="stack" style={{ gap: 8 }}>
                    {preparedJoinDiagnostics.map((join, idx) => {
                      const info = asRecord(join);
                      return (
                        <div key={`${info.left_table ?? "left"}-${info.right_table ?? "right"}-${idx}`} className="issue-row">
                          <Tag tone="blue">{String(info.how ?? "join")}</Tag>
                          <div className="mono" style={{ fontSize: 12.5 }}>
                            {String(info.left_table ?? "left")} → {String(info.right_table ?? "right")} | rows {formatValue(info.left_rows)} → {formatValue(info.result_rows)} | x{formatValue(info.row_growth_ratio)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              )}
              {preparedWarnings.length > 0 && (
                <Card className="dataset-inline-card" style={{ marginTop: 12 }}>
                  <div className="section-title">Warnings</div>
                  <div className="stack" style={{ gap: 8 }}>
                    {preparedWarnings.map((warning, idx) => <Tag key={`${warning}-${idx}`} tone="blue">{String(warning)}</Tag>)}
                  </div>
                </Card>
              )}
              <div style={{ marginTop: 12 }}>
                <div className="section-title">meta.json snapshot</div>
                <pre className="code">{JSON.stringify(preparedMeta, null, 2)}</pre>
              </div>
            </Card>
          )}

          <Card>
            <div className="spread" style={{ marginBottom: 14, gap: 10, flexWrap: "wrap" }}>
              <div className="section-title" style={{ marginBottom: 0 }}>Prepared split preview</div>
              <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
                <select className="input" value={preparedSplit} onChange={(e) => setPreparedSplit(e.target.value)}>
                  <option value="train">train</option>
                  <option value="val">val</option>
                  <option value="test">test</option>
                </select>
                <Button variant="secondary" onClick={() => setPreparedVersion((value) => value + 1)}>Обновить</Button>
              </div>
            </div>

            {preparedPreview.loading && !preparedPreview.data && <Skeleton h={220} />}
            {preparedPreview.data && preparedPreview.data.columns.length > 0 ? (
              <div className="table-wrap">
                <table className="data">
                  <thead><tr>{preparedPreview.data.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                  <tbody>
                    {preparedPreview.data.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((value, cellIndex) => (
                          <td key={`${rowIndex}-${cellIndex}`} className="mono" style={{ fontSize: 12.5 }}>
                            {value === null ? <span className="faint">∅</span> : String(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState icon="database" title="Prepared данных пока нет" text="Нажмите Prepare dataset, чтобы собрать совместимый train/val/test набор." />
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
