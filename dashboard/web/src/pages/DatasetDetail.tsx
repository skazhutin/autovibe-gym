import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, type AgentNotes, type Dataset, type DatasetConfig, type DatasetSource } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Skeleton, Spinner, Tabs, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { MiniHist } from "../components/charts";

const TABS = [
  { id: "overview", label: "Обзор", icon: "database" },
  { id: "preview", label: "Превью", icon: "table" },
  { id: "columns", label: "Колонки", icon: "sliders" },
  { id: "splits", label: "Сплиты", icon: "layers" },
  { id: "config", label: "Конфиг", icon: "settings" },
  { id: "sources", label: "Источники", icon: "external" },
  { id: "notes", label: "Заметки агента", icon: "notebook" },
];

type Split = "train" | "val" | "test";
const STATUS_LABEL: Record<string, string> = {
  prepared: "подготовлен",
  partial: "частичный",
  unprepared: "не подготовлен",
};
const TASK_LABEL: Record<string, string> = {
  auto: "авто",
  classification: "классификация",
  regression: "регрессия",
  unknown: "неизвестно",
};
const METRIC_GOAL_LABEL: Record<string, string> = {
  max: "больше лучше",
  min: "меньше лучше",
};
const SPLIT_MODE_LABEL: Record<string, string> = {
  raw_split: "raw split",
  prepared_files: "готовые файлы",
};

function sourceText(dataset: Dataset) {
  const value = dataset.source && dataset.source !== "-" ? dataset.source : dataset.sources?.[0]?.name || dataset.sources?.[0]?.url || "-";
  return !value || value === "source" ? "-" : value;
}

function SplitSelector({ split, onChange, dataset }: { split: Split; onChange: (s: Split) => void; dataset: Dataset }) {
  const flags = { train: dataset.hasTrain, val: dataset.hasVal, test: dataset.hasTest };
  return (
    <div className="segmented small">
      {(["train", "val", "test"] as const).map((s) => (
        <button key={s} className={split === s ? "active" : ""} onClick={() => onChange(s)} disabled={!flags[s]}>
          {s}
        </button>
      ))}
    </div>
  );
}

function OverviewTab({ dataset, config }: { dataset: Dataset; config: DatasetConfig | null }) {
  const warnings = config?.warnings ?? dataset.warnings ?? [];
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="grid-4">
        <Card className="metric-card"><span className="metric-label">Статус</span><span className="metric-num" style={{ fontSize: 24 }}>{STATUS_LABEL[dataset.status ?? (dataset.prepared ? "prepared" : "partial")]}</span></Card>
        <Card className="metric-card"><span className="metric-label">Строки</span><span className="metric-num">{dataset.rows?.toLocaleString() ?? "-"}</span></Card>
        <Card className="metric-card"><span className="metric-label">Признаки</span><span className="metric-num">{dataset.cols || "-"}</span></Card>
        <Card className="metric-card"><span className="metric-label">Seed</span><span className="metric-num">{dataset.seed ?? 42}</span></Card>
      </div>
      <Card>
        <div className="dataset-overview-grid">
          <div><span className="k">ID</span><span className="v mono">{dataset.id}</span></div>
          <div><span className="k">задача</span><span className="v">{TASK_LABEL[dataset.taskType ?? dataset.task] ?? dataset.task}</span></div>
          <div><span className="k">target</span><span className="v mono">{dataset.target}</span></div>
          <div><span className="k">метрика</span><span className="v mono">{dataset.metric} ({METRIC_GOAL_LABEL[dataset.metricGoal ?? "max"] ?? dataset.metricGoal})</span></div>
          <div><span className="k">источник</span><span className="v">{sourceText(dataset)}</span></div>
          <div><span className="k">создан</span><span className="v">{dataset.createdAt ? new Date(dataset.createdAt).toLocaleString() : "-"}</span></div>
          <div><span className="k">обновлен</span><span className="v">{dataset.updatedAt ? new Date(dataset.updatedAt).toLocaleString() : "-"}</span></div>
        </div>
        <div className="split-pills" style={{ marginTop: 16 }}>
          <Tag tone={dataset.hasTrain ? "green" : "neutral"}>train</Tag>
          <Tag tone={dataset.hasVal ? "green" : "neutral"}>val</Tag>
          <Tag tone={dataset.hasTest ? "green" : "neutral"}>test</Tag>
          {(dataset.tags ?? []).map((tag) => <Tag key={tag}>{tag}</Tag>)}
        </div>
      </Card>
      {warnings.length > 0 && <div className="warn-box">{warnings.join(" ")}</div>}
      {dataset.desc && <Card><div className="metric-label">Описание</div><p style={{ marginBottom: 0 }}>{dataset.desc}</p></Card>}
    </div>
  );
}

function PreviewTab({ id, dataset }: { id: string; dataset: Dataset }) {
  const [split, setSplit] = useState<Split>("train");
  const { data, loading } = useAsync(() => api.datasetPreview(id, split, 50), [id, split]);
  if (loading && !data) return <Skeleton h={240} />;
  return (
    <div className="stack" style={{ gap: 12 }}>
      <SplitSelector split={split} onChange={setSplit} dataset={dataset} />
      {!data || !data.columns.length ? (
        <EmptyState icon="table" title={`Нет данных ${split}`} text={`${split}.csv недоступен.`} />
      ) : (
        <Card style={{ padding: 0 }}>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  {data.columns.map((c) => <th key={c} className={c === dataset.target ? "target-col" : undefined}>{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((v, j) => (
                      <td key={j} className={`mono${data.columns[j] === dataset.target ? " target-col" : ""}`} style={{ fontSize: 12.5 }}>
                        {v === null ? <span className="faint">пусто</span> : String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="faint" style={{ padding: "10px 14px", fontSize: 12 }}>
            показано {data.shown} из {data.total ?? "неизвестно"} строк
          </div>
        </Card>
      )}
    </div>
  );
}

function ColumnsTab({ id, dataset, config }: { id: string; dataset: Dataset; config: DatasetConfig | null }) {
  const [split, setSplit] = useState<Split>("train");
  const { data, loading } = useAsync(() => api.datasetColumns(id, split), [id, split]);
  const ignored = new Set(config?.task.ignore_columns ?? []);
  const idCols = new Set(config?.task.id_columns ?? []);
  if (loading && !data) return <Skeleton h={240} />;
  return (
    <div className="stack" style={{ gap: 12 }}>
      <SplitSelector split={split} onChange={setSplit} dataset={dataset} />
      {!data || !data.length ? (
        <EmptyState icon="sliders" title={`Нет статистики колонок для ${split}`} />
      ) : (
        <Card style={{ padding: 0 }}>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Колонка</th><th>Тип</th><th>Вид</th><th>Пропуски</th><th>Уникальных</th><th>Маркеры</th><th>Распределение</th></tr></thead>
              <tbody>
                {data.map((c) => (
                  <tr key={c.name}>
                    <td className="mono">{c.name}</td>
                    <td className="mono faint">{c.dtype}</td>
                    <td><Tag tone={c.kind === "numeric" ? "blue" : "neutral"}>{c.kind === "numeric" ? "числовая" : "категориальная"}</Tag></td>
                    <td className="mono" style={{ color: c.missingPct > 0 ? "var(--orange)" : "var(--text-dim)" }}>{c.missingPct}%</td>
                    <td className="mono faint">{c.unique}</td>
                    <td>
                      <div className="split-pills">
                        {c.name === dataset.target && <Tag tone="accent">цель</Tag>}
                        {(c.ignored || ignored.has(c.name)) && <Tag tone="red">игнор</Tag>}
                        {(c.idColumn || idCols.has(c.name)) && <Tag tone="blue">id</Tag>}
                      </div>
                    </td>
                    <td><MiniHist data={c.hist} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function SplitsTab({ config }: { config: DatasetConfig | null }) {
  if (!config) return <Skeleton h={220} />;
  const splits = config.splits;
  return (
    <Card style={{ padding: 0 }}>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Сплит</th><th>Подготовленный файл</th><th>Исходный файл</th><th>Строки</th><th>Колонки</th></tr></thead>
          <tbody>
            {(["train", "val", "test"] as const).map((split) => {
              const item = splits[split];
              return (
                <tr key={split}>
                  <td><Tag tone={item ? "green" : "neutral"}>{split}</Tag></td>
                  <td className="mono">{item?.path ?? "-"}</td>
                  <td className="mono faint">{item?.source_path ?? "-"}</td>
                  <td className="mono">{item?.rows ?? "-"}</td>
                  <td className="mono">{item?.cols ?? "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="dataset-split-meta">
        <Tag mono>{SPLIT_MODE_LABEL[splits.mode] ?? splits.mode}</Tag>
        <span>seed: <span className="mono">{splits.seed}</span></span>
        <span>перемешивание: <span className="mono">{splits.shuffle ?? true ? "да" : "нет"}</span></span>
        <span>стратификация: <span className="mono">{splits.stratify === "on" ? "вкл" : splits.stratify === "off" ? "выкл" : "авто"}</span></span>
        {splits.ratios && <span>доли: <span className="mono">{splits.ratios.train}/{splits.ratios.val}/{splits.ratios.test}</span></span>}
      </div>
    </Card>
  );
}

function ConfigTab({ id, config, onSaved }: { id: string; config: DatasetConfig | null; onSaved: () => void }) {
  const [form, setForm] = useState<DatasetConfig | null>(config);
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => setForm(config), [config]);
  if (!form) return <Skeleton h={240} />;
  const task = form.task;
  const setTask = (patch: Partial<typeof task>) => setForm((s) => s && { ...s, task: { ...s.task, ...patch } });
  async function save() {
    const current = form;
    if (!current) return;
    setBusy(true);
    setOk(false);
    setErr(null);
    try {
      await api.updateDatasetConfig(id, current);
      setOk(true);
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card>
      <div className="stack" style={{ gap: 16 }}>
        <div className="grid-3">
          <Field label="Имя"><input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="Тип задачи">
            <select className="input" value={task.task_type} onChange={(e) => setTask({ task_type: e.target.value as typeof task.task_type })}>
              <option value="auto">авто</option><option value="classification">классификация</option><option value="regression">регрессия</option>
            </select>
          </Field>
          <Field label="Цель метрики">
            <select className="input" value={task.metric_goal} onChange={(e) => setTask({ metric_goal: e.target.value as typeof task.metric_goal })}>
              <option value="max">больше лучше</option><option value="min">меньше лучше</option>
            </select>
          </Field>
          <Field label="Target column"><input className="input mono" value={task.target_col} onChange={(e) => setTask({ target_col: e.target.value })} /></Field>
          <Field label="Метрика"><input className="input mono" value={task.metric_name} onChange={(e) => setTask({ metric_name: e.target.value })} /></Field>
        </div>
        <details className="disclosure">
          <summary>Расширенный конфиг</summary>
          <div className="grid-2" style={{ marginTop: 14 }}>
            <Field label="ID-колонки"><input className="input mono" value={(task.id_columns ?? []).join(", ")} onChange={(e) => setTask({ id_columns: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) })} /></Field>
            <Field label="Игнорируемые колонки"><input className="input mono" value={(task.ignore_columns ?? []).join(", ")} onChange={(e) => setTask({ ignore_columns: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) })} /></Field>
            <Field label="Колонка весов"><input className="input mono" value={task.sample_weight_col ?? ""} onChange={(e) => setTask({ sample_weight_col: e.target.value || null })} /></Field>
            <Field label="Group column"><input className="input mono" value={task.group_col ?? ""} onChange={(e) => setTask({ group_col: e.target.value || null })} /></Field>
            <Field label="Временная колонка"><input className="input mono" value={task.time_col ?? ""} onChange={(e) => setTask({ time_col: e.target.value || null })} /></Field>
            <Field label="Положительный класс"><input className="input mono" value={task.positive_label ?? ""} onChange={(e) => setTask({ positive_label: e.target.value || null })} /></Field>
          </div>
        </details>
        <div className="row">
          <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить конфиг"}</Button>
          {ok && <span className="success-inline"><Icon name="check" size={14} /> сохранено</span>}
          {err && <span className="error-inline">{err}</span>}
        </div>
      </div>
    </Card>
  );
}

function SourcesTab({ id, config, onSaved }: { id: string; config: DatasetConfig | null; onSaved: () => void }) {
  const [sources, setSources] = useState<DatasetSource[]>(config?.sources ?? []);
  const [busy, setBusy] = useState(false);
  useEffect(() => setSources(config?.sources ?? []), [config]);
  if (!config) return <Skeleton h={200} />;
  const update = (idx: number, patch: DatasetSource) => setSources((all) => all.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  async function save() {
    setBusy(true);
    try {
      await api.updateDatasetConfig(id, { sources } as Partial<DatasetConfig>);
      onSaved();
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="stack" style={{ gap: 12 }}>
      {sources.map((source, idx) => (
        <Card key={idx}>
          <div className="spread" style={{ marginBottom: 12 }}>
            <strong>Источник {idx + 1}</strong>
            <Button size="sm" variant="ghost" icon="trash" onClick={() => setSources((s) => s.filter((_, i) => i !== idx))}>Удалить</Button>
          </div>
          <div className="grid-2">
            <Field label="Название"><input className="input" value={source.name ?? ""} onChange={(e) => update(idx, { name: e.target.value })} /></Field>
            <Field label="URL"><input className="input" value={source.url ?? ""} onChange={(e) => update(idx, { url: e.target.value })} /></Field>
            <Field label="Лицензия"><input className="input" value={source.license ?? ""} onChange={(e) => update(idx, { license: e.target.value })} /></Field>
            <Field label="Цитирование"><input className="input" value={source.citation ?? ""} onChange={(e) => update(idx, { citation: e.target.value })} /></Field>
          </div>
          <Field label="Заметки"><textarea className="input" rows={2} value={source.notes ?? ""} onChange={(e) => update(idx, { notes: e.target.value })} /></Field>
        </Card>
      ))}
      <div className="row">
        <Button icon="plus" onClick={() => setSources((s) => [...s, {}])}>Добавить источник</Button>
        <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить источники"}</Button>
      </div>
    </div>
  );
}

function AgentNotesTab({ id, config, onSaved }: { id: string; config: DatasetConfig | null; onSaved: () => void }) {
  const [notes, setNotes] = useState<AgentNotes | null>(config?.agent_notes ?? null);
  const [cols, setCols] = useState("{}");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setNotes(config?.agent_notes ?? null);
    setCols(JSON.stringify(config?.agent_notes?.column_descriptions ?? {}, null, 2));
  }, [config]);
  if (!notes) return <Skeleton h={220} />;
  const set = (patch: Partial<AgentNotes>) => setNotes((n) => n && { ...n, ...patch });
  async function save() {
    setBusy(true);
    setErr(null);
    try {
      const parsed = JSON.parse(cols || "{}");
      await api.updateDatasetConfig(id, { agent_notes: { ...notes, column_descriptions: parsed } } as Partial<DatasetConfig>);
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card>
      <div className="stack" style={{ gap: 14 }}>
        <div className="warn-box">Эти поля могут быть видны LLM-агенту. Не добавляйте тестовые метки, скрытые ответы или утечки.</div>
        <label className="check-row"><input type="checkbox" checked={notes.visible_to_agent} onChange={(e) => set({ visible_to_agent: e.target.checked })} /> Видно агенту</label>
        <Field label="Описание задачи"><textarea className="input" rows={3} value={notes.task_description} onChange={(e) => set({ task_description: e.target.value })} /></Field>
        <Field label="Структура данных"><textarea className="input" rows={3} value={notes.data_structure} onChange={(e) => set({ data_structure: e.target.value })} /></Field>
        <Field label="JSON описаний колонок"><textarea className="input mono" rows={6} value={cols} onChange={(e) => setCols(e.target.value)} /></Field>
        <Field label="Дополнительные комментарии"><textarea className="input" rows={3} value={notes.additional_comments} onChange={(e) => set({ additional_comments: e.target.value })} /></Field>
        <Field label="Предупреждение об утечке"><textarea className="input" rows={2} value={notes.leakage_warning} onChange={(e) => set({ leakage_warning: e.target.value })} /></Field>
        <div className="row">
          <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить заметки"}</Button>
          {err && <span className="error-inline">{err}</span>}
        </div>
      </div>
    </Card>
  );
}

export default function DatasetDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [tab, setTab] = useState("overview");
  const { data, loading, reload } = useAsync(() => api.getDataset(id), [id]);
  const { data: config, loading: configLoading, reload: reloadConfig } = useAsync(() => api.getDatasetConfig(id), [id]);

  function refresh() {
    reload();
    reloadConfig();
  }

  if (loading && !data) return <Skeleton h={300} />;
  if (!data) return <EmptyState icon="alert" title="Датасет не найден" action={<Button onClick={() => nav("/datasets")}>К датасетам</Button>} />;

  return (
    <div>
      <button className="back-link" onClick={() => nav("/datasets")}><Icon name="chevronLeft" size={16} /> Все датасеты</button>
      <Card style={{ marginBottom: 20 }}>
        <div className="spread">
          <div>
            <div className="ds-title" style={{ fontSize: 18 }}>{data.name}</div>
            <div className="run-meta-line" style={{ margin: "10px 0 0" }}>
              <Tag tone={data.status === "prepared" ? "green" : data.status === "partial" ? "blue" : "red"}>{STATUS_LABEL[data.status ?? (data.prepared ? "prepared" : "partial")]}</Tag>
              <span className="mono faint">метрика: {data.metric}</span>
              <span className="mono faint">target: {data.target}</span>
              <span className="mono faint">путь: {data.datasetDir}</span>
            </div>
          </div>
          <div className="chip-metrics">
            <div className="chip-metric"><div><div className="cm-label">строки</div><div className="cm-val">{data.rows.toLocaleString()}</div></div></div>
            <span className="vline" />
            <div className="chip-metric"><div><div className="cm-label">признаки</div><div className="cm-val">{data.cols}</div></div></div>
          </div>
        </div>
      </Card>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "overview" && <OverviewTab dataset={data} config={config} />}
      {tab === "preview" && <PreviewTab id={id} dataset={data} />}
      {tab === "columns" && <ColumnsTab id={id} dataset={data} config={config} />}
      {tab === "splits" && (configLoading ? <Skeleton h={220} /> : <SplitsTab config={config} />)}
      {tab === "config" && <ConfigTab id={id} config={config} onSaved={refresh} />}
      {tab === "sources" && <SourcesTab id={id} config={config} onSaved={refresh} />}
      {tab === "notes" && <AgentNotesTab id={id} config={config} onSaved={refresh} />}
    </div>
  );
}
