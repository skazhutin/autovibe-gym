import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, type AgentNotes, type Task, type TaskConfig, type TaskSource } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, FieldInfo, Field, Modal, SelectDropdown, Skeleton, Spinner, Tabs, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { MiniHist } from "../components/charts";

const TABS = [
  { id: "overview", label: "Обзор", icon: "database" },
  { id: "preview", label: "Превью", icon: "table" },
  { id: "columns", label: "Колонки", icon: "sliders" },
  { id: "splits", label: "Сплиты", icon: "layers" },
  { id: "notes", label: "Заметки агенту", icon: "notebook" },
  { id: "sources", label: "Источники", icon: "external" },
  { id: "config", label: "Параметры", icon: "settings" },
];

type Split = "train" | "val" | "test";
const STATUS_LABEL: Record<string, string> = {
  prepared: "подготовлен",
  partial: "частичный",
  unprepared: "не подготовлен",
};
const TASK_LABEL: Record<string, string> = {
  auto: "auto",
  classification: "classification",
  regression: "regression",
  unknown: "unknown",
};
const METRIC_GOAL_LABEL: Record<string, string> = {
  max: "maximize",
  min: "меньше лучше",
};
const SPLIT_MODE_LABEL: Record<string, string> = {
  raw_split: "raw split",
  prepared_files: "готовые файлы",
};

const METRICS = {
  classification: ["f1_macro", "f1_weighted", "accuracy", "roc_auc", "logloss"],
  regression: ["neg_rmse", "rmse", "mae", "r2"],
  auto: ["f1_macro", "f1_weighted", "neg_rmse", "rmse"],
};

function inferGoal(metric: string): "max" | "min" {
  const m = metric.toLowerCase();
  if (m.startsWith("neg_")) return "max";
  return ["rmse", "rmsle", "mae", "mse", "logloss"].includes(m) ? "min" : "max";
}


function sourceText(task: Task) {
  const value = task.source && task.source !== "-" ? task.source : task.sources?.[0]?.name || task.sources?.[0]?.url || "-";
  return !value || value === "source" ? "-" : value;
}

function SplitSelector({ split, onChange, task }: { split: Split; onChange: (s: Split) => void; task: Task }) {
  const flags = { train: task.hasTrain, val: task.hasVal, test: task.hasTest };
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

function OverviewTab({ task, config }: { task: Task; config: TaskConfig | null }) {
  const warnings = config?.warnings ?? task.warnings ?? [];
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="grid-4">
        <Card className="metric-card"><span className="metric-label">Статус</span><span className="metric-num" style={{ fontSize: 24 }}>{STATUS_LABEL[task.status ?? (task.prepared ? "prepared" : "partial")]}</span></Card>
        <Card className="metric-card"><span className="metric-label">Строки</span><span className="metric-num">{task.rows?.toLocaleString() ?? "-"}</span></Card>
        <Card className="metric-card"><span className="metric-label">Признаки</span><span className="metric-num">{task.cols || "-"}</span></Card>
        <Card className="metric-card"><span className="metric-label">Seed</span><span className="metric-num">{task.seed ?? 42}</span></Card>
      </div>
      <Card>
        <div className="task-overview-grid">
          <div><span className="k">ID</span><span className="v mono">{task.id}</span></div>
          <div><span className="k">задача</span><span className="v">{TASK_LABEL[task.taskType ?? task.task] ?? task.task}</span></div>
          <div><span className="k">target</span><span className="v mono">{task.target}</span></div>
          <div><span className="k">метрика</span><span className="v mono">{task.metric} ({METRIC_GOAL_LABEL[task.metricGoal ?? "max"] ?? task.metricGoal})</span></div>
          <div><span className="k">источник</span><span className="v">{sourceText(task)}</span></div>
          <div><span className="k">создана</span><span className="v">{task.createdAt ? new Date(task.createdAt).toLocaleString() : "-"}</span></div>
          <div><span className="k">обновлен</span><span className="v">{task.updatedAt ? new Date(task.updatedAt).toLocaleString() : "-"}</span></div>
        </div>
        <div className="split-pills" style={{ marginTop: 16 }}>
          <Tag tone={task.hasTrain ? "green" : "neutral"}>train</Tag>
          <Tag tone={task.hasVal ? "green" : "neutral"}>val</Tag>
          <Tag tone={task.hasTest ? "green" : "neutral"}>test</Tag>
          {(task.tags ?? []).map((tag) => <Tag key={tag}>{tag}</Tag>)}
        </div>
      </Card>
      {warnings.length > 0 && <div className="warn-box">{warnings.join(" ")}</div>}
      {task.desc && <Card><div className="metric-label">Описание</div><p style={{ marginBottom: 0 }}>{task.desc}</p></Card>}
    </div>
  );
}

function PreviewTab({ id, task }: { id: string; task: Task }) {
  const [split, setSplit] = useState<Split>("train");
  const { data, loading } = useAsync(() => api.taskPreview(id, split, 50), [id, split]);
  if (loading && !data) return <Skeleton h={240} />;
  return (
    <div className="stack" style={{ gap: 12 }}>
      <SplitSelector split={split} onChange={setSplit} task={task} />
      {!data || !data.columns.length ? (
        <EmptyState icon="table" title={`Нет данных ${split}`} text={`${split}.csv недоступен.`} />
      ) : (
        <Card style={{ padding: 0 }}>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  {data.columns.map((c) => <th key={c} className={c === task.target ? "target-col" : undefined}>{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((v, j) => (
                      <td key={j} className={`mono${data.columns[j] === task.target ? " target-col" : ""}`} style={{ fontSize: 12.5 }}>
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

function ColumnsTab({ id, task, config }: { id: string; task: Task; config: TaskConfig | null }) {
  const [split, setSplit] = useState<Split>("train");
  const { data, loading } = useAsync(() => api.taskColumns(id, split), [id, split]);
  const ignored = new Set(config?.task.ignore_columns ?? []);
  const idCols = new Set(config?.task.id_columns ?? []);
  if (loading && !data) return <Skeleton h={240} />;
  return (
    <div className="stack" style={{ gap: 12 }}>
      <SplitSelector split={split} onChange={setSplit} task={task} />
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
                        {c.name === task.target && <Tag tone="accent">цель</Tag>}
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

function SplitsTab({ config }: { config: TaskConfig | null }) {
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
      <div className="task-split-meta">
        <Tag mono>{SPLIT_MODE_LABEL[splits.mode] ?? splits.mode}</Tag>
        <span>seed: <span className="mono">{splits.seed}</span></span>
        <span>перемешивание: <span className="mono">{splits.shuffle ?? true ? "да" : "нет"}</span></span>
        <span>стратификация: <span className="mono">{splits.stratify === "on" ? "вкл" : splits.stratify === "off" ? "выкл" : "авто"}</span></span>
        {splits.ratios && <span>доли: <span className="mono">{splits.ratios.train}/{splits.ratios.val}/{splits.ratios.test}</span></span>}
      </div>
    </Card>
  );
}

function ConfigTab({ id, config, archived, onSaved }: { id: string; config: TaskConfig | null; archived: boolean; onSaved: () => void }) {
  const nav = useNavigate();
  const [form, setForm] = useState<TaskConfig | null>(config);
  const [tagsStr, setTagsStr] = useState((config?.tags ?? []).join(", "));
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [archiving, setArchiving] = useState(false);

  async function doDelete() {
    setDeleting(true);
    try {
      await api.deleteTask(id);
      nav("/problems");
    } finally {
      setDeleting(false);
    }
  }

  async function doArchive() {
    setArchiving(true);
    try {
      if (archived) await api.unarchiveTasks([id]);
      else await api.archiveTasks([id]);
      nav("/problems");
    } finally {
      setArchiving(false);
    }
  }
  useEffect(() => {
    setForm(config);
    setTagsStr((config?.tags ?? []).join(", "));
  }, [config]);
  if (!form) return <Skeleton h={240} />;
  const task = form.task;
  const setTask = (patch: Partial<typeof task>) => setForm((s) => s && { ...s, task: { ...s.task, ...patch } });
  const metricSuggestions = useMemo(
    () => METRICS[task.task_type as keyof typeof METRICS] ?? METRICS.auto,
    [task.task_type]
  );
  function updateMetric(value: string) {
    setTask({ metric_name: value, metric_goal: inferGoal(value) });
  }
  async function save() {
    const current = form;
    if (!current) return;
    setBusy(true); setOk(false); setErr(null);
    try {
      await api.updateTaskConfig(id, {
        ...current,
        tags: tagsStr.split(",").map((s) => s.trim()).filter(Boolean),
      });
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
          <FieldInfo label="Тип задачи" info="classification — предсказание категорий (классов). regression — предсказание числа. auto — тип определится автоматически по данным.">
            <SelectDropdown value={task.task_type} options={[{ value: "auto", label: "auto" }, { value: "classification", label: "classification" }, { value: "regression", label: "regression" }]} onChange={(v) => setTask({ task_type: v as typeof task.task_type })} />
          </FieldInfo>
          <FieldInfo label="Target column" info="Название колонки с целевой переменной — то, что агент должен научиться предсказывать. Эта колонка никогда не включается в признаки." required>
            <input className="input mono" value={task.target_col} onChange={(e) => setTask({ target_col: e.target.value })} placeholder="target" />
          </FieldInfo>
          <FieldInfo label="Metric goal" info="Направление оптимизации: maximize — чем больше, тем лучше (accuracy, f1); minimize — чем меньше, тем лучше (rmse, mae). Выводится автоматически из имени метрики.">
            <SelectDropdown value={task.metric_goal} options={[{ value: "max", label: "maximize" }, { value: "min", label: "minimize" }]} onChange={(v) => setTask({ metric_goal: v as typeof task.metric_goal })} />
          </FieldInfo>
          <FieldInfo
            label={<>Метрика <a className="docs-link" href="https://scikit-learn.org/stable/modules/model_evaluation.html" target="_blank" rel="noopener noreferrer">все метрики sklearn ↗</a></>}
            info="Название метрики из sklearn.metrics. Считается на test после submit; агент видит только val. Примеры: f1_macro, accuracy, neg_rmse, roc_auc"
            required
          >
            <input className="input mono" value={task.metric_name} onChange={(e) => updateMetric(e.target.value)} list="config-metric-suggestions" />
            <datalist id="config-metric-suggestions">
              {metricSuggestions.map((m) => <option key={m} value={m} />)}
            </datalist>
          </FieldInfo>
          <div style={{ gridColumn: "span 2" }}>
            <FieldInfo label="Теги" info="Ключевые слова через запятую — используются для поиска и фильтрации. Пример: tabular, benchmark, uci">
              <input className="input" value={tagsStr} onChange={(e) => setTagsStr(e.target.value)} placeholder="tabular, benchmark" />
            </FieldInfo>
          </div>
        </div>
        <details className="disclosure">
          <summary>Расширенный конфиг</summary>
          <div className="grid-2" style={{ marginTop: 14 }}>
            <FieldInfo label="ID-колонки" info="Колонки-идентификаторы (id, row_id) — автоматически исключаются из признаков. Через запятую.">
              <input className="input mono" value={(task.id_columns ?? []).join(", ")} onChange={(e) => setTask({ id_columns: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) })} placeholder="id, row_id" />
            </FieldInfo>
            <FieldInfo label="Игнорируемые колонки" info="Колонки, которые точно не должны стать признаками X — например, служебные поля или дубли target.">
              <input className="input mono" value={(task.ignore_columns ?? []).join(", ")} onChange={(e) => setTask({ ignore_columns: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) })} />
            </FieldInfo>
            <FieldInfo label="Разрешённые библиотеки" info="Ограничение набора инструментов агента. Пусто = разрешено всё. Через запятую.">
              <input className="input mono" value={(task.allowed_libraries ?? []).join(", ")} onChange={(e) => setTask({ allowed_libraries: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) })} placeholder="sklearn, xgboost" />
            </FieldInfo>
          </div>
        </details>
        <div className="spread" style={{ alignItems: "center" }}>
          <div className="row">
            <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить конфиг"}</Button>
            {ok && <span className="success-inline"><Icon name="check" size={14} /> сохранено</span>}
            {err && <span className="error-inline">{err}</span>}
          </div>
          <div className="row" style={{ gap: 8 }}>
            <Button variant="ghost" icon="archive" onClick={doArchive} disabled={archiving}>{archiving ? <Spinner /> : archived ? "Разархивировать" : "Архивировать"}</Button>
            <Button variant="ghost" icon="trash" onClick={() => setConfirmDelete(true)}>Удалить датасет</Button>
          </div>
        </div>
      </div>

      {confirmDelete && (
        <Modal
          title="Удалить датасет?"
          width={420}
          onClose={() => setConfirmDelete(false)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setConfirmDelete(false)} disabled={deleting}>Отменить</Button>
              <Button variant="danger" onClick={doDelete} disabled={deleting}>{deleting ? <Spinner /> : "Удалить"}</Button>
            </>
          }
        >
          <p style={{ margin: 0, fontSize: 13, color: "var(--text-dim)", lineHeight: 1.55 }}>
            Все исходные и подготовленные файлы будут удалены без возможности восстановления.
          </p>
        </Modal>
      )}
    </Card>
  );
}

function SourcesTab({ id, config, onSaved }: { id: string; config: TaskConfig | null; onSaved: () => void }) {
  const [sources, setSources] = useState<TaskSource[]>(config?.sources ?? []);
  const [busy, setBusy] = useState(false);
  useEffect(() => setSources(config?.sources ?? []), [config]);
  if (!config) return <Skeleton h={200} />;
  const update = (idx: number, patch: TaskSource) => setSources((all) => all.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  async function save() {
    setBusy(true);
    try {
      await api.updateTaskConfig(id, { sources } as Partial<TaskConfig>);
      onSaved();
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="stack" style={{ gap: 12 }}>
      {sources.map((source, idx) => (
        <Card key={idx} className="compact-card">
          <div className="spread" style={{ marginBottom: 12 }}>
            <strong>Источник {idx + 1}</strong>
            <Button size="sm" variant="ghost" icon="trash" onClick={() => setSources((s) => s.filter((_, i) => i !== idx))}>Удалить</Button>
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
      <div className="row">
        <Button icon="plus" onClick={() => setSources((s) => [...s, {}])}>Добавить источник</Button>
        <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить источники"}</Button>
      </div>
    </div>
  );
}

function AgentNotesTab({ id, config, onSaved }: { id: string; config: TaskConfig | null; onSaved: () => void }) {
  const [notes, setNotes] = useState<AgentNotes | null>(config?.agent_notes ?? null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setNotes(config?.agent_notes ?? null);
  }, [config]);
  if (!notes) return <Skeleton h={220} />;
  const set = (patch: Partial<AgentNotes>) => setNotes((n) => n && { ...n, ...patch });
  async function save() {
    setBusy(true);
    setErr(null);
    try {
      await api.updateTaskConfig(id, { agent_notes: notes } as Partial<TaskConfig>);
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
        <div className="warn-box">Эти поля могут быть видны LLM-агенту. Не добавляйте тестовые метки или скрытые ответы.</div>
        <label className="check-row"><input type="checkbox" checked={notes.visible_to_agent} onChange={(e) => set({ visible_to_agent: e.target.checked })} /> Передавать заметки агенту</label>
        <FieldInfo label="Описание задачи" info="Что нужно предсказать и зачем. Если включено 'Передавать агенту' — попадает в промпт как контекст задачи.">
          <textarea className="input" rows={3} value={notes.task_description} onChange={(e) => set({ task_description: e.target.value })} />
        </FieldInfo>
        <FieldInfo label="Структура данных" info="Описание колонок: типы, форматы, особенности (пропуски, выбросы). Помогает агенту быстрее разобраться с данными.">
          <textarea className="input" rows={3} value={notes.data_structure} onChange={(e) => set({ data_structure: e.target.value })} />
        </FieldInfo>
        <FieldInfo label="Дополнительные комментарии" info="Любые пояснения для агента: известные задачи в данных, советы по feature engineering, особенности задачи.">
          <textarea className="input" rows={3} value={notes.additional_comments} onChange={(e) => set({ additional_comments: e.target.value })} />
        </FieldInfo>
        <FieldInfo label="Предупреждения" info="Потенциальные источники data leakage или других нюансов, о которых должен знать агент. Не указывайте тестовые ответы.">
          <textarea className="input" rows={2} value={notes.leakage_warning} onChange={(e) => set({ leakage_warning: e.target.value })} />
        </FieldInfo>
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
  const { data, loading, reload } = useAsync(() => api.getTask(id), [id]);
  const { data: config, loading: configLoading, reload: reloadConfig } = useAsync(() => api.getTaskConfig(id), [id]);
  const { data: archivedTasks } = useAsync(() => api.listArchivedTasks(), [id]);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [preparing, setPreparing] = useState(false);
  const [prepareErr, setPrepareErr] = useState<string | null>(null);
  const titleInputRef = useRef<HTMLInputElement>(null);
  const archived = (archivedTasks ?? []).some((task) => task.id === id);

  function refresh() {
    reload();
    reloadConfig();
  }

  function startEditName() {
    if (!data) return;
    setDraftName(data.name);
    setEditingName(true);
    setTimeout(() => titleInputRef.current?.select(), 0);
  }

  async function commitName() {
    const trimmed = draftName.trim();
    setEditingName(false);
    if (!trimmed || trimmed === data?.name) return;
    try {
      await api.updateTaskConfig(id, { name: trimmed } as Partial<TaskConfig>);
      refresh();
    } catch {}
  }

  function cancelName() {
    setEditingName(false);
  }

  async function prepareTask() {
    setPreparing(true);
    setPrepareErr(null);
    try {
      await api.prepareTask(id);
      refresh();
    } catch (e) {
      setPrepareErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPreparing(false);
    }
  }

  if (loading && !data) return <Skeleton h={300} />;
  if (!data) return <EmptyState icon="alert" title="Датасет не найден" action={<Button onClick={() => nav("/problems")}>К датасетам</Button>} />;

  return (
    <div>
      <button className="back-link" onClick={() => nav("/problems")}><Icon name="chevronLeft" size={16} /> Все задачи</button>
      <Card style={{ marginBottom: 20 }}>
        <div className="spread" style={{ alignItems: "flex-start" }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              {editingName ? (
                <input
                  ref={titleInputRef}
                  className="input ds-title-input"
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  onBlur={commitName}
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commitName(); } if (e.key === "Escape") cancelName(); }}
                  style={{ fontSize: 18, fontWeight: 600 }}
                />
              ) : (
                <div className="ds-title ds-title-editable" style={{ fontSize: 18 }} onClick={startEditName} title="Нажмите, чтобы изменить название">{data.name}</div>
              )}
            </div>
            <div className="stack" style={{ gap: 4 }}>
              <span className="mono faint" style={{ fontSize: 13 }}>metric: <strong className="mono" style={{ color: "var(--text)" }}>{data.metric}</strong></span>
              <span className="mono faint" style={{ fontSize: 13 }}>target: <strong className="mono" style={{ color: "var(--text)" }}>{data.target}</strong></span>
              <span className="mono faint" style={{ fontSize: 13 }}>path: {data.taskDir}</span>
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 12 }}>
            <Tag tone={data.status === "prepared" ? "green" : data.status === "partial" ? "blue" : "red"}>{STATUS_LABEL[data.status ?? (data.prepared ? "prepared" : "partial")]}</Tag>
            <div className="chip-metrics">
              <div className="chip-metric"><div><div className="cm-label">строки</div><div className="cm-val">{data.rows.toLocaleString()}</div></div></div>
              <span className="vline" />
              <div className="chip-metric"><div><div className="cm-label">признаки</div><div className="cm-val">{data.cols}</div></div></div>
            </div>
          </div>
        </div>
      </Card>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "overview" && <OverviewTab task={data} config={config} />}
      {tab === "preview" && <PreviewTab id={id} task={data} />}
      {tab === "columns" && <ColumnsTab id={id} task={data} config={config} />}
      {tab === "splits" && (configLoading ? <Skeleton h={220} /> : <SplitsTab config={config} />)}
      {tab === "config" && <ConfigTab id={id} config={config} archived={archived} onSaved={refresh} />}
      {tab === "sources" && <SourcesTab id={id} config={config} onSaved={refresh} />}
      {tab === "notes" && <AgentNotesTab id={id} config={config} onSaved={refresh} />}

      {!archived && !data.prepared && (
        <Card style={{ marginTop: 20 }}>
          <div className="spread" style={{ alignItems: "center", gap: 12 }}>
            <div className="stack" style={{ gap: 4 }}>
              <div style={{ fontWeight: 600 }}>Подготовка датасета</div>
              <div className="faint" style={{ fontSize: 13 }}>
                Соберет подготовленные `train` / `val` / `test` файлы по текущему конфигу задачи.
              </div>
              {prepareErr && <div className="error-inline">{prepareErr}</div>}
            </div>
            <Button variant="primary" onClick={prepareTask} disabled={preparing || configLoading}>
              {preparing ? <Spinner /> : "Подготовить"}
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
