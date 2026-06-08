import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { api, type Task, type TaskConfig, type LaunchRunMode, type ModelRec } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS } from "../lib/format";
import { Button, Card, Dot, Field, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const MAX_SELECTED_MODES = 5;
const STORAGE_MODEL = "newrun_modelId";
const STORAGE_DATASET = "newrun_taskId";

const MODE_INFO: {
  id: LaunchRunMode; desc: string; env: boolean;
  hasSteps: boolean; hasShots: boolean; hasHint: boolean; hasThoughts: boolean;
}[] = [
  { id: "single",   env: false, desc: "Один полный ответ без обратной связи от среды.",                                                     hasSteps: false, hasShots: false, hasHint: false, hasThoughts: false },
  { id: "repeated", env: false, desc: "N независимых попыток (multi-shot), только скалярная val-метрика между ними.",                       hasSteps: false, hasShots: true,  hasHint: false, hasThoughts: false },
  { id: "iterative",env: true,  desc: "Итеративная среда без подсказок чеклиста: runtime и contract feedback.",                             hasSteps: true,  hasShots: false, hasHint: false, hasThoughts: true  },
  { id: "gym",      env: true,  desc: "Свободная интерактивная среда с неявными подсказками чеклиста DS-пайплайна.",                       hasSteps: true,  hasShots: false, hasHint: true,  hasThoughts: true  },
  { id: "fixed",    env: true,  desc: "Фиксированные стадии gym: EDA, preprocessing, feature engineering, model selection, tuning.",       hasSteps: true,  hasShots: false, hasHint: false, hasThoughts: true  },
];

type ModeParams = { maxSteps: number; shots: number; hintCooldown: number; enableThoughts: boolean; temp: number };
const DEFAULT_MODE_PARAMS: ModeParams = { maxSteps: 30, shots: 5, hintCooldown: 2, enableThoughts: false, temp: 0.4 };

const TASK_TYPES = [
  { value: "auto" as const,           label: "Авто" },
  { value: "classification" as const, label: "Классификация" },
  { value: "regression" as const,     label: "Регрессия" },
];

// ── Info tooltip (same pattern as Models page) ──────────────────────────────
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
      <span ref={dotRef} className="info-dot" onMouseEnter={show} onMouseLeave={() => setVisible(false)}>?</span>
      {visible && createPortal(<div className="tooltip-portal" style={{ top: pos.top, left: pos.left }}>{text}</div>, document.body)}
    </>
  );
}

function FI({ label, info, children }: { label: ReactNode; info: string; children: ReactNode }) {
  return (
    <Field label={<span className="field-info-label">{label}<Info text={info} /></span>}>
      {children}
    </Field>
  );
}

// ── Stepper ──────────────────────────────────────────────────────────────────
function Stepper({ value, onChange, min = 1, max = 999999, step = 1, suffix }: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; suffix?: string;
}) {
  return (
    <div className="stepper">
      <button type="button" onClick={() => onChange(Math.max(min, value - step))}>−</button>
      <input value={suffix ? `${value}${suffix}` : value}
        onChange={(e) => {
          const n = parseInt(e.target.value.replace(/\D/g, ""), 10);
          if (!Number.isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
        }} />
      <button type="button" onClick={() => onChange(Math.min(max, value + step))}>+</button>
    </div>
  );
}

// ── Model Picker Modal ───────────────────────────────────────────────────────
function ModelPickerModal({ models, current, onSelect, onClose }: {
  models: ModelRec[]; current: string; onSelect: (id: string) => void; onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return models.filter(m => !term || `${m.name} ${m.provider} ${m.baseUrl ?? ""}`.toLowerCase().includes(term));
  }, [models, q]);

  return createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-box" style={{ width: 520, maxWidth: "96vw" }} onClick={e => e.stopPropagation()}>
        <h3 className="modal-title">Выбрать модель</h3>
        <div className="filters" style={{ margin: "8px 0 4px" }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" placeholder="Поиск по имени, провайдеру…" value={q}
              onChange={e => setQ(e.target.value)} autoFocus />
          </div>
        </div>
        <div style={{ maxHeight: 360, overflowY: "auto", margin: "8px -4px 0" }}>
          {filtered.map(m => (
            <div key={m.id}
              className={`cmp-picker-row${current === m.id ? " selected" : ""}${m.online === false ? " disabled" : ""}`}
              onClick={() => { if (m.online !== false) { onSelect(m.id); onClose(); } }}>
              <Dot tone={m.online === false ? "red" : m.online ? "green" : "gray"} />
              <span className="mono" style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</span>
              <span className="tag">{m.provider}</span>
              <span className="tag mono">{(m.ctx / 1024).toFixed(0)}k ctx</span>
            </div>
          ))}
          {!filtered.length && <div className="muted" style={{ padding: "16px 8px", fontSize: 13 }}>Нет подходящих моделей.</div>}
        </div>
        <div className="modal-actions"><Button variant="ghost" onClick={onClose}>Закрыть</Button></div>
      </div>
    </div>,
    document.body
  );
}

// ── Task Picker Modal ───────────────────────────────────────────
function TaskPickerModal({ tasks, current, onSelect, onClose }: {
  tasks: Task[]; current: string; onSelect: (id: string) => void; onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return tasks.filter(d => !term || `${d.name} ${d.task} ${d.metric}`.toLowerCase().includes(term));
  }, [tasks, q]);

  return createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-box" style={{ width: 560, maxWidth: "96vw" }} onClick={e => e.stopPropagation()}>
        <h3 className="modal-title">Выбрать задачу</h3>
        <div className="filters" style={{ margin: "8px 0 4px" }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" placeholder="Поиск по названию, задаче, метрике…" value={q}
              onChange={e => setQ(e.target.value)} autoFocus />
          </div>
        </div>
        <div style={{ maxHeight: 360, overflowY: "auto", margin: "8px -4px 0" }}>
          {filtered.map(d => (
            <div key={d.id}
              className={`cmp-picker-row${current === d.id ? " selected" : ""}${!d.prepared ? " disabled" : ""}`}
              onClick={() => { if (d.prepared) { onSelect(d.id); onClose(); } }}>
              <span className="mono" style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
              <span className="tag">{d.task}</span>
              <span className="tag mono">{d.metric}</span>
              {!d.prepared && <span className="tag tag-red">не подготовлен</span>}
            </div>
          ))}
          {!filtered.length && <div className="muted" style={{ padding: "16px 8px", fontSize: 13 }}>Нет подходящих задач.</div>}
        </div>
        <div className="modal-actions"><Button variant="ghost" onClick={onClose}>Закрыть</Button></div>
      </div>
    </div>,
    document.body
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function NewRun() {
  const nav = useNavigate();
  const { data: models } = useAsync(() => api.listModels(), []);
  const { data: tasks } = useAsync(() => api.listTasks(), []);
  const { data: settings } = useAsync(() => api.getSettings(), []);
  const serverAvailable = !!(settings?.remote_ssh && settings?.remote_repo);
  const [execution, setExecution] = useState<"local" | "server">("local");

  // ── Model ──
  const [modelId, setModelId] = useState<string>(() => localStorage.getItem(STORAGE_MODEL) ?? "");
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [maxTokens, setMaxTokens] = useState(8192);

  // ── Mode ──
  const [selectedModes, setSelectedModes] = useState<LaunchRunMode[]>(["gym"]);
  const [expandedModes, setExpandedModes] = useState<Set<LaunchRunMode>>(new Set(["gym"] as LaunchRunMode[]));
  const [modeParams, setModeParams] = useState<Partial<Record<LaunchRunMode, ModeParams>>>({});

  // ── Task ──
  const [taskId, setTaskId] = useState<string>(() => localStorage.getItem(STORAGE_DATASET) ?? "");
  const [taskPickerOpen, setTaskPickerOpen] = useState(false);
  const [datasetSettingsOpen, setDatasetSettingsOpen] = useState(false);
  const [datasetConfig, setTaskConfig] = useState<TaskConfig | null>(null);
  const [datasetConfigLoading, setTaskConfigLoading] = useState(false);
  const [datasetConfigSaving, setTaskConfigSaving] = useState(false);
  const [dsSection, setDsSection] = useState<"config" | "notes" | null>(null);

  // ── Other ──
  const [seed, setSeed] = useState(42);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Persist selections
  useEffect(() => { if (modelId) localStorage.setItem(STORAGE_MODEL, modelId); }, [modelId]);
  useEffect(() => { if (taskId) localStorage.setItem(STORAGE_DATASET, taskId); }, [taskId]);

  useEffect(() => {
    if (settings) setExecution(settings.remote_enabled && serverAvailable ? "server" : "local");
  }, [settings, serverAvailable]);

  // Fallback: if stored id not in list, pick first available
  useEffect(() => {
    if (models?.length && (!modelId || !models.find(m => m.id === modelId))) {
      setModelId((models.find(m => m.online !== false) ?? models[0]).id);
    }
  }, [models]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (tasks && (!taskId || !tasks.find(d => d.id === taskId))) {
      const first = tasks.find(d => d.prepared);
      if (first) setTaskId(first.id);
    }
  }, [tasks]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load task config when settings panel is opened
  useEffect(() => {
    if (datasetSettingsOpen && taskId && !datasetConfig) {
      setTaskConfigLoading(true);
      api.getTaskConfig(taskId).then(cfg => {
        setTaskConfig(cfg);
        setTaskConfigLoading(false);
      }).catch(() => setTaskConfigLoading(false));
    }
  }, [datasetSettingsOpen, taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { setTaskConfig(null); }, [taskId]);

  // Per-mode params helpers
  function getModeParams(mode: LaunchRunMode): ModeParams {
    return modeParams[mode] ?? { ...DEFAULT_MODE_PARAMS };
  }
  function setModeParam<K extends keyof ModeParams>(mode: LaunchRunMode, key: K, value: ModeParams[K]) {
    setModeParams(prev => ({ ...prev, [mode]: { ...getModeParams(mode), [key]: value } }));
  }

  const model = models?.find(m => m.id === modelId);
  const task = tasks?.find(d => d.id === taskId);
  const selectedCount = selectedModes.length;
  const primaryMode = selectedModes[0] ?? "gym";
  const multiMode = selectedCount > 1;
  const stepBased = selectedModes.some(m => m === "iterative" || m === "gym" || m === "fixed");
  const repeatedLike = selectedModes.includes("repeated");
  const checklistMode = selectedModes.includes("gym");
  const thoughtsSupported = selectedModes.some(m => m === "gym" || m === "iterative" || m === "fixed");
  const canLaunch = selectedCount > 0 && !!modelId && !!task?.prepared && !launching;

  function toggleMode(id: LaunchRunMode) {
    setSelectedModes(prev => {
      if (prev.includes(id)) {
        if (prev.length === 1) return prev;
        setExpandedModes(s => { const n = new Set(s); n.delete(id); return n; });
        return prev.filter(m => m !== id);
      }
      if (prev.length >= MAX_SELECTED_MODES) return prev;
      setExpandedModes(s => new Set([...s, id]));
      return [...prev, id];
    });
  }

  function toggleExpand(id: LaunchRunMode) {
    setExpandedModes(prev => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  }

  async function saveTaskConfig() {
    if (!datasetConfig || !taskId) return;
    setTaskConfigSaving(true);
    try {
      await api.updateTaskConfig(taskId, {
        agent_notes: datasetConfig.agent_notes,
        task: datasetConfig.task,
      });
    } finally { setTaskConfigSaving(false); }
  }

  async function launch() {
    if (!canLaunch) return;
    setLaunching(true);
    setError(null);
    const p = getModeParams(primaryMode);
    try {
      const run = await api.launchRun({
        modelId,
        mode: multiMode ? "batch" : primaryMode,
        modes: selectedModes,
        taskId,
        maxSteps: stepBased ? p.maxSteps : undefined,
        maxTokens, temp: p.temp, seed,
        shots: repeatedLike ? getModeParams("repeated").shots : undefined,
        execution,
        enableThoughts: thoughtsSupported ? p.enableThoughts : undefined,
        hintCooldown: checklistMode ? getModeParams("gym").hintCooldown : undefined,
      });
      nav(multiMode ? "/runs" : `/runs/${run.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLaunching(false);
    }
  }

  return (
    <div className="newrun">
      <div className="steps">

        {/* 0. Execution */}
        <Card>
          <div className="step-head"><span className="step-num">⚙</span><span className="step-title">Среда выполнения</span></div>
          <div className="grid-2">
            <div className={`pick${execution === "local" ? " pick-active" : ""}`} onClick={() => setExecution("local")}>
              <div className="pick-title">На компьютере</div>
              <div className="pick-desc">gym считается локально, LLM — на сервере. Работает с любого WiFi.</div>
              {execution === "local" && <span className="pick-check"><Icon name="check" size={18} /></span>}
            </div>
            <div className={`pick${execution === "server" ? " pick-active" : ""}${!serverAvailable ? " pick-disabled" : ""}`}
              onClick={() => serverAvailable && setExecution("server")}>
              <div className="pick-title">На сервере (SSH)</div>
              <div className="pick-desc">{serverAvailable ? "gym и обучение — на сервере, мак не нагружается." : "Не настроено — Настройки → «Выполнение на сервере (SSH)»."}</div>
              {execution === "server" && <span className="pick-check"><Icon name="check" size={18} /></span>}
            </div>
          </div>
        </Card>

        {/* 1. Model */}
        <Card>
          <div className="step-head"><span className="step-num">1</span><span className="step-title">Модель</span></div>
          {model ? (
            <>
              <div className="picker-selected">
                <div className="picker-selected-info">
                  <Dot tone={model.online === false ? "red" : model.online ? "green" : "gray"} />
                  <span className="mono" style={{ fontWeight: 500, flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>
                  <span className="tag">{model.provider}</span>
                  <span className="tag mono">{(model.ctx / 1024).toFixed(0)}k ctx</span>
                </div>
                <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                  <Button variant="ghost" onClick={() => setModelPickerOpen(true)}>Изменить</Button>
                  <button className={`icon-btn${modelSettingsOpen ? " icon-btn-active" : ""}`}
                    title="Настройки модели" onClick={() => setModelSettingsOpen(v => !v)}>
                    <Icon name="settings" size={16} />
                  </button>
                </div>
              </div>
              {modelSettingsOpen && (
                <div className="picker-settings">
                  <FI label="Output limit" info="Максимум токенов в одном ответе модели. Если модель упирается в лимит — ответ обрезается и прогон завершается с ошибкой. Напр. 8192 для большинства моделей.">
                    <Stepper value={maxTokens} onChange={setMaxTokens} min={256} max={131072} step={256} />
                  </FI>
                </div>
              )}
            </>
          ) : (
            <Button variant="secondary" onClick={() => setModelPickerOpen(true)}>
              <Icon name="plus" size={15} /> Выбрать модель
            </Button>
          )}
          {models && !models.length && <div className="muted" style={{ marginTop: 8 }}>Нет моделей. Добавьте на экране «Модели».</div>}
        </Card>

        {/* 2. Mode */}
        <Card>
          <div className="step-head step-head-spread">
            <span className="step-head-left"><span className="step-num">2</span><span className="step-title">Тип прогона</span></span>
            <span className="step-hint">Можно выбрать до {MAX_SELECTED_MODES}</span>
          </div>
          <div className="mode-list">
            {MODE_INFO.map(mi => {
              const active = selectedModes.includes(mi.id);
              const expanded = active && expandedModes.has(mi.id);
              const disabled = !active && selectedCount >= MAX_SELECTED_MODES;
              const p = getModeParams(mi.id);
              return (
                <div key={mi.id} className={`mode-row${active ? " mode-row-active" : ""}${disabled ? " mode-row-disabled" : ""}`}>
                  <div className="mode-row-head" onClick={() => !disabled && toggleMode(mi.id)}>
                    <div className={`mode-row-check${active ? " mode-row-check-active" : ""}`}>
                      {active && <Icon name="check" size={12} />}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="mode-row-title">
                        {MODE_LABELS[mi.id]}
                        <span className={`tag${mi.env ? " tag-accent" : ""}`}>{mi.env ? "Среда" : "Без среды"}</span>
                      </div>
                      <div className="mode-row-desc">{mi.desc}</div>
                    </div>
                    {active && (
                      <button className="icon-btn" style={{ flexShrink: 0 }}
                        onClick={e => { e.stopPropagation(); toggleExpand(mi.id); }}>
                        <Icon name="chevronDown" size={15} style={{ transform: expanded ? "rotate(180deg)" : undefined, transition: "transform 0.15s" }} />
                      </button>
                    )}
                  </div>
                  {expanded && (
                    <div className="mode-row-params">
                      {mi.hasSteps && (
                        <FI label="Макс. шагов" info="Сколько интерактивных шагов доступно режиму до финального submit.">
                          <Stepper value={p.maxSteps} onChange={v => setModeParam(mi.id, "maxSteps", v)} min={1} max={200} />
                        </FI>
                      )}
                      {mi.hasShots && (
                        <FI label="Число попыток" info="Количество независимых попыток. Каждая попытка независима, между ними только val-метрика.">
                          <Stepper value={p.shots} onChange={v => setModeParam(mi.id, "shots", v)} min={2} max={50} />
                        </FI>
                      )}
                      {mi.hasHint && (
                        <FI label="Подсказка каждые N шагов" info="Через сколько шагов агенту даётся следующая подсказка чеклиста DS-пайплайна. 1 — на каждом шаге, 3 — каждые три шага.">
                          <Stepper value={p.hintCooldown} onChange={v => setModeParam(mi.id, "hintCooldown", v)} min={1} max={20} />
                        </FI>
                      )}
                      <FI label={`Температура: ${p.temp.toFixed(2)}`} info="Случайность ответов: 0 — всегда одинаково, 1 — очень вариативно. Рекомендуется 0.3–0.6 для кода.">
                        <div className="stepper-height-wrap">
                          <input className="range" type="range" min={0} max={1} step={0.05} value={p.temp}
                            onChange={e => setModeParam(mi.id, "temp", parseFloat(e.target.value))} />
                        </div>
                      </FI>
                      {mi.hasThoughts && (
                        <FI label="Мысли LLM" info="Агент ведёт внутренние заметки между шагами. Улучшает качество решений за счёт рассуждений, но заметно увеличивает расход токенов.">
                          <div className="wide-toggle" onClick={() => setModeParam(mi.id, "enableThoughts", !p.enableThoughts)}>
                            <div className={`wide-toggle-thumb${p.enableThoughts ? " on" : ""}`} />
                            <span className="wide-toggle-off">Выкл</span>
                            <span className="wide-toggle-on">Вкл</span>
                          </div>
                        </FI>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>

        {/* 3. Task */}
        <Card>
          <div className="step-head"><span className="step-num">3</span><span className="step-title">Задача</span></div>
          {task ? (
            <>
              <div className="ds-picker-card">
                <div className="spread" style={{ marginBottom: 6 }}>
                  <div className="ds-title" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{task.name}</div>
                  <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                    <Button variant="ghost" onClick={() => setTaskPickerOpen(true)}>Изменить</Button>
                    <button className={`icon-btn${datasetSettingsOpen ? " icon-btn-active" : ""}`}
                      title="Настройки задачи" onClick={() => setDatasetSettingsOpen(v => !v)}>
                      <Icon name="settings" size={16} />
                    </button>
                  </div>
                </div>
                {task.desc && <div className="muted clamp-2" style={{ fontSize: 12.5, marginBottom: 8 }}>{task.desc}</div>}
                <div className="run-meta-line" style={{ margin: "0 0 10px" }}>
                  <Tag tone={task.taskType === "regression" ? "blue" : task.taskType === "classification" ? "accent" : "neutral"}>
                    {task.taskType ?? task.task}
                  </Tag>
                  <Tag mono>{task.metricGoal === "min" ? "minimize" : "maximize"}</Tag>
                  {(task.tags ?? []).slice(0, 3).map(t => <Tag key={t} tone="neutral">{t}</Tag>)}
                  {!task.prepared && <Tag tone="red">не подготовлен</Tag>}
                </div>
                <div className="ds-stats rich" style={{ fontSize: 12 }}>
                  <div className="ds-stat"><span className="k">rows</span><span className="v">{task.rows ? task.rows.toLocaleString() : "—"}</span></div>
                  <div className="ds-stat"><span className="k">features</span><span className="v">{task.cols || "—"}</span></div>
                  <div className="ds-stat"><span className="k">target</span><span className="v">{task.target}</span></div>
                  <div className="ds-stat"><span className="k">metric</span><span className="v">{task.metric}</span></div>
                </div>
              </div>
              {datasetSettingsOpen && (
                <div className="picker-settings">
                  {datasetConfigLoading ? (
                    <div style={{ display: "flex", justifyContent: "center", padding: 24 }}><Spinner /></div>
                  ) : datasetConfig ? (
                    <div className="stack" style={{ gap: 8 }}>
                      {/* Accordion: Конфиг */}
                      <div className="ds-accordion-item">
                        <button className="ds-accordion-head" onClick={() => setDsSection(s => s === "config" ? null : "config")}>
                          <Icon name="settings" size={14} />
                          <span>Конфиг</span>
                          <Icon name="chevronDown" size={14} style={{ marginLeft: "auto", transform: dsSection === "config" ? "rotate(180deg)" : undefined, transition: "transform .15s" }} />
                        </button>
                        {dsSection === "config" && (
                          <div className="ds-accordion-body">
                            <div className="grid-3" style={{ gap: 14 }}>
                              <FI label="Тип задачи" info="classification — предсказание категорий. regression — предсказание числа. auto — определится автоматически по данным.">
                                <div className="sort-tabs">
                                  {TASK_TYPES.map(tt => (
                                    <button key={tt.value}
                                      className={`sort-tab${datasetConfig.task.task_type === tt.value ? " active" : ""}`}
                                      onClick={() => setTaskConfig(c => c ? { ...c, task: { ...c.task, task_type: tt.value } } : c)}>
                                      {tt.label}
                                    </button>
                                  ))}
                                </div>
                              </FI>
                              <FI label={<>Метрика <a className="docs-link" href="https://scikit-learn.org/stable/modules/model_evaluation.html" target="_blank" rel="noopener noreferrer">sklearn ↗</a></>} info="Название метрики sklearn, считается на test после submit. Агент видит только val. Напр. f1_macro, accuracy, neg_rmse, roc_auc.">
                                <input className="input mono" value={datasetConfig.task.metric_name}
                                  onChange={e => setTaskConfig(c => c ? { ...c, task: { ...c.task, metric_name: e.target.value } } : c)} />
                              </FI>
                              <FI label="Target column" info="Колонка с целевой переменной — то, что агент должен предсказывать. Эта колонка никогда не включается в признаки.">
                                <input className="input mono" value={datasetConfig.task.target_col}
                                  onChange={e => setTaskConfig(c => c ? { ...c, task: { ...c.task, target_col: e.target.value } } : c)} />
                              </FI>
                            </div>
                            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
                              <Button variant="primary" onClick={saveTaskConfig} disabled={datasetConfigSaving}>
                                {datasetConfigSaving ? <Spinner /> : "Сохранить"}
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>
                      {/* Accordion: Комментарии */}
                      <div className="ds-accordion-item">
                        <button className="ds-accordion-head" onClick={() => setDsSection(s => s === "notes" ? null : "notes")}>
                          <Icon name="edit" size={14} />
                          <span>Комментарии для LLM</span>
                          <Icon name="chevronDown" size={14} style={{ marginLeft: "auto", transform: dsSection === "notes" ? "rotate(180deg)" : undefined, transition: "transform .15s" }} />
                        </button>
                        {dsSection === "notes" && (
                          <div className="ds-accordion-body">
                            <div className="stack" style={{ gap: 12 }}>
                              <FI label="Описание задачи" info="Текст, который агент получит как описание задачи в начале прогона. Объясните что нужно предсказать и почему.">
                                <textarea className="input" rows={3} style={{ resize: "vertical" }}
                                  value={datasetConfig.agent_notes.task_description}
                                  onChange={e => setTaskConfig(c => c ? { ...c, agent_notes: { ...c.agent_notes, task_description: e.target.value } } : c)} />
                              </FI>
                              <FI label="Дополнительные комментарии" info="Дополнительные подсказки агенту: особенности данных, известные ограничения, запреты. Агент видит это в каждом шаге.">
                                <textarea className="input" rows={2} style={{ resize: "vertical" }}
                                  value={datasetConfig.agent_notes.additional_comments}
                                  onChange={e => setTaskConfig(c => c ? { ...c, agent_notes: { ...c.agent_notes, additional_comments: e.target.value } } : c)} />
                              </FI>
                              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                                <Button variant="primary" onClick={saveTaskConfig} disabled={datasetConfigSaving}>
                                  {datasetConfigSaving ? <Spinner /> : "Сохранить"}
                                </Button>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </>
          ) : (
            <Button variant="secondary" onClick={() => setTaskPickerOpen(true)}>
              <Icon name="plus" size={15} /> Выбрать задачу
            </Button>
          )}
        </Card>

        {/* 4. Seed */}
        <Card>
          <div className="step-head"><span className="step-num">4</span><span className="step-title">Параметры</span></div>
          <div style={{ maxWidth: 220 }}>
            <FI label="Seed" info="Фиксирует случайность запуска, чтобы результаты было проще воспроизводить и сравнивать.">
              <Stepper value={seed} onChange={setSeed} min={0} max={999999} />
            </FI>
          </div>
        </Card>

      </div>

      {/* Preview sidebar */}
      <div className="preview">
        <h3>Превью конфигурации</h3>
        <div className="preview-row"><span className="k">Среда</span><span className={`v${execution === "server" ? " acc" : ""}`}>{execution === "server" ? "на сервере" : "на компьютере"}</span></div>
        <div className="preview-row"><span className="k">Модель</span><span className="v mono" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{model?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Провайдер</span><span className="v">{model?.provider ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Режим</span><span className={`v${selectedModes.includes("gym") ? " acc" : ""}`}>{multiMode ? `${selectedCount} режима` : MODE_LABELS[primaryMode]}</span></div>
        <div className="preview-row"><span className="k">Задача</span><span className="v">{task?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Задача</span><span className="v">{task?.task ?? "—"}</span></div>
        {multiMode && <div className="preview-row"><span className="k">Прогонов</span><span className="v acc">{selectedCount} отдельных</span></div>}
        {stepBased && <div className="preview-row"><span className="k">Шагов</span><span className="v">{getModeParams(primaryMode).maxSteps}</span></div>}
        {repeatedLike && <div className="preview-row"><span className="k">Попыток</span><span className="v">{getModeParams("repeated").shots}</span></div>}
        {checklistMode && <div className="preview-row"><span className="k">Подсказка каждые</span><span className="v">{getModeParams("gym").hintCooldown} шаг.</span></div>}
        <div className="preview-row"><span className="k">Токены / темп.</span><span className="v">{(maxTokens / 1024).toFixed(0)}k / {getModeParams(primaryMode).temp.toFixed(2)}</span></div>

        <div style={{ marginTop: 18 }}>
          <Button variant="primary" size="lg" block disabled={!canLaunch} onClick={launch}>
            {launching ? <Spinner /> : <Icon name="play" size={18} fill />}
            {launching ? "Запуск…" : multiMode ? `Запустить ${selectedCount} прогона` : "Запустить прогон"}
          </Button>
        </div>
        {!task && <div className="preview-est">Выберите задачу</div>}
        {task && !task.prepared && <div className="preview-est">Задача не подготовлена</div>}
        {error && <div className="preview-est" style={{ color: "var(--red)" }}>{error}</div>}
      </div>

      {modelPickerOpen && models && (
        <ModelPickerModal models={models} current={modelId} onSelect={setModelId} onClose={() => setModelPickerOpen(false)} />
      )}
      {taskPickerOpen && tasks && (
        <TaskPickerModal tasks={tasks} current={taskId} onSelect={setTaskId} onClose={() => setTaskPickerOpen(false)} />
      )}
    </div>
  );
}
