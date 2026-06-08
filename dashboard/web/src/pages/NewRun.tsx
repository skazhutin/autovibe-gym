import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { api, type Dataset, type LaunchRunMode, type ModelRec } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS } from "../lib/format";
import { Button, Card, Dot, Field, Spinner } from "../components/ui";
import { Icon } from "../components/Icon";

const MAX_SELECTED_MODES = 5;

const MODE_INFO: { id: LaunchRunMode; desc: string; env: boolean }[] = [
  { id: "single", env: false, desc: "Один полный ответ без обратной связи от среды." },
  { id: "repeated", env: false, desc: "N независимых попыток, только скалярная val-метрика между ними." },
  { id: "iterative", env: true, desc: "Итеративная среда без подсказок чеклиста: runtime и contract feedback." },
  { id: "gym", env: true, desc: "Гибкая интерактивная среда с неявными подсказками чеклиста DS-пайплайна." },
  { id: "fixed", env: true, desc: "Фиксированные стадии gym: EDA, preprocessing, feature engineering, model selection, tuning." },
];

const BUDGET_DEFAULTS = {
  local: { maxSteps: 30, maxTokens: 8192 },
  cloud: { maxSteps: 20, maxTokens: 4096 },
};

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
      <span ref={dotRef} className="info-dot" aria-label={text} onMouseEnter={show} onMouseLeave={() => setVisible(false)}>?</span>
      {visible && createPortal(<div className="tooltip-portal" style={{ top: pos.top, left: pos.left }}>{text}</div>, document.body)}
    </>
  );
}

function FieldInfo({ label, info, children }: { label: ReactNode; info: string; children: ReactNode }) {
  return (
    <Field label={<span className="field-info-label">{label}<Info text={info} /></span>}>
      {children}
    </Field>
  );
}

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

export default function NewRun() {
  const nav = useNavigate();
  const { data: models } = useAsync(() => api.listModels(), []);
  const { data: datasets } = useAsync(() => api.listDatasets(), []);
  const { data: settings } = useAsync(() => api.getSettings(), []);
  const serverAvailable = !!(settings?.remote_ssh && settings?.remote_repo);
  const [execution, setExecution] = useState<"local" | "server">("local");

  const [modelId, setModelId] = useState<string>("");
  const [selectedModes, setSelectedModes] = useState<LaunchRunMode[]>(["gym"]);
  const [datasetId, setDatasetId] = useState<string>("");
  const [budgetMode, setBudgetMode] = useState<"local" | "cloud">("local");
  const [maxSteps, setMaxSteps] = useState(30);
  const [maxTokens, setMaxTokens] = useState(8192);
  const [temp, setTemp] = useState(0.4);
  const [seed, setSeed] = useState(42);
  const [shots, setShots] = useState(5);
  const [enableThoughts, setEnableThoughts] = useState(false);
  const [hintCooldown, setHintCooldown] = useState(2);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (settings) setExecution(settings.remote_enabled && serverAvailable ? "server" : "local");
  }, [settings, serverAvailable]);

  useEffect(() => {
    setMaxSteps(BUDGET_DEFAULTS[budgetMode].maxSteps);
    setMaxTokens(BUDGET_DEFAULTS[budgetMode].maxTokens);
  }, [budgetMode]);

  useEffect(() => {
    if (!modelId && models?.length) setModelId(models[0].id);
    if (!datasetId && datasets) {
      const first = datasets.find((d) => d.prepared);
      if (first) setDatasetId(first.id);
    }
  }, [models, datasets, modelId, datasetId]);

  const model: ModelRec | undefined = models?.find((m) => m.id === modelId);
  const dataset: Dataset | undefined = datasets?.find((d) => d.id === datasetId);
  const selectedCount = selectedModes.length;
  const primaryMode = selectedModes[0] ?? "gym";
  const multiMode = selectedCount > 1;
  const stepBased = selectedModes.some((m) => m === "iterative" || m === "gym" || m === "fixed");
  const repeatedLike = selectedModes.includes("repeated");
  const thoughtsSupported = selectedModes.some((m) => m === "gym" || m === "iterative");
  const checklistMode = selectedModes.includes("gym");
  const canLaunch = selectedCount > 0 && !!modelId && !!dataset?.prepared && !launching;

  function toggleMode(id: LaunchRunMode) {
    setSelectedModes((prev) => {
      if (prev.includes(id)) {
        return prev.length === 1 ? prev : prev.filter((m) => m !== id);
      }
      if (prev.length >= MAX_SELECTED_MODES) return prev;
      return [...prev, id];
    });
  }

  function launchLabel() {
    if (!multiMode) return "Запустить прогон";
    return `Запустить ${selectedCount} ${selectedCount === 1 ? "прогон" : "прогона"}`;
  }

  async function launch() {
    if (!canLaunch) return;
    setLaunching(true);
    setError(null);
    try {
      const run = await api.launchRun({
        modelId,
        mode: multiMode ? "batch" : primaryMode,
        modes: selectedModes,
        datasetId, budgetMode,
        maxSteps: stepBased ? maxSteps : undefined,
        maxTokens, temp, seed,
        shots: repeatedLike ? shots : undefined,
        execution,
        enableThoughts: thoughtsSupported ? enableThoughts : undefined,
        hintCooldown: checklistMode ? hintCooldown : undefined,
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
        {/* 0. Execution location */}
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
          <div className="grid-2">
            {(models ?? []).map((m) => (
              <div key={m.id} className={`pick${modelId === m.id ? " pick-active" : ""}${m.online === false ? " pick-disabled" : ""}`}
                onClick={() => m.online !== false && setModelId(m.id)}>
                <div className="pick-title"><Dot tone={m.online === false ? "gray" : m.online ? "green" : "gray"} />{m.name}</div>
                <div className="pick-chips">
                  <span className="tag">{m.provider}</span>
                  <span className="tag mono">{(m.ctx / 1024).toFixed(0)}k ctx</span>
                </div>
                {modelId === m.id && <span className="pick-check"><Icon name="check" size={18} /></span>}
              </div>
            ))}
            {models && !models.length && <div className="muted">Нет моделей. Добавьте на экране «Модели».</div>}
          </div>
        </Card>

        {/* 2. Mode */}
        <Card>
          <div className="step-head step-head-spread">
            <span className="step-head-left"><span className="step-num">2</span><span className="step-title">Тип прогона</span></span>
            <span className="step-hint">Можно выбрать до 5</span>
          </div>
          <div className="grid-2">
            {MODE_INFO.map((mi) => {
              const active = selectedModes.includes(mi.id);
              const disabled = !active && selectedCount >= MAX_SELECTED_MODES;
              return (
              <div key={mi.id} className={`pick${active ? " pick-active" : ""}${disabled ? " pick-disabled" : ""}`} onClick={() => !disabled && toggleMode(mi.id)}>
                <div className="pick-title">
                  {MODE_LABELS[mi.id]}
                  <span className={`tag${mi.env ? " tag-accent" : ""}`}>{mi.env ? "Среда" : "Без среды"}</span>
                </div>
                <div className="pick-desc">{mi.desc}</div>
                {active && <span className="pick-check"><Icon name="check" size={18} /></span>}
              </div>
              );
            })}
          </div>
        </Card>

        {/* 3. Dataset */}
        <Card>
          <div className="step-head"><span className="step-num">3</span><span className="step-title">Датасет</span></div>
          <div className="grid-3">
            {(datasets ?? []).map((d) => (
              <div key={d.id} className={`pick${datasetId === d.id ? " pick-active" : ""}${!d.prepared ? " pick-disabled" : ""}`}
                onClick={() => d.prepared && setDatasetId(d.id)}>
                <div className="pick-title" style={{ fontSize: 13 }}>{d.name}</div>
                <div className="pick-chips">
                  <span className="tag">{d.task}</span>
                  {d.prepared ? <span className="tag mono">{d.rows.toLocaleString()}×{d.cols}</span> : <span className="tag tag-red">не подготовлен</span>}
                </div>
                {d.prepared && <div className="pick-desc">метрика: {d.metric}</div>}
                {datasetId === d.id && <span className="pick-check"><Icon name="check" size={18} /></span>}
              </div>
            ))}
          </div>
        </Card>

        {/* 4. Budget */}
        <Card>
          <div className="step-head"><span className="step-num">4</span><span className="step-title">Параметры бюджета</span></div>
          <div className="grid-2" style={{ gap: 18 }}>
            <FieldInfo label="Пресет бюджета" info="Выбирает стартовые значения для шагов и токенов: local даёт больше бюджета, cloud — более экономный пресет.">
              <select className="input" value={budgetMode} onChange={(e) => setBudgetMode(e.target.value as "local" | "cloud")}>
                <option value="local">local</option>
                <option value="cloud">cloud</option>
              </select>
            </FieldInfo>
            <FieldInfo label="Лимит токенов" info="Максимум токенов ответа модели на один запрос. Больше токенов даёт больше места для решения, но повышает стоимость и время.">
              <Stepper value={maxTokens} onChange={setMaxTokens} min={256} max={131072} step={256} />
            </FieldInfo>
            {stepBased && (
              <FieldInfo label="Макс. шагов" info="Сколько интерактивных шагов доступно режимам со средой до финального submit.">
                <Stepper value={maxSteps} onChange={setMaxSteps} min={1} max={200} />
              </FieldInfo>
            )}
            {checklistMode && (
              <FieldInfo label="Подсказка каждые N шагов" info="Через сколько шагов агенту даётся новая подсказка чеклиста (Gym). 1 — на каждом шаге, 2 — через шаг, и т.д.">
                <Stepper value={hintCooldown} onChange={setHintCooldown} min={1} max={20} />
              </FieldInfo>
            )}
            {repeatedLike && (
              <FieldInfo label="Число попыток (shots)" info="Количество независимых single-shot попыток в repeated-режиме.">
                <Stepper value={shots} onChange={setShots} min={2} max={50} />
              </FieldInfo>
            )}
            <FieldInfo label={`Температура: ${temp.toFixed(2)}`} info="Насколько разнообразными будут ответы модели: ниже — стабильнее, выше — больше вариативности.">
              <input className="range" type="range" min={0} max={1} step={0.05} value={temp} onChange={(e) => setTemp(parseFloat(e.target.value))} />
            </FieldInfo>
            <FieldInfo label="Seed" info="Фиксирует случайность запуска, чтобы результаты было проще воспроизводить и сравнивать.">
              <Stepper value={seed} onChange={setSeed} min={0} max={999999} />
            </FieldInfo>
          </div>
          {thoughtsSupported && (
            <div className="spread" style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
              <div>
                <div className="field-label">Мысли LLM (scratchpad)</div>
                <div className="field-hint">Агент ведёт заметки/рассуждения, которые сохраняются и возвращаются ему между шагами. Доступно для Gym и Iterative.</div>
              </div>
              <div className={`toggle${enableThoughts ? " on" : ""}`} onClick={() => setEnableThoughts((v) => !v)} />
            </div>
          )}
        </Card>
      </div>

      {/* Preview */}
      <div className="preview">
        <h3>Превью конфигурации</h3>
        <div className="preview-row"><span className="k">Среда</span><span className={`v${execution === "server" ? " acc" : ""}`}>{execution === "server" ? "на сервере" : "на компьютере"}</span></div>
        <div className="preview-row"><span className="k">Модель</span><span className="v">{model?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Провайдер</span><span className="v">{model?.provider ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Режим</span><span className={`v${selectedModes.includes("gym") ? " acc" : ""}`}>{multiMode ? `${selectedCount} режима` : MODE_LABELS[primaryMode]}</span></div>
        <div className="preview-row"><span className="k">Датасет</span><span className="v">{dataset?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Задача</span><span className="v">{dataset?.task ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Бюджет</span><span className="v">{budgetMode}</span></div>
        {multiMode && <div className="preview-row"><span className="k">Прогонов</span><span className="v acc">{selectedCount} отдельных</span></div>}
        {stepBased && <div className="preview-row"><span className="k">Шагов</span><span className="v">{maxSteps}</span></div>}
        {repeatedLike && <div className="preview-row"><span className="k">Попыток</span><span className="v">{shots}</span></div>}
        {checklistMode && <div className="preview-row"><span className="k">Подсказка каждые</span><span className="v">{hintCooldown} шаг.</span></div>}
        <div className="preview-row"><span className="k">Токены / темп.</span><span className="v">{(maxTokens / 1024).toFixed(0)}k / {temp.toFixed(2)}</span></div>

        <div style={{ marginTop: 18 }}>
          <Button variant="primary" size="lg" block disabled={!canLaunch} onClick={launch}>
            {launching ? <Spinner /> : <Icon name="play" size={18} fill />}
            {launching ? "Запуск…" : launchLabel()}
          </Button>
        </div>
        {!dataset?.prepared && <div className="preview-est">Выберите подготовленный датасет</div>}
        {error && <div className="preview-est" style={{ color: "var(--red)" }}>{error}</div>}
      </div>
    </div>
  );
}
