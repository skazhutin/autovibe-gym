import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Dataset, type ModelRec, type RunMode } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS } from "../lib/format";
import { Button, Card, Dot, Field, Spinner } from "../components/ui";
import { Icon } from "../components/Icon";

const MODE_INFO: { id: RunMode; desc: string; rec?: boolean }[] = [
  { id: "single", desc: "Один полный ответ без обратной связи от среды." },
  { id: "repeated", desc: "N независимых попыток, только скалярная val-метрика между ними." },
  { id: "iterative", desc: "Итеративно с runtime/contract-фидбэком, без подсказок чеклиста." },
  { id: "gym", desc: "Итеративно + неявные подсказки чеклиста DS-пайплайна.", rec: true },
];

const BUDGET_DEFAULTS = {
  local: { maxSteps: 30, maxTokens: 8192 },
  cloud: { maxSteps: 20, maxTokens: 4096 },
};

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
  const [mode, setMode] = useState<RunMode>("gym");
  const [datasetId, setDatasetId] = useState<string>("");
  const [budgetMode, setBudgetMode] = useState<"local" | "cloud">("local");
  const [maxSteps, setMaxSteps] = useState(30);
  const [maxTokens, setMaxTokens] = useState(8192);
  const [temp, setTemp] = useState(0.4);
  const [seed, setSeed] = useState(42);
  const [shots, setShots] = useState(5);
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
  const iterative = mode === "gym" || mode === "iterative";
  const canLaunch = !!modelId && !!dataset?.prepared && !launching;

  async function launch() {
    if (!canLaunch) return;
    setLaunching(true);
    setError(null);
    try {
      const run = await api.launchRun({
        modelId, mode, datasetId, budgetMode,
        maxSteps: iterative ? maxSteps : undefined,
        maxTokens, temp, seed,
        shots: mode === "repeated" ? shots : undefined,
        execution,
      });
      nav(`/runs/${run.id}`);
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
          <div className="step-head"><span className="step-num">2</span><span className="step-title">Тип прогона</span></div>
          <div className="grid-2">
            {MODE_INFO.map((mi) => (
              <div key={mi.id} className={`pick${mode === mi.id ? " pick-active" : ""}`} onClick={() => setMode(mi.id)}>
                <div className="pick-title">{MODE_LABELS[mi.id]}{mi.rec && <span className="rec-badge">РЕКОМЕНДУЕМ</span>}</div>
                <div className="pick-desc">{mi.desc}</div>
                {mode === mi.id && <span className="pick-check"><Icon name="check" size={18} /></span>}
              </div>
            ))}
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
            <Field label="Пресет бюджета" hint="local — длиннее, cloud — экономнее">
              <select className="input" value={budgetMode} onChange={(e) => setBudgetMode(e.target.value as "local" | "cloud")}>
                <option value="local">local</option>
                <option value="cloud">cloud</option>
              </select>
            </Field>
            <Field label="Лимит токенов">
              <Stepper value={maxTokens} onChange={setMaxTokens} min={256} max={131072} step={256} />
            </Field>
            {iterative && (
              <Field label="Макс. шагов">
                <Stepper value={maxSteps} onChange={setMaxSteps} min={1} max={200} />
              </Field>
            )}
            {mode === "repeated" && (
              <Field label="Число попыток (shots)">
                <Stepper value={shots} onChange={setShots} min={2} max={50} />
              </Field>
            )}
            <Field label={`Температура: ${temp.toFixed(2)}`}>
              <input className="range" type="range" min={0} max={1} step={0.05} value={temp} onChange={(e) => setTemp(parseFloat(e.target.value))} />
            </Field>
            <Field label="Seed">
              <Stepper value={seed} onChange={setSeed} min={0} max={999999} />
            </Field>
          </div>
        </Card>
      </div>

      {/* Preview */}
      <div className="preview">
        <h3>Превью конфигурации</h3>
        <div className="preview-row"><span className="k">Среда</span><span className={`v${execution === "server" ? " acc" : ""}`}>{execution === "server" ? "на сервере" : "на компьютере"}</span></div>
        <div className="preview-row"><span className="k">Модель</span><span className="v">{model?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Провайдер</span><span className="v">{model?.provider ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Режим</span><span className={`v${mode === "gym" ? " acc" : ""}`}>{MODE_LABELS[mode]}</span></div>
        <div className="preview-row"><span className="k">Датасет</span><span className="v">{dataset?.name ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Задача</span><span className="v">{dataset?.task ?? "—"}</span></div>
        <div className="preview-row"><span className="k">Бюджет</span><span className="v">{budgetMode}</span></div>
        {iterative && <div className="preview-row"><span className="k">Шагов</span><span className="v">{maxSteps}</span></div>}
        {mode === "repeated" && <div className="preview-row"><span className="k">Попыток</span><span className="v">{shots}</span></div>}
        <div className="preview-row"><span className="k">Токены / темп.</span><span className="v">{(maxTokens / 1024).toFixed(0)}k / {temp.toFixed(2)}</span></div>

        <div style={{ marginTop: 18 }}>
          <Button variant="primary" size="lg" block disabled={!canLaunch} onClick={launch}>
            {launching ? <Spinner /> : <Icon name="play" size={18} fill />}
            {launching ? "Запуск…" : "Запустить прогон"}
          </Button>
        </div>
        {!dataset?.prepared && <div className="preview-est">Выберите подготовленный датасет</div>}
        {error && <div className="preview-est" style={{ color: "var(--red)" }}>{error}</div>}
      </div>
    </div>
  );
}
