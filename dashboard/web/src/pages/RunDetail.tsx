import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS, formatScore, formatTokens, improvementPct } from "../lib/format";
import { Button, Card, EmptyState, LiveDuration, ProgressBar, ProgressRing, Skeleton, Spinner, StatusBadge, Tabs, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { CodeBlock } from "../components/CodeBlock";
import { Donut } from "../components/charts";

const STEP_ICON: Record<string, string> = {
  add_cell: "plus", update_cell: "edit", edit_cell: "edit", delete_cell: "trash",
  run_cell: "play", restart_and_run_all: "refresh", inspect_notebook: "notebook",
  validate: "check", submit: "check2", think: "sparkles",
};

const TABS = [
  { id: "notebook", label: "Ноутбук", icon: "notebook" },
  { id: "trajectory", label: "Траектория", icon: "route" },
  { id: "thoughts", label: "Мысли", icon: "sparkles" },
  { id: "checklist", label: "Чеклист", icon: "check2" },
  { id: "errors", label: "Ошибки", icon: "bug" },
  { id: "logs", label: "Логи", icon: "terminal" },
];

const ACTION_TYPE_LABEL: Record<string, string> = {
  code: "код", add_cell: "ячейка", update_cell: "правка", delete_cell: "удаление",
  run_cell: "запуск", restart_and_run_all: "перезапуск", validate: "валидация",
  submit: "сабмит", finalize: "финал", inspect_notebook: "осмотр", think: "мысль",
};

const STAGE_LABELS: Record<string, string> = {
  planning: "Планирование",
  data_schema_inspection: "Структура данных",
  target_metric_inspection: "Target и метрика",
  data_quality_inspection: "Качество данных",
  leakage_split_inspection: "Утечки и split",
  preprocessing_design: "Проектирование preprocessing",
  feature_pipeline_building: "Сборка pipeline",
  baseline_modeling: "Baseline-модель",
  candidate_training: "Обучение candidate",
  validation_analysis: "Анализ валидации",
  model_improvement: "Улучшение модели",
  reproducibility_check: "Проверка воспроизводимости",
  submission: "Сабмит",
  unknown: "—",
};

function stageLabel(stage?: string | null) {
  return STAGE_LABELS[stage || "unknown"] ?? STAGE_LABELS.unknown;
}

function actionTone(type: string) {
  if (type === "submit") return "green";
  if (type === "validate" || type === "quick_validate" || type === "check_candidate") return "blue";
  if (type === "think") return "accent";
  return "neutral";
}

/** Render a small subset of markdown (**bold** + "- " bullet lists) used by the
 *  model's self-summary. We don't pull in a markdown dependency for this. */
function renderInline(text: string, keyBase: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={`${keyBase}-${i}`}>{part}</span>
    )
  );
}

function MarkdownLite({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {bullets.map((b, i) => (
            <li key={i}>{renderInline(b, `li-${blocks.length}-${i}`)}</li>
          ))}
        </ul>
      );
      bullets = [];
    }
  };
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (/^[-*]\s+/.test(line)) {
      bullets.push(line.replace(/^[-*]\s+/, ""));
    } else if (line) {
      flush();
      blocks.push(<p key={`p-${i}`}>{renderInline(line, `p-${i}`)}</p>);
    } else {
      flush();
    }
  });
  flush();
  return <>{blocks}</>;
}

function SummaryCard({ text, model }: { text: string; model?: string | null }) {
  return (
    <div className="run-summary">
      <div className="run-summary-head">
        <Icon name="sparkles" size={15} />
        <span className="run-summary-title">Саммари решения</span>
        {model && <span className="run-summary-model mono">{model}</span>}
      </div>
      <div className="run-summary-body">
        <MarkdownLite text={text} />
      </div>
    </div>
  );
}

function ThoughtsTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.thoughts(id), [id], live ? 2500 : 0);
  const { data: summary } = useAsync(() => api.runSummary(id), [id], live ? 2500 : 0);
  const thoughtItems = data ?? [];
  const summaryText = summary?.summary?.trim();
  if (loading && !data && !summaryText) return <Skeleton h={200} />;
  if (!thoughtItems.length && !summaryText)
    return (
      <EmptyState
        icon="sparkles"
        title="Мыслей нет"
        text="Саммари решения появится здесь после завершения прогона. Пошаговые мысли сохраняются только при включённом флаге «Мысли LLM» (Gym и Iterative)."
      />
    );
  return (
    <div className="thoughts">
      {summaryText && <SummaryCard text={summaryText} model={summary?.model} />}
      {thoughtItems.map((n, i) => (
        <div key={i} className="thought">
          <div className="thought-rail">
            <span className="thought-dot" />
          </div>
          <div className="thought-body">
            <div className="thought-head">
              <span className="st mono">шаг {n.step}</span>
              <span className="tag">{ACTION_TYPE_LABEL[n.type] ?? n.type}</span>
              <span className="tag">{stageLabel(n.stage)}</span>
            </div>
            <div className="thought-text">{n.thoughts}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChipMetric({ label, value, ring }: { label: string; value: React.ReactNode; ring?: React.ReactNode }) {
  return (
    <div className="chip-metric">
      {ring}
      <div>
        <div className="cm-label">{label}</div>
        <div className="cm-val">{value}</div>
      </div>
    </div>
  );
}

/* ---- tabs ---- */
function NotebookTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.notebook(id), [id], live ? 2500 : 0);
  if (loading && !data) return <Skeleton h={300} />;
  const cells = data?.cells ?? [];
  if (!cells.length) return <EmptyState icon="notebook" title="Ноутбук пуст" text="Решение ещё не сформировано или артефакт недоступен." />;
  return (
    <div>
      {cells.map((c, i) =>
        c.type === "markdown" ? (
          <div key={i} className="nb-md">{c.text}</div>
        ) : (
          <div key={i} className="nb-cell">
            <div className="nb-cell-head">In [{c.n ?? " "}]</div>
            <CodeBlock code={c.code ?? ""} />
            {(c.outputs ?? []).map((o, oi) =>
              o.type === "table" && o.html ? (
                <div key={oi} className="nb-out table" dangerouslySetInnerHTML={{ __html: o.html }} />
              ) : o.type === "error" ? (
                <div key={oi} className="nb-out error"><pre>{o.ename ? o.ename + "\n" : ""}{o.text}</pre></div>
              ) : (
                <div key={oi} className="nb-out"><pre>{o.text}</pre></div>
              )
            )}
          </div>
        )
      )}
      {live && <div className="spinner-row"><Spinner /> агент выполняет шаг…</div>}
    </div>
  );
}

function TrajectoryTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.trajectory(id), [id], live ? 2500 : 0);
  if (loading && !data) return <Skeleton h={300} />;
  const steps = data ?? [];
  if (!steps.length) return <EmptyState icon="route" title="Нет траектории" text="Шаги агента появятся здесь." />;
  return (
    <div className="traj">
      {steps.map((s, i) => (
        <div key={i} className={`traj-step${live && i === steps.length - 1 ? " before-live" : ""}`}>
          <div className="traj-marker">
            <div className={`traj-dot ${s.type}`}>
              <Icon name={STEP_ICON[s.type] ?? "code"} size={15} />
            </div>
          </div>
          <div className="traj-card">
            <div className="th">
              <span className="st">шаг {s.step}</span>
              <Tag tone={actionTone(s.type)}>{ACTION_TYPE_LABEL[s.type] ?? s.type}</Tag>
              <Tag tone="neutral">{stageLabel(s.stage)}</Tag>
              <span className="st">{s.title}</span>
              {s.budgetRemaining !== undefined && s.budgetRemaining !== null && <span className="st">осталось {s.budgetRemaining}</span>}
            </div>
            {s.thoughts && <div className="thought-inline">{s.thoughts}</div>}
            {s.code && <CodeBlock code={s.code} />}
            {s.feedback.map((f, fi) => (
              <div key={fi} className="fb">
                <span className={`fb-badge fb-${f.ch}`}>{f.ch}</span>
                <span>{f.text}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
      {live && (
        <div className="traj-step traj-live">
          <div className="traj-marker">
            <Spinner size={18} />
          </div>
          <div className="spinner-row">агент выполняет шаг…</div>
        </div>
      )}
    </div>
  );
}

function ChecklistTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.checklist(id), [id], live ? 2500 : 0);
  if (loading && !data) return <Skeleton h={300} />;
  if (!data) return <EmptyState icon="check2" title="Нет данных чеклиста" />;
  return (
    <div>
      <div className="cl-summary">
        <Donut value={data.closed} total={data.total} size={104} percent={data.coverage} />
        <div>
          <div style={{ fontWeight: 700, fontSize: 16 }}>Покрытие DS-пайплайна</div>
          <div className="muted" style={{ maxWidth: 460, marginTop: 4, fontSize: 13.5 }}>
            Подсказки чеклиста неявные: среда лишь намекает на пропущенные этапы.
            Закрыто {data.closed} из {data.total} пунктов{data.coverage != null ? ` · официальное покрытие ${Math.round(data.coverage * 100)}%` : ""}.
          </div>
        </div>
      </div>
      <div className="cl-grid">
        {data.items.map((it) => (
          <div key={it.id} className={`cl-item${it.closed ? " closed" : ""}`}>
            <span className={`cl-ic ${it.closed ? "yes" : "no"}`}><Icon name={it.closed ? "check" : "x"} size={13} /></span>
            <div>
              <div className="cl-label">{it.label}</div>
              {it.desc && <div className="cl-desc">{it.desc}</div>}
              {it.closed && it.closedStep != null && <div className="cl-step">закрыт на шаге {it.closedStep}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ErrorsTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.errors(id), [id], live ? 3000 : 0);
  if (loading && !data) return <Skeleton h={200} />;
  const errs = data ?? [];
  if (!errs.length) return <EmptyState icon="check2" title="Ошибок нет" text="Ни на одном шаге не было исключений." />;
  return (
    <div>
      {errs.map((e, i) => (
        <div key={i} className="err-block">
          <div className="err-head"><Icon name="bug" size={15} /> шаг {e.step}: {e.type}{e.value ? ` — ${e.value}` : ""}</div>
          <CodeBlock code={e.traceback || e.stderr || e.value} />
        </div>
      ))}
    </div>
  );
}

function LogsTab({ id, live }: { id: string; live: boolean }) {
  const { data, loading } = useAsync(() => api.logs(id), [id], live ? 2500 : 0);
  if (loading && !data) return <Skeleton h={300} />;
  const msgs = data?.messages ?? [];
  return (
    <div>
      {data?.processLog && (
        <>
          <div className="cm-label" style={{ marginBottom: 8 }}>Лог процесса</div>
          <div className="process-log">{data.processLog}{live ? "\n…" : ""}</div>
        </>
      )}
      {msgs.length > 0 && (
        <div className="logs" style={{ marginTop: data?.processLog ? 20 : 0 }}>
          {msgs.map((m, i) => (
            <div key={i} className={`log-msg ${m.role}`}>
              <div className="role">{m.role === "assistant" ? "агент" : m.role === "tool" ? "среда" : m.role}</div>
              {m.role === "assistant" && m.type !== "submit" ? <CodeBlock code={m.text} /> : <pre>{m.text}</pre>}
            </div>
          ))}
        </div>
      )}
      {!msgs.length && !data?.processLog && <EmptyState icon="terminal" title="Логи пусты" />}
    </div>
  );
}

/* ---- page ---- */
export default function RunDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [tab, setTab] = useState("notebook");
  const { data: run, loading, reload } = useAsync(() => api.getRun(id), [id], 2500);

  if (loading && !run) return <Skeleton h={400} />;
  if (!run) return <EmptyState icon="alert" title="Прогон не найден" action={<Button onClick={() => nav("/runs")}>К прогонам</Button>} />;

  const live = run.status === "running";
  const pct = run.steps ? (run.step / run.steps) * 100 : 6;
  // Show the «Мысли» tab when the run kept a scratchpad OR produced a post-run
  // self-summary. New runs always have a summary; old runs with neither stay hidden.
  const showThoughts = !!run.thoughtsEnabled || !!run.hasSummary;
  const visibleTabs = TABS.filter((t) => t.id !== "thoughts" || showThoughts);
  const activeTab = tab === "thoughts" && !showThoughts ? "notebook" : tab;
  const imp = improvementPct(run);

  async function stop() {
    try { await api.stopRun(id); reload(); } catch { /* ignore */ }
  }

  const scoreCls = run.status === "success" ? "success" : run.status === "running" ? "running" : run.status === "failed" ? "failed" : "null";

  return (
    <div>
      <button className="back-link" onClick={() => nav("/runs")}><Icon name="chevronLeft" size={16} /> Все прогоны</button>

      {live && (
        <div className="live-bar">
          <StatusBadge status="running" />
          <span className="lb-status">{run.command ? `выполняется: ${run.model}` : "агент работает…"}</span>
          <ProgressBar pct={pct} animated />
          <span className="mono" style={{ fontSize: 13 }}>шаг {run.step}{run.steps ? `/${run.steps}` : ""}</span>
          <span className="mono" style={{ fontSize: 13, opacity: 0.8 }}><LiveDuration startedMs={run.startedMs} running dur={run.dur} /></span>
          <Button variant="danger" size="sm" icon="stop" onClick={stop}>Остановить</Button>
        </div>
      )}

      <Card className="run-head">
        <div>
          <div className="row" style={{ gap: 12 }}>
            <span className="mono faint">{run.shortId}</span>
            <StatusBadge status={run.status} />
          </div>
          <div className="run-meta-line">
            <span className="mono">{run.model}</span>
            <Tag tone={run.mode === "gym" || run.mode === "batch" ? "accent" : "neutral"}>{MODE_LABELS[run.mode]}</Tag>
            <span>·</span>
            <span>{run.dataset}</span>
          </div>
          <div className="chip-metrics">
            <ChipMetric label="чеклист" value={`${run.checklist}/${run.checklistTotal}`}
              ring={<ProgressRing value={run.checklist} max={run.checklistTotal} size={40} tone="green" label={run.checklistCoverage != null ? `${Math.round(run.checklistCoverage * 100)}%` : ""} />} />
            <span className="vline" />
            <ChipMetric label="ошибок" value={run.errors} />
            <span className="vline" />
            <ChipMetric label="шагов" value={`${run.step}${run.steps ? `/${run.steps}` : ""}`} />
            <span className="vline" />
            <ChipMetric label="Этап" value={stageLabel(run.currentStage)} />
            <span className="vline" />
            <ChipMetric label="токены (in+out)" value={`${formatTokens(run.tokIn)} + ${formatTokens(run.tokOut)}`} />
            <span className="vline" />
            <ChipMetric label="время" value={<LiveDuration startedMs={run.startedMs} running={live} dur={run.dur} />} />
          </div>
        </div>

        <div className={`score-panel ${scoreCls}`}>
          <span className="sp-label">{run.metric ?? "метрика"} · test</span>
          <span className="sp-val">{run.status === "running" ? "…" : formatScore(run.score, run.metric)}</span>
          {run.status === "failed" || run.status === "null" ? (
            <span className="sp-base">{run.status === "null" ? "не дошёл до сабмита" : "сабмит не прошёл"}</span>
          ) : imp != null ? (
            <span className="sp-base" title="Финальный скор на тесте относительно лучшей метрики на валидации в этом прогоне"><Icon name={imp >= 0 ? "arrowUp" : "arrowDown"} size={14} /> {imp >= 0 ? "+" : ""}{imp.toFixed(1)}% к лучшей валидации</span>
          ) : run.baseline != null ? (
            <span className="sp-base">лучшая валидация {formatScore(run.baseline, run.metric)}</span>
          ) : <span className="sp-base">&nbsp;</span>}
        </div>
      </Card>

      {(run.status === "failed" || run.status === "null") && run.failReason && (
        <div className="fail-banner">
          <Icon name="alert" size={18} />
          <span>{run.failReason}</span>
          {run.finalStatus && <span className="mono faint" style={{ marginLeft: "auto", fontSize: 12 }}>{run.finalStatus}</span>}
        </div>
      )}

      <div style={{ marginTop: 24 }}>
        <Tabs tabs={visibleTabs} active={activeTab} onChange={setTab} />
        {activeTab === "notebook" && <NotebookTab id={id} live={live} />}
        {activeTab === "trajectory" && <TrajectoryTab id={id} live={live} />}
        {activeTab === "thoughts" && <ThoughtsTab id={id} live={live} />}
        {activeTab === "checklist" && <ChecklistTab id={id} live={live} />}
        {activeTab === "errors" && <ErrorsTab id={id} live={live} />}
        {activeTab === "logs" && <LogsTab id={id} live={live} />}
      </div>
    </div>
  );
}
