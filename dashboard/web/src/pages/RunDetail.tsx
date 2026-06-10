import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { ChecklistItem, Run, RunError, TrajectoryStep } from "../lib/api";
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
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${keyBase}-${i}`} className="run-summary-code">{part.slice(1, -1)}</code>;
    }
    return <span key={`${keyBase}-${i}`}>{part}</span>;
  });
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

type SummarySection = { title: string; body: string };

function parseSummarySections(text: string): SummarySection[] {
  const sections: SummarySection[] = [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let current: SummarySection | null = null;

  const pushCurrent = () => {
    if (!current) return;
    current.body = current.body.replace(/\s+/g, " ").trim();
    if (current.title && current.body) sections.push(current);
    current = null;
  };

  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) {
      pushCurrent();
      return;
    }
    const match = line.match(/^\*\*(.+?)\*\*\s*[—-]\s*(.+)$/);
    if (match) {
      pushCurrent();
      current = { title: match[1].trim(), body: match[2].trim() };
      return;
    }
    if (!current) {
      current = { title: "", body: line };
      return;
    }
    current.body = `${current.body} ${line}`.trim();
  });
  pushCurrent();

  return sections.filter((section) => section.title);
}

function SummaryCard({ text, model }: { text: string; model?: string | null }) {
  const sections = parseSummarySections(text);
  return (
    <div className="run-summary">
      <div className="run-summary-head">
        <Icon name="sparkles" size={15} />
        <span className="run-summary-title">Саммари решения</span>
        {model && <span className="run-summary-model mono">{model}</span>}
      </div>
      <div className="run-summary-body">
        {sections.length > 0 ? (
          <div className="run-summary-sections">
            {sections.map((section, index) => (
              <div key={`${section.title}-${index}`} className="run-summary-section">
                <div className="run-summary-section-title">{section.title}</div>
                <div className="run-summary-section-body">
                  {renderInline(section.body, `summary-${index}`)}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <MarkdownLite text={text} />
        )}
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
    <div className="thoughts-tab">
      {summaryText && <SummaryCard text={summaryText} model={summary?.model} />}
      {thoughtItems.length > 0 && (
        <>
          {summaryText && <div className="thoughts-steps-head">Ход рассуждений по шагам</div>}
          <div className="thoughts">
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
        </>
      )}
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

const CHECKLIST_GROUPS = [
  {
    title: "Данные и задача",
    note: "Понимание цели, схемы и качества входных данных.",
    ids: ["task_understanding", "schema_review", "target_distribution_review", "missing_values_audit", "categorical_features_audit", "duplicates_audit", "suspicious_columns_audit"],
  },
  {
    title: "Модельный контур",
    note: "Безопасное отделение target и первый рабочий кандидат.",
    ids: ["target_exclusion", "baseline_candidate_created"],
  },
  {
    title: "Валидация и submit",
    note: "Проверка решения перед единственным hidden-test submit.",
    ids: ["validation_evaluated", "reproducible_solution", "submit_ready_artifact"],
  },
];

function ChecklistGroup({ title, note, items }: { title: string; note: string; items: ChecklistItem[] }) {
  const closed = items.filter((it) => it.closed).length;
  const pct = items.length ? Math.round((closed / items.length) * 100) : 0;
  return (
    <section className="cl-group">
      <div className="cl-group-head">
        <div>
          <div className="cl-group-title">{title}</div>
          <div className="cl-group-note">{note}</div>
        </div>
        <div className="cl-group-score mono">{closed}/{items.length}</div>
      </div>
      <div className="cl-group-bar" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
      <div className="cl-grid">
        {items.map((it) => {
          const evidence = it.evidence ?? [];
          return (
            <div key={it.id} className={`cl-item${it.closed ? " closed" : ""}`}>
              <span className={`cl-ic ${it.closed ? "yes" : "no"}`}><Icon name={it.closed ? "check" : "x"} size={13} /></span>
              <div>
                <div className="cl-label">{it.label}</div>
                {it.desc && <div className="cl-desc">{it.desc}</div>}
                {it.closed && it.closedStep != null && <div className="cl-step">закрыт на шаге {it.closedStep}</div>}
                {it.closed && evidence.length > 0 && (
                  <details className="cl-evidence">
                    <summary>доказательства</summary>
                    <div className="cl-evidence-list">
                      {evidence.slice(0, 4).map((ev, idx) => (
                        <div key={idx} className="cl-evidence-row">
                          <span className="mono">{ev.step != null ? `шаг ${ev.step}` : "шаг ?"}</span>
                          {ev.cellId && <span className="mono">{ev.cellId}</span>}
                          {ev.reason && <span>{ev.reason}</span>}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ChecklistGroups({ items }: { items: ChecklistItem[] }) {
  const grouped = CHECKLIST_GROUPS.map((group) => ({
    ...group,
    items: group.ids
      .map((itemId) => items.find((item) => item.id === itemId))
      .filter((item): item is ChecklistItem => Boolean(item)),
  })).filter((group) => group.items.length > 0);
  const groupedIds = new Set(grouped.flatMap((group) => group.items.map((item) => item.id)));
  const otherItems = items.filter((item) => !groupedIds.has(item.id));

  return (
    <div className="cl-groups">
      {grouped.map((group) => <ChecklistGroup key={group.title} title={group.title} note={group.note} items={group.items} />)}
      {otherItems.length > 0 && <ChecklistGroup title="Другое" note="Пункты без назначенной группы." items={otherItems} />}
    </div>
  );
}

function healthTone(status: "good" | "warn" | "bad" | "neutral") {
  return `health-item health-${status}`;
}

function RunHealthStrip({ run }: { run: Run }) {
  const checklistPct = run.checklistTotal ? run.checklist / run.checklistTotal : 0;
  const checklistTone = checklistPct >= 0.75 ? "good" : checklistPct >= 0.5 ? "warn" : "neutral";
  const submitTone = run.status === "success" ? "good" : run.status === "failed" ? "bad" : run.status === "running" ? "warn" : "neutral";
  const validationTone = run.baseline != null || run.score != null ? "good" : run.status === "running" ? "warn" : "neutral";
  const budgetTone = run.steps && run.step >= run.steps && run.status !== "success" ? "warn" : "neutral";
  const resultLabel = run.status === "success" ? "Успешно" : run.status === "running" ? "В работе" : run.status === "failed" ? "Ошибка" : "Нет submit";
  const validationLabel = run.baseline != null ? formatScore(run.baseline, run.metric) : run.score != null ? "score есть" : "нет";

  return (
    <div className="run-health-strip" aria-label="Состояние прогона">
      <div className={healthTone(submitTone)}>
        <Icon name={run.status === "success" ? "check2" : run.status === "failed" ? "alert" : "route"} size={16} />
        <div>
          <span>Итог</span>
          <strong title={run.finalStatus ?? run.status}>{resultLabel}</strong>
        </div>
      </div>
      <div className={healthTone(validationTone)}>
        <Icon name="check" size={16} />
        <div>
          <span>Валидация</span>
          <strong>{validationLabel}</strong>
        </div>
      </div>
      <div className={healthTone(checklistTone)}>
        <Icon name="check2" size={16} />
        <div>
          <span>Чеклист</span>
          <strong>{run.checklist}/{run.checklistTotal}</strong>
        </div>
      </div>
      <div className={healthTone(run.errors > 0 ? "bad" : "good")}>
        <Icon name="bug" size={16} />
        <div>
          <span>Ошибки</span>
          <strong>{run.errors}</strong>
        </div>
      </div>
      <div className={healthTone(budgetTone)}>
        <Icon name="clock" size={16} />
        <div>
          <span>Шаги</span>
          <strong>{run.step}{run.steps ? `/${run.steps}` : ""}</strong>
        </div>
      </div>
    </div>
  );
}

function metricHigherIsBetter(metric?: string | null) {
  const m = (metric ?? "").toLowerCase();
  return !(m.includes("rmse") || m.includes("mae") || m.includes("mse") || m.includes("loss") || m.includes("error"));
}

function compareScore(a: Run, b: Run, higherBetter: boolean) {
  if (a.score == null && b.score == null) return 0;
  if (a.score == null) return 1;
  if (b.score == null) return -1;
  return higherBetter ? b.score - a.score : a.score - b.score;
}

function comparisonRuns(run: Run, runs?: Run[]) {
  const source = runs ?? [];
  const batchRuns = run.batchId ? source.filter((candidate) => candidate.batchId === run.batchId) : [];
  const related = batchRuns.length > 1
    ? batchRuns
    : source.filter((candidate) => candidate.taskDir ? candidate.taskDir === run.taskDir : candidate.task === run.task);
  const unique = new Map<string, Run>();
  [run, ...related].forEach((candidate) => unique.set(candidate.id, candidate));
  return Array.from(unique.values());
}

function metricDirectionLabel(metric?: string | null) {
  return metricHigherIsBetter(metric) ? "выше score лучше" : "ниже score лучше";
}

function ExperimentStoryPanel({ run, runs }: { run: Run; runs?: Run[] }) {
  const peers = comparisonRuns(run, runs);
  const higherBetter = metricHigherIsBetter(run.metric);
  const scored = peers.filter((candidate) => candidate.score != null).sort((a, b) => compareScore(a, b, higherBetter));
  const best = scored[0];
  const currentRank = scored.findIndex((candidate) => candidate.id === run.id) + 1;
  const isBatchStory = Boolean(run.batchId && peers.some((candidate) => candidate.id !== run.id && candidate.batchId === run.batchId));
  const scopeLabel = isBatchStory ? "Пачка эксперимента" : "Та же задача";
  const bestLabel = best
    ? best.id === run.id
      ? "текущий run"
      : `${MODE_LABELS[best.mode]} · ${best.model}`
    : "нет scored run";
  const currentGap = best?.score != null && run.score != null
    ? Math.abs(run.score - best.score)
    : null;
  const checklistPct = run.checklistTotal ? Math.round((run.checklist / run.checklistTotal) * 100) : 0;
  const verdict = run.status === "running"
    ? "Run ещё идёт: история будет уточняться по мере появления score и checklist."
    : run.status === "success"
      ? currentRank === 1
        ? "Этот прогон сейчас выглядит как лучший scored вариант в выбранном сравнении."
        : "Есть scored вариант лучше; полезно сравнить отличия по траектории и чеклисту."
      : run.errors > 0
        ? "Главная ценность этого run сейчас в диагностике: он показывает, где агент сорвался."
        : "Run не дошёл до полноценного результата; стоит проверить финальный статус и логи.";

  return (
    <section className="story-panel">
      <div className="story-head">
        <div>
          <div className="story-title"><Icon name="sparkles" size={15} /> История эксперимента</div>
          <div className="story-note">{scopeLabel}: {peers.length} run · {metricDirectionLabel(run.metric)}</div>
        </div>
        {currentRank > 0 && <div className="story-rank mono">#{currentRank}/{scored.length}</div>}
      </div>
      <div className="story-grid">
        <div className="story-card">
          <span>Лучший score</span>
          <strong>{best ? formatScore(best.score, run.metric) : "-"}</strong>
          <em>{bestLabel}</em>
        </div>
        <div className="story-card">
          <span>Текущий run</span>
          <strong>{formatScore(run.score, run.metric)}</strong>
          <em>{currentGap == null || best?.id === run.id ? "без отставания" : `gap ${formatScore(currentGap, run.metric)}`}</em>
        </div>
        <div className="story-card">
          <span>Контроль качества</span>
          <strong>{checklistPct}%</strong>
          <em>чеклист {run.checklist}/{run.checklistTotal} · ошибки {run.errors}</em>
        </div>
      </div>
      <div className="story-verdict">{verdict}</div>
    </section>
  );
}

function diagnoseFailure(run: Run, errors: RunError[]) {
  const first = errors[0];
  const text = [
    run.failReason,
    run.finalStatus,
    first?.type,
    first?.value,
    first?.traceback,
    first?.stderr,
  ].filter(Boolean).join("\n").toLowerCase();

  if (text.includes("model check") || text.includes("raw val") || text.includes("predict")) {
    return {
      title: "Модель не прошла readiness check",
      detail: "Кандидат, похоже, не умеет предсказывать на raw validation dataframe перед hidden-test submit.",
      next: "Открыть notebook и проверить, что preprocessing встроен в pipeline модели.",
    };
  }
  if (text.includes("json") || text.includes("contract") || text.includes("invalid action") || text.includes("schema")) {
    return {
      title: "Сорвался action contract",
      detail: "Агент отправил действие, которое среда не смогла принять как корректный протокол.",
      next: "Смотреть траекторию вокруг первого contract/runtime feedback.",
    };
  }
  if (text.includes("timeout") || text.includes("timed out")) {
    return {
      title: "Таймаут выполнения",
      detail: "Код или проверка заняли слишком много времени для текущего бюджета исполнения.",
      next: "Проверить тяжёлые операции, подбор гиперпараметров и повторные полные прогоны.",
    };
  }
  if (text.includes("submit")) {
    return {
      title: "Submit не прошёл",
      detail: "Run дошёл до финальной зоны, но submit завершился ошибкой или был отклонён.",
      next: "Сравнить final status, ошибки и последний кандидат модели.",
    };
  }
  if (run.status === "null") {
    return {
      title: "Не дошёл до submit",
      detail: "У run нет финального test score; обычно это значит, что бюджет или сценарий закончился раньше submit.",
      next: "Проверить последние шаги траектории и логи запуска.",
    };
  }
  return {
    title: run.errors > 0 ? "Ошибка выполнения кода" : "Нужна ручная проверка",
    detail: first ? `Первый сбой на шаге ${first.step}: ${first.type || first.value || "runtime error"}.` : "Явной ошибки в artifact API не найдено.",
    next: "Открыть Errors и Logs, затем сравнить с успешным run той же задачи.",
  };
}

function FailureDiagnosisPanel({
  id,
  live,
  run,
  onOpenErrors,
  onOpenLogs,
}: {
  id: string;
  live: boolean;
  run: Run;
  onOpenErrors: () => void;
  onOpenLogs: () => void;
}) {
  const { data } = useAsync(() => api.errors(id), [id], live ? 3000 : 0);
  const errors = data ?? [];
  const diagnosis = diagnoseFailure(run, errors);
  const first = errors[0];

  return (
    <section className="diagnosis-panel">
      <div className="diagnosis-main">
        <div className="diagnosis-title"><Icon name="alert" size={16} /> Диагностика сбоя</div>
        <strong>{diagnosis.title}</strong>
        <p>{diagnosis.detail}</p>
        <div className="diagnosis-next">{diagnosis.next}</div>
      </div>
      <div className="diagnosis-side">
        <div className="diagnosis-kv"><span>Этап</span><strong>{stageLabel(run.currentStage)}</strong></div>
        <div className="diagnosis-kv"><span>Первый error</span><strong>{first ? `шаг ${first.step}` : "не найден"}</strong></div>
        <div className="diagnosis-actions">
          <Button size="sm" icon="bug" onClick={onOpenErrors}>Ошибки</Button>
          <Button size="sm" icon="terminal" onClick={onOpenLogs}>Логи</Button>
        </div>
      </div>
    </section>
  );
}

function PeerRunPanel({ run, runs }: { run: Run; runs?: Run[] }) {
  const peers = comparisonRuns(run, runs)
    .filter((candidate) => candidate.id !== run.id)
    .sort((a, b) => b.startedMs - a.startedMs)
    .slice(0, 5);
  if (!peers.length) return null;

  const higherBetter = metricHigherIsBetter(run.metric);
  const scored = [run, ...peers]
    .filter((candidate) => candidate.score != null)
    .sort((a, b) => compareScore(a, b, higherBetter));
  const best = scored[0];
  const currentRank = scored.findIndex((candidate) => candidate.id === run.id) + 1;
  const delta = best?.score != null && run.score != null ? run.score - best.score : null;
  const isCurrentBest = best?.id === run.id;
  const gapText = delta == null ? "-" : isCurrentBest ? "лидер" : formatScore(Math.abs(delta), run.metric);

  return (
    <div className="peer-panel">
      <div className="peer-head">
        <div>
          <div className="peer-title">Сравнение по этой задаче</div>
          <div className="peer-note">Последние 5 запусков той же задачи. Ранг считается только среди прогонов, где есть test score.</div>
        </div>
        {currentRank > 0 && <div className="peer-rank mono">#{currentRank}/{scored.length}</div>}
      </div>
      <div className="peer-grid">
        <div className="peer-best">
          <span>Лучший из сравнения</span>
          <strong>{best ? formatScore(best.score, run.metric) : "-"}</strong>
          {best && <em>{best.id === run.id ? "текущий run" : `${MODE_LABELS[best.mode]} · ${best.model}`}</em>}
        </div>
        <div className="peer-best">
          <span>{isCurrentBest ? "Позиция текущего" : "Отставание от лучшего"}</span>
          <strong>{gapText}</strong>
          <em>{higherBetter ? "выше score лучше" : "ниже score лучше"}</em>
        </div>
        <div className="peer-list">
          {peers.map((peer) => (
            <div key={peer.id} className="peer-row">
              <span className="mono">{peer.shortId}</span>
              <Tag tone={peer.status === "success" ? "green" : peer.status === "failed" ? "red" : peer.status === "running" ? "blue" : "neutral"}>{MODE_LABELS[peer.mode]}</Tag>
              <span className="peer-score mono">{formatScore(peer.score, peer.metric ?? run.metric)}</span>
              <span className="peer-meta">чеклист {peer.checklist}/{peer.checklistTotal} · ошибки {peer.errors}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function milestoneForStep(step: TrajectoryStep): { label: string; tone: "green" | "blue" | "accent" | "red" } | null {
  if (step.type === "submit") return { label: "Финальный submit", tone: "green" };
  if (step.type === "validate" || step.type === "quick_validate" || step.type === "check_candidate") return { label: "Проверка модели", tone: "blue" };
  if (step.type === "restart_and_run_all") return { label: "Чистый прогон", tone: "accent" };
  if (step.feedback.some((f) => f.ch === "contract")) return { label: "Проверка контракта", tone: "blue" };
  if (step.feedback.some((f) => f.ch === "checklist" || f.ch === "checklist-hint")) return { label: "Подсказка чеклиста", tone: "green" };
  if (step.feedback.some((f) => f.text.toLowerCase().includes("error") || f.text.toLowerCase().includes("traceback"))) return { label: "Ошибка выполнения", tone: "red" };
  return null;
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
  const milestones = steps
    .map((step) => ({ step, milestone: milestoneForStep(step) }))
    .filter((item) => item.milestone)
    .slice(0, 6);
  if (!steps.length) return <EmptyState icon="route" title="Нет траектории" text="Шаги агента появятся здесь." />;
  return (
    <div className="traj">
      {milestones.length > 0 && (
        <div className="milestone-strip">
          {milestones.map(({ step, milestone }) => (
            <span key={`${step.step}-${milestone!.label}`} className={`milestone-chip milestone-${milestone!.tone}`}>
              <Icon name={STEP_ICON[step.type] ?? "route"} size={13} />
              <span>{milestone!.label}</span>
              <span className="mono">#{step.step}</span>
            </span>
          ))}
        </div>
      )}
      {steps.map((s, i) => {
        const milestone = milestoneForStep(s);
        return (
        <div key={i} className={`traj-step${milestone ? ` milestone milestone-${milestone.tone}` : ""}${live && i === steps.length - 1 ? " before-live" : ""}`}>
          <div className="traj-marker">
            <div className={`traj-dot ${s.type}`}>
              <Icon name={STEP_ICON[s.type] ?? "code"} size={15} />
            </div>
          </div>
          <div className="traj-card">
            <div className="th">
              <span className="st">шаг {s.step}</span>
              <Tag tone={actionTone(s.type)}>{ACTION_TYPE_LABEL[s.type] ?? s.type}</Tag>
              {milestone && <Tag tone={milestone.tone === "red" ? "red" : milestone.tone === "green" ? "green" : milestone.tone === "blue" ? "blue" : "accent"}>{milestone.label}</Tag>}
              <Tag tone="neutral">{stageLabel(s.stage)}</Tag>
              <span className="st">{s.title}</span>
              {s.budgetRemaining !== undefined && s.budgetRemaining !== null && <span className="st">осталось {s.budgetRemaining}</span>}
            </div>
            {s.code && <CodeBlock code={s.code} />}
            {s.feedback.map((f, fi) => (
              <div key={fi} className="fb">
                <span className={`fb-badge fb-${f.ch}`}>{f.ch}</span>
                <span>{f.text}</span>
              </div>
            ))}
          </div>
        </div>
      )})}
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
      <ChecklistGroups items={data.items} />
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
  const { data: runs } = useAsync(() => api.listRuns(), [], 5000);

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
            <Tag tone={run.mode === "directive" || run.mode === "batch" ? "accent" : "neutral"}>{MODE_LABELS[run.mode]}</Tag>
            <span>·</span>
            <span>{run.task}</span>
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

      <RunHealthStrip run={run} />
      <ExperimentStoryPanel run={run} runs={runs ?? undefined} />
      <PeerRunPanel run={run} runs={runs ?? undefined} />

      {(run.status === "failed" || run.status === "null") && run.failReason && (
        <div className="fail-banner">
          <Icon name="alert" size={18} />
          <span>{run.failReason}</span>
          {run.finalStatus && <span className="mono faint" style={{ marginLeft: "auto", fontSize: 12 }}>{run.finalStatus}</span>}
        </div>
      )}

      {(run.status === "failed" || run.status === "null" || run.errors > 0) && (
        <FailureDiagnosisPanel
          id={id}
          live={live}
          run={run}
          onOpenErrors={() => setTab("errors")}
          onOpenLogs={() => setTab("logs")}
        />
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
