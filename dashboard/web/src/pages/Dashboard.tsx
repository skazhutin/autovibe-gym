import { useNavigate } from "react-router-dom";
import { api, type Run } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_SHORT, formatScore, formatTokens, timeAgo } from "../lib/format";
import { Button, Card, EmptyState, LiveDuration, ProgressBar, ProgressRing, Skeleton, StatusBadge } from "../components/ui";
import { Icon } from "../components/Icon";
import { BarChart, Sparkline } from "../components/charts";
import { ModeTag, ScoreCell } from "../components/runbits";

function MetricCard({ label, value, icon, spark, delta, dark }: {
  label: string; value: string; icon: string; spark?: number[]; delta?: string; dark?: boolean;
}) {
  return (
    <Card className={`metric-card${dark ? " dark" : ""}`} style={dark ? { background: "var(--ink)", color: "#f4f4f4", borderColor: "transparent" } : undefined}>
      <div className="metric-foot" style={{ marginTop: 0 }}>
        <span className="metric-label">{label}</span>
        <span className="ic"><Icon name={icon} size={18} /></span>
      </div>
      <span className="metric-num">{value}</span>
      <div className="metric-foot">
        {spark && spark.length > 1 ? <Sparkline data={spark} tone={dark ? "accent" : "dim"} /> : <span />}
        {delta && <span className="metric-delta faint">{delta}</span>}
      </div>
    </Card>
  );
}

function ActiveRunCard({ run }: { run: Run }) {
  const nav = useNavigate();
  const pct = run.steps ? (run.step / run.steps) * 100 : 8;
  return (
    <Card hover className="runcard" onClick={() => nav(`/runs/${run.id}`)}>
      <div className="top">
        <div>
          <div className="model">{run.model}</div>
          <div className="chips" style={{ marginTop: 8 }}>
            <ModeTag mode={run.mode} />
            <span className="tag">{run.task}</span>
          </div>
        </div>
        <StatusBadge status={run.status} />
      </div>
      <div className="body">
        <ProgressRing value={run.step} max={run.steps ?? 1} size={52} label={`${run.step}/${run.steps ?? "?"}`} />
        <div style={{ flex: 1 }}>
          <ProgressBar pct={pct} animated />
          <div className="meta" style={{ marginTop: 10 }}>
            <span>{formatTokens(run.tokIn + run.tokOut)} токенов</span>
            <span><LiveDuration startedMs={run.startedMs} running dur={run.dur} /></span>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default function Dashboard() {
  const nav = useNavigate();
  const { data: runs, loading } = useAsync(() => api.listRuns(), [], 5000);

  if (loading && !runs) {
    return (
      <div className="stack">
        <div className="grid-4">{[0, 1, 2, 3].map((i) => <Card key={i}><Skeleton h={70} /></Card>)}</div>
        <Card><Skeleton h={200} /></Card>
      </div>
    );
  }

  const all = runs ?? [];
  const active = all.filter((r) => r.status === "running");
  const finished = all.filter((r) => r.status !== "running");
  const scored = finished.filter((r) => r.score !== null).sort((a, b) => b.startedMs - a.startedMs);
  const avgTokens = finished.length
    ? Math.round(finished.reduce((s, r) => s + r.tokIn + r.tokOut, 0) / finished.length)
    : 0;

  const lastScored = scored[0] ?? null;

  const FIVE_MODES: Run["mode"][] = ["single", "repeated", "iterative", "gym", "fixed"];
  const modeBars = FIVE_MODES.map((m) => {
    const last = scored.find((r) => r.mode === m);
    return last ? { label: MODE_SHORT[m], value: last.score!, best: false } : null;
  }).filter((x): x is { label: string; value: number; best: boolean } => x !== null);

  const scoreSpark = scored.slice(0, 8).map((r) => r.score!).reverse();

  return (
    <div className="stack">
      <div className="grid-4">
        <MetricCard label="Всего прогонов" value={String(all.length)} icon="runs" spark={scoreSpark} delta={`${finished.length} завершено`} />
        <MetricCard dark label="Активных сейчас" value={String(active.length)} icon="play" delta={active.length ? "идут прямо сейчас" : "нет активных"} />
        <MetricCard label="Последний test-скор" value={lastScored ? formatScore(lastScored.score, lastScored.metric) : "—"} icon="check2" spark={scoreSpark} delta={lastScored ? `${MODE_SHORT[lastScored.mode]} · ${lastScored.task}` : "нет данных"} />
        <MetricCard label="Средний расход токенов" value={formatTokens(avgTokens)} icon="coins" delta="на прогон" />
      </div>

      <section>
        <h2 className="section-title">Активные прогоны</h2>
        {active.length ? (
          <div className="grid-2">{active.map((r) => <ActiveRunCard key={r.id} run={r} />)}</div>
        ) : (
          <Card>
            <EmptyState icon="play" title="Нет активных прогонов"
              text="Запустите LLM-агента, чтобы наблюдать за решением в реальном времени."
              action={<Button variant="primary" icon="plus" onClick={() => nav("/new")}>Новый прогон</Button>} />
          </Card>
        )}
      </section>

      <div className="dash-bottom" style={{ display: "grid", gridTemplateColumns: "1.55fr 1fr", gap: 16 }}>
        <Card style={{ minWidth: 0 }}>
          <div className="spread" style={{ marginBottom: 14 }}>
            <h2 className="section-title" style={{ margin: 0 }}>Последние завершённые</h2>
            <Button variant="ghost" size="sm" onClick={() => nav("/runs")}>Все прогоны</Button>
          </div>
          {finished.length ? (
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>ID</th><th>Модель</th><th>Режим</th><th>Датасет</th><th>Скор</th><th>Статус</th><th>Когда</th></tr></thead>
                <tbody>
                  {finished.slice(0, 7).map((r) => (
                    <tr key={r.id} className="clickable" onClick={() => nav(`/runs/${r.id}`)}>
                      <td className="mono faint">{r.shortId}</td>
                      <td className="mono">{r.model}</td>
                      <td><ModeTag mode={r.mode} /></td>
                      <td>{r.task}</td>
                      <td><ScoreCell run={r} /></td>
                      <td><StatusBadge status={r.status} /></td>
                      <td className="faint">{timeAgo(r.startedMs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState icon="runs" title="Прогонов пока нет" text="Завершённые запуски появятся здесь." />
          )}
        </Card>

        <Card style={{ minWidth: 0 }}>
          <h2 className="section-title">Последние скоры по режимам</h2>
          {modeBars.length ? (
            <BarChart data={modeBars} fmt={(v) => v.toFixed(3)} />
          ) : (
            <EmptyState icon="compare" title="Нет успешных прогонов" text="Сравнение появится после первого сабмита." />
          )}
        </Card>
      </div>
    </div>
  );
}
