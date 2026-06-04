import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Run } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_SHORT, formatDuration, formatScore, formatTokens } from "../lib/format";
import { Card, EmptyState, Skeleton } from "../components/ui";
import { BarChart, Scatter } from "../components/charts";
import { ModeTag } from "../components/runbits";

type GroupBy = "none" | "mode" | "model" | "dataset";

export default function Compare() {
  const nav = useNavigate();
  const { data: runs, loading } = useAsync(() => api.listRuns(), []);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [groupBy, setGroupBy] = useState<GroupBy>("none");

  const successful = useMemo(() => (runs ?? []).filter((r) => r.status === "success" && r.score !== null), [runs]);

  useEffect(() => {
    if (selected.size === 0 && successful.length) {
      setSelected(new Set(successful.slice(0, Math.min(5, successful.length)).map((r) => r.id)));
    }
  }, [successful, selected.size]);

  const picked = successful.filter((r) => selected.has(r.id));

  // best per column (higher score = better assuming neg/normalized metrics)
  const bestScore = Math.max(...picked.map((r) => r.score ?? -Infinity));
  const bestChecklist = Math.max(...picked.map((r) => r.checklist));
  const fewestErrors = Math.min(...picked.map((r) => r.errors));
  const fewestTokens = Math.min(...picked.map((r) => r.tokIn + r.tokOut));

  // bar chart only meaningful for one dataset (comparable metric)
  const datasets = new Set(picked.map((r) => r.dataset));
  const barData = datasets.size === 1
    ? picked.map((r) => ({ label: `${MODE_SHORT[r.mode]}`, value: r.score!, best: r.score === bestScore, sub: r.model }))
    : [];

  // Cost-per-point: tokens vs improvement over a per-dataset baseline (the
  // single-shot score if present, else the weakest run on that dataset).
  const scoredPicked = picked.filter((r) => r.score !== null && r.score !== undefined);
  const goalMin = (m?: string | null) =>
    !!(m && /rmse|rmsle|mae|mse|logloss/.test(m.toLowerCase()) && !m.toLowerCase().startsWith("neg_"));
  const baselineByDs = new Map<string, number>();
  for (const ds of new Set(scoredPicked.map((r) => r.dataset))) {
    const dsRuns = scoredPicked.filter((r) => r.dataset === ds);
    const single = dsRuns.find((r) => r.mode === "single");
    baselineByDs.set(ds, single ? single.score! : Math.min(...dsRuns.map((r) => r.score!)));
  }
  const scatterPts = scoredPicked.map((r) => {
    const base = baselineByDs.get(r.dataset) ?? 0;
    const imp = base
      ? ((goalMin(r.metric) ? base - r.score! : r.score! - base) / Math.abs(base)) * 100
      : 0;
    return {
      x: r.tokIn + r.tokOut,
      y: imp,
      label: `${MODE_SHORT[r.mode]} · ${r.dataset}`,
      highlight: r.mode === "gym" || r.mode === "iterative",
    };
  });

  const sortedPicked = [...picked].sort((a, b) => {
    if (groupBy === "none") return 0;
    const key = (r: Run) => (groupBy === "mode" ? r.mode : groupBy === "model" ? r.model : r.dataset);
    return key(a).localeCompare(key(b));
  });

  function toggle(idr: string) {
    setSelected((s) => {
      const n = new Set(s);
      n.has(idr) ? n.delete(idr) : n.add(idr);
      return n;
    });
  }

  if (loading && !runs) return <Skeleton h={400} />;

  return (
    <div className="compare">
      <Card className="cmp-list">
        <div className="spread" style={{ marginBottom: 8 }}>
          <strong style={{ fontSize: 14 }}>Прогоны</strong>
          <span className="faint" style={{ fontSize: 12 }}>{selected.size} выбр.</span>
        </div>
        {successful.length === 0 && <div className="muted" style={{ fontSize: 13 }}>Нет успешных прогонов для сравнения.</div>}
        {successful.map((r) => (
          <label key={r.id} className="cmp-pick">
            <input type="checkbox" checked={selected.has(r.id)} onChange={() => toggle(r.id)} />
            <div>
              <div className="nm">{r.model}</div>
              <div className="row" style={{ gap: 6, marginTop: 4 }}>
                <ModeTag mode={r.mode} />
                <span className="faint" style={{ fontSize: 11 }}>{r.dataset}</span>
              </div>
            </div>
          </label>
        ))}
      </Card>

      <div className="cmp-right">
        {picked.length === 0 ? (
          <Card><EmptyState icon="compare" title="Выберите прогоны" text="Отметьте слева успешные прогоны, чтобы сравнить их." /></Card>
        ) : (
          <>
            <Card style={{ minWidth: 0 }}>
              <div className="spread" style={{ marginBottom: 12 }}>
                <strong>Сводная таблица</strong>
                <select className="select-sm" value={groupBy} onChange={(e) => setGroupBy(e.target.value as GroupBy)}>
                  <option value="none">Без группировки</option>
                  <option value="mode">По режиму</option>
                  <option value="model">По модели</option>
                  <option value="dataset">По датасету</option>
                </select>
              </div>
              <div className="table-wrap">
                <table className="data">
                  <thead><tr><th>Прогон</th><th>Модель</th><th>Режим</th><th>Датасет</th><th>Test</th><th>Чеклист</th><th>Ошибки</th><th>Шагов</th><th>Токены</th><th>Время</th></tr></thead>
                  <tbody>
                    {sortedPicked.map((r) => (
                      <tr key={r.id} className="clickable" onClick={() => nav(`/runs/${r.id}`)}>
                        <td className="mono faint">{r.shortId}</td>
                        <td className="mono">{r.model}</td>
                        <td><ModeTag mode={r.mode} /></td>
                        <td>{r.dataset}</td>
                        <td>{r.score === bestScore ? <span className="best-pill">{formatScore(r.score, r.metric)}</span> : <span className="mono">{formatScore(r.score, r.metric)}</span>}</td>
                        <td>{r.checklist === bestChecklist ? <span className="best-pill">{r.checklist}/{r.checklistTotal}</span> : <span className="mono">{r.checklist}/{r.checklistTotal}</span>}</td>
                        <td>{r.errors === fewestErrors ? <span className="best-pill">{r.errors}</span> : <span className="mono">{r.errors}</span>}</td>
                        <td className="mono faint">{r.step}{r.steps ? `/${r.steps}` : ""}</td>
                        <td>{(r.tokIn + r.tokOut) === fewestTokens ? <span className="best-pill">{formatTokens(r.tokIn + r.tokOut)}</span> : <span className="mono">{formatTokens(r.tokIn + r.tokOut)}</span>}</td>
                        <td className="mono faint">{formatDuration(r.dur)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <div className="grid-2">
              <Card style={{ minWidth: 0 }}>
                <strong>Test-метрика</strong>
                <div className="faint" style={{ fontSize: 12, marginBottom: 8 }}>{datasets.size === 1 ? `датасет: ${[...datasets][0]}` : "выберите прогоны одного датасета для сопоставимой метрики"}</div>
                {barData.length ? <BarChart data={barData} fmt={(v) => formatScore(v, picked[0]?.metric)} /> : <EmptyState icon="compare" title="Разные датасеты" text="Метрики несопоставимы между датасетами." />}
              </Card>
              <Card style={{ minWidth: 0 }}>
                <strong>Цена за очко метрики</strong>
                <div className="faint" style={{ fontSize: 12, marginBottom: 8 }}>токены против улучшения над baseline (single-shot или слабейший на датасете)</div>
                {scatterPts.length ? <Scatter points={scatterPts} xLabel="токены" yLabel="улучшение %" /> : <EmptyState icon="coins" title="Недостаточно данных" text="Нужны успешные прогоны со скором." />}
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
