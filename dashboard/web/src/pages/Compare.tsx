import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Run, type RunMode } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS, formatDuration, formatScore, formatTokens, timeAgo } from "../lib/format";
import { Button, Card, EmptyState, SelectDropdown, Skeleton, StatusBadge } from "../components/ui";
import { Icon } from "../components/Icon";
import { BarChart } from "../components/charts";
import { ModeTag } from "../components/runbits";

type GroupBy = "none" | "mode" | "model" | "task";
type ModeFilter = RunMode | "any";
const MAX_PICK = 10;
const RUN_MODE_OPTIONS = (Object.keys(MODE_LABELS) as RunMode[]).filter((m) => m !== "batch");

function RunPickerModal({ successful, selected, onDone }: {
  successful: Run[];
  selected: Set<string>;
  onDone: (next: Set<string>) => void;
}) {
  const [local, setLocal] = useState(new Set(selected));
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<ModeFilter>("any");
  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return successful.filter((r) => {
      if (mode !== "any" && r.mode !== mode) return false;
      if (term && !`${r.shortId} ${r.model} ${r.task}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [successful, q, mode]);

  function toggle(id: string) {
    setLocal((s) => {
      const n = new Set(s);
      if (n.has(id)) { n.delete(id); return n; }
      if (n.size >= MAX_PICK) return s;
      n.add(id);
      return n;
    });
  }

  function pickTop10() {
    setLocal(new Set(filtered.slice(0, MAX_PICK).map((r) => r.id)));
  }

  return (
    <div className="modal-backdrop" onClick={() => onDone(selected)}>
      <div className="cmp-picker-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cmp-picker-head">
          <span style={{ fontWeight: 600, fontSize: 15 }}>Выбрать прогоны</span>
          <span className="faint" style={{ fontSize: 13 }}>{local.size} / {MAX_PICK}</span>
        </div>
        <div className="filters" style={{ margin: "12px 0 0", padding: "0 16px" }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" placeholder="Поиск по ID, модели, датасету…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <SelectDropdown
            value={mode}
            options={[{ value: "any", label: "Все режимы" }, ...RUN_MODE_OPTIONS.map((m) => ({ value: m, label: MODE_LABELS[m] }))]}
            onChange={(v) => setMode(v as ModeFilter)}
          />
          <Button variant="secondary" onClick={pickTop10}>Топ {MAX_PICK}</Button>
        </div>
        <div className="cmp-picker-list">
          {filtered.length === 0 && <div className="muted" style={{ padding: "24px 16px", fontSize: 13 }}>Ничего не найдено.</div>}
          {filtered.map((r) => {
            const checked = local.has(r.id);
            const disabled = !checked && local.size >= MAX_PICK;
            return (
              <label key={r.id} className={`cmp-picker-row${checked ? " selected" : ""}${disabled ? " disabled" : ""}`}>
                <input type="checkbox" checked={checked} disabled={disabled} onChange={() => toggle(r.id)} />
                <span className="mono faint" style={{ fontSize: 12, width: 80, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.shortId}</span>
                <span className="mono" style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.model}</span>
                <ModeTag mode={r.mode} />
                <span className="faint" style={{ fontSize: 12 }}>{r.task}</span>
                <StatusBadge status={r.status} />
                <span className="mono faint" style={{ fontSize: 12 }}>{formatScore(r.score, r.metric)}</span>
                <span className="faint" style={{ fontSize: 12 }}>{timeAgo(r.startedMs)}</span>
              </label>
            );
          })}
        </div>
        <div className="cmp-picker-foot">
          <Button variant="ghost" onClick={() => onDone(selected)}>Отмена</Button>
          <Button variant="primary" onClick={() => onDone(local)}>Применить ({local.size})</Button>
        </div>
      </div>
    </div>
  );
}

export default function Compare() {
  const nav = useNavigate();
  const { data: runs, loading } = useAsync(() => api.listRuns(), []);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [groupBy, setGroupBy] = useState<GroupBy>("none");
  const [pickerOpen, setPickerOpen] = useState(false);

  const successful = useMemo(() => (runs ?? []).filter((r) => r.score !== null), [runs]);

  const picked = successful.filter((r) => selected.has(r.id));

  const bestScore = Math.max(...picked.map((r) => r.score ?? -Infinity));
  const bestChecklist = Math.max(...picked.map((r) => r.checklist));
  const fewestErrors = Math.min(...picked.map((r) => r.errors));
  const fewestTokens = Math.min(...picked.map((r) => r.tokIn + r.tokOut));
  const pickedWithDuration = picked.filter((r): r is Run & { dur: number } => r.dur !== null && r.dur !== undefined);
  const shortestDuration = Math.min(...pickedWithDuration.map((r) => r.dur));

  const uniqueTasks = new Set(picked.map((r) => r.task));
  const scoredPicked = picked.filter((r) => r.score !== null && r.score !== undefined);
  const goalMin = (m?: string | null) =>
    !!(m && /rmse|rmsle|mae|mse|logloss/.test(m.toLowerCase()) && !m.toLowerCase().startsWith("neg_"));

  const sortedPicked = [...picked].sort((a, b) => {
    if (groupBy === "none") return 0;
    const key = (r: Run) => (groupBy === "mode" ? r.mode : groupBy === "model" ? r.model : r.task);
    return key(a).localeCompare(key(b));
  });

  if (loading && !runs) return <Skeleton h={400} />;

  return (
    <div className="compare">
      <Card className="cmp-list">
        <div className="spread" style={{ marginBottom: 8 }}>
          <strong style={{ fontSize: 14 }}>Прогоны</strong>
          <span className="faint" style={{ fontSize: 12 }}>{selected.size} / {MAX_PICK}</span>
        </div>
        {picked.length === 0 && (
          <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>Добавьте прогоны для сравнения.</div>
        )}
        {picked.map((r) => (
          <div key={r.id} className="cmp-pick">
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="nm">{r.model}</div>
              <div className="row" style={{ gap: 6, marginTop: 4 }}>
                <ModeTag mode={r.mode} />
                <span className="faint" style={{ fontSize: 11 }}>{r.task}</span>
              </div>
            </div>
            <button className="icon-btn" style={{ flexShrink: 0 }} onClick={() => setSelected((s) => { const n = new Set(s); n.delete(r.id); return n; })}>
              <Icon name="x" size={14} />
            </button>
          </div>
        ))}
        <button className="cmp-add-btn" onClick={() => setPickerOpen(true)}>
          <Icon name="plus" size={15} /> Выбрать прогоны
        </button>
      </Card>

      <div className="cmp-right">
        {picked.length === 0 ? (
          <Card><EmptyState icon="compare" title="Выберите прогоны" text="Нажмите «Выбрать прогоны» слева, чтобы добавить до 10 прогонов для сравнения." /></Card>
        ) : (
          <>
            <Card style={{ minWidth: 0 }}>
              <div className="spread" style={{ marginBottom: 12 }}>
                <strong>Сводная таблица</strong>
                <SelectDropdown
                  value={groupBy}
                  options={[
                    { value: "none", label: "Без группировки" },
                    { value: "mode", label: "По режиму" },
                    { value: "model", label: "По модели" },
                    { value: "task", label: "По задаче" },
                  ]}
                  onChange={(v) => setGroupBy(v as GroupBy)}
                />
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
                        <td>{r.task}</td>
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

            <Card>
              <strong>Test-метрика</strong>
              <div className="faint" style={{ fontSize: 12, marginBottom: 12 }}>{uniqueTasks.size === 1 ? `задача: ${[...uniqueTasks][0]}` : "выберите прогоны одной задачи для сопоставимой метрики"}</div>
              {scoredPicked.length ? (
                <div className="table-wrap">
                  <table className="data">
                    <thead><tr><th>Прогон</th><th>Модель</th><th>Режим</th><th>Датасет</th><th>Test-скор</th><th>Метрика</th></tr></thead>
                    <tbody>
                      {[...scoredPicked].sort((a, b) => (goalMin(a.metric) ? a.score! - b.score! : b.score! - a.score!)).map((r) => (
                        <tr key={r.id} className="clickable" onClick={() => nav(`/runs/${r.id}`)}>
                          <td className="mono faint">{r.shortId}</td>
                          <td className="mono">{r.model}</td>
                          <td><ModeTag mode={r.mode} /></td>
                          <td>{r.task}</td>
                          <td>{r.score === bestScore ? <span className="best-pill">{formatScore(r.score, r.metric)}</span> : <span className="mono">{formatScore(r.score, r.metric)}</span>}</td>
                          <td className="mono faint">{r.metric ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <EmptyState icon="compare" title="Нет данных" text="Нужны прогоны со скором." />}
            </Card>

            <Card>
              <strong>Токены</strong>
              <div className="faint" style={{ fontSize: 12, marginBottom: 8 }}>суммарный расход токенов на прогон</div>
              <BarChart
                data={[...picked].sort((a, b) => a.tokIn + a.tokOut - (b.tokIn + b.tokOut)).map((r) => ({ label: r.shortId, sub: r.model, value: r.tokIn + r.tokOut, best: (r.tokIn + r.tokOut) === fewestTokens }))}
                fmt={formatTokens}
              />
            </Card>

            <Card>
              <strong>Время</strong>
              <div className="faint" style={{ fontSize: 12, marginBottom: 8 }}>длительность прогона</div>
              <BarChart
                data={[...pickedWithDuration].sort((a, b) => a.dur - b.dur).map((r) => ({ label: r.shortId, sub: r.model, value: r.dur, best: r.dur === shortestDuration }))}
                fmt={formatDuration}
              />
            </Card>
          </>
        )}
      </div>

      {pickerOpen && (
        <RunPickerModal
          successful={successful}
          selected={selected}
          onDone={(next) => { setSelected(next); setPickerOpen(false); }}
        />
      )}
    </div>
  );
}
