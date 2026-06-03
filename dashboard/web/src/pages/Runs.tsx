import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Run, type RunMode, type RunStatus } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS, STATUS_LABELS, formatTokens, timeAgo } from "../lib/format";
import { Button, Card, EmptyState, Skeleton, StatusBadge } from "../components/ui";
import { Icon } from "../components/Icon";
import { ModeTag, ScoreCell } from "../components/runbits";

type ModeFilter = RunMode | "any";
const RUN_MODE_OPTIONS = Object.keys(MODE_LABELS) as RunMode[];

export default function Runs() {
  const nav = useNavigate();
  const { data: runs, loading } = useAsync(() => api.listRuns(), [], 5000);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<ModeFilter>("any");
  const [status, setStatus] = useState<RunStatus | "all">("all");

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (runs ?? []).filter((r) => {
      if (mode !== "any" && r.mode !== mode) {
        return false;
      }
      if (status !== "all" && r.status !== status) return false;
      if (term && !`${r.shortId} ${r.model} ${r.dataset} ${r.batchId ?? ""}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [runs, q, mode, status]);

  return (
    <div>
      <div className="filters">
        <div className="search">
          <Icon name="search" size={17} />
          <input className="input" placeholder="Поиск по ID, модели, датасету…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <select className="select-sm" value={mode} onChange={(e) => setMode(e.target.value as ModeFilter)}>
          <option value="any">Все режимы</option>
          {RUN_MODE_OPTIONS.map((m) => <option key={m} value={m}>{MODE_LABELS[m]}</option>)}
        </select>
        <select className="select-sm" value={status} onChange={(e) => setStatus(e.target.value as RunStatus | "all")}>
          <option value="all">Все статусы</option>
          {(Object.keys(STATUS_LABELS) as RunStatus[]).map((s) => <option key={s} value={s}>{STATUS_LABELS[s]}</option>)}
        </select>
      </div>

      <Card style={{ padding: 0 }}>
        {loading && !runs ? (
          <div style={{ padding: 20 }}><Skeleton h={200} /></div>
        ) : filtered.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr><th>ID</th><th>Модель</th><th>Режим</th><th>Датасет</th><th>Скор</th><th>Статус</th><th>Шагов</th><th>Токены</th><th>Когда</th><th></th></tr>
              </thead>
              <tbody>
                {filtered.map((r: Run) => (
                  <tr key={r.id} className="clickable" onClick={() => nav(`/runs/${r.id}`)}>
                    <td className="mono faint">{r.shortId}</td>
                    <td className="mono">{r.model}</td>
                    <td><ModeTag mode={r.mode} /></td>
                    <td>{r.dataset}</td>
                    <td><ScoreCell run={r} /></td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="mono faint">{r.step}{r.steps ? `/${r.steps}` : ""}</td>
                    <td className="mono faint">{formatTokens(r.tokIn + r.tokOut)}</td>
                    <td className="faint">{timeAgo(r.startedMs)}</td>
                    <td className="faint"><Icon name="chevronRight" size={16} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon="runs"
            title={runs && runs.length ? "Ничего не найдено" : "Прогонов пока нет"}
            text={runs && runs.length ? "Измените фильтры или поисковый запрос." : "Запустите первый прогон, чтобы он появился в истории."}
            action={<Button variant="primary" icon="plus" onClick={() => nav("/new")}>Новый прогон</Button>}
          />
        )}
      </Card>
    </div>
  );
}
