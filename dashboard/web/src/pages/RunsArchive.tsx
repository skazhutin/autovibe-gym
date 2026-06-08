import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { api, type Run, type RunMode, type RunStatus } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { MODE_LABELS, STATUS_LABELS, formatTokens, timeAgo } from "../lib/format";
import { Button, Card, EmptyState, SelectDropdown, Skeleton, StatusBadge } from "../components/ui";
import { Icon } from "../components/Icon";
import { ModeTag, ScoreCell } from "../components/runbits";

type ModeFilter = RunMode | "any";
const RUN_MODE_OPTIONS = Object.keys(MODE_LABELS) as RunMode[];

function ConfirmModal({ count, onConfirm, onCancel, busy }: { count: number; onConfirm: () => void; onCancel: () => void; busy: boolean }) {
  return createPortal(
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">Вернуть прогоны?</h3>
        <p className="modal-desc">
          {count === 1 ? "1 прогон будет возвращён из архива." : `${count} прогонов будут возвращены из архива.`}
        </p>
        <div className="modal-actions">
          <Button variant="secondary" onClick={onCancel} disabled={busy}>Отменить</Button>
          <Button variant="primary" onClick={onConfirm} disabled={busy}>
            {busy ? "Возвращение…" : "Вернуть"}
          </Button>
        </div>
      </div>
    </div>,
    document.body
  );
}

export default function RunsArchive() {
  const nav = useNavigate();
  const { data: runs, loading, reload } = useAsync(() => api.listArchivedRuns(), [], 10000);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<ModeFilter>("any");
  const [status, setStatus] = useState<RunStatus | "all">("all");

  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState(false);
  const [restoring, setRestoring] = useState(false);

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (runs ?? []).filter((r) => {
      if (mode !== "any" && r.mode !== mode) return false;
      if (status !== "all" && r.status !== status) return false;
      if (term && !`${r.shortId} ${r.model} ${r.task} ${r.batchId ?? ""}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [runs, q, mode, status]);

  function toggleSelect(id: string) {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  const allFilteredSelected = filtered.length > 0 && filtered.every((r) => selected.has(r.id));

  function toggleSelectAll() {
    setSelected((s) => {
      const next = new Set(s);
      if (allFilteredSelected) {
        filtered.forEach((r) => next.delete(r.id));
      } else {
        filtered.forEach((r) => next.add(r.id));
      }
      return next;
    });
  }

  function cancelSelect() {
    setSelecting(false);
    setSelected(new Set());
  }

  async function doRestore() {
    setRestoring(true);
    try {
      await api.unarchiveRuns([...selected]);
      setConfirm(false);
      cancelSelect();
      reload();
    } finally {
      setRestoring(false);
    }
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <button className="archive-link" onClick={() => nav("/runs")}>
          <Icon name="chevronLeft" size={15} /> Назад к прогонам
        </button>
      </div>

      <div className="filters">
        <div className="search">
          <Icon name="search" size={17} />
          <input className="input" placeholder="Поиск по ID, модели, датасету…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <SelectDropdown
          value={mode}
          options={[{ value: "any", label: "Все режимы" }, ...RUN_MODE_OPTIONS.map((m) => ({ value: m, label: MODE_LABELS[m] }))]}
          onChange={(v) => setMode(v as ModeFilter)}
        />
        <SelectDropdown
          value={status}
          options={[{ value: "all", label: "Все статусы" }, ...(Object.keys(STATUS_LABELS) as RunStatus[]).map((s) => ({ value: s, label: STATUS_LABELS[s] }))]}
          onChange={(v) => setStatus(v as RunStatus | "all")}
        />
        {selecting ? (
          <>
            <Button variant="secondary" onClick={toggleSelectAll}>
              {allFilteredSelected ? "Снять выделение" : "Выбрать все"}
            </Button>
            <Button variant="secondary" onClick={cancelSelect}>Отмена</Button>
          </>
        ) : (
          <Button variant="secondary" onClick={() => setSelecting(true)}>Выбрать</Button>
        )}
      </div>

      <Card style={{ padding: 0 }}>
        {loading && !runs ? (
          <div style={{ padding: 20 }}><Skeleton h={200} /></div>
        ) : filtered.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  {selecting && <th style={{ width: 36 }} />}
                  <th>ID</th><th>Модель</th><th>Режим</th><th>Датасет</th><th>Скор</th><th>Статус</th><th>Шагов</th><th>Токены</th><th>Когда</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r: Run) => (
                  <tr
                    key={r.id}
                    className={`clickable${selecting && selected.has(r.id) ? " row-selected" : ""}`}
                    onClick={() => selecting ? toggleSelect(r.id) : nav(`/runs/${r.id}`)}
                  >
                    {selecting && (
                      <td onClick={(e) => { e.stopPropagation(); toggleSelect(r.id); }}>
                        <input type="checkbox" className="row-checkbox" checked={selected.has(r.id)} onChange={() => toggleSelect(r.id)} />
                      </td>
                    )}
                    <td className="mono faint">{r.shortId}</td>
                    <td className="mono">{r.model}</td>
                    <td><ModeTag mode={r.mode} /></td>
                    <td>{r.task}</td>
                    <td><ScoreCell run={r} /></td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="mono faint">{r.step}{r.steps ? `/${r.steps}` : ""}</td>
                    <td className="mono faint">{formatTokens(r.tokIn + r.tokOut)}</td>
                    <td className="faint">{timeAgo(r.startedMs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon="runs"
            title="Архив пуст"
            text="Архивированные прогоны появятся здесь."
          />
        )}
      </Card>

      {selecting && selected.size > 0 && createPortal(
        <div className="selection-bar">
          <span className="selection-bar-label">Выбрано {selected.size} прогон{selected.size === 1 ? "" : selected.size < 5 ? "а" : "ов"}</span>
          <Button variant="primary" onClick={() => setConfirm(true)}>
            <Icon name="undo" size={15} /> Вернуть
          </Button>
          <Button variant="secondary" onClick={cancelSelect}>Отменить</Button>
        </div>,
        document.body
      )}

      {confirm && (
        <ConfirmModal count={selected.size} onConfirm={doRestore} onCancel={() => setConfirm(false)} busy={restoring} />
      )}
    </div>
  );
}
