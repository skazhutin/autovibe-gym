import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { api, type Task, type TaskStatus } from "../lib/api";
import { formatDateOnly } from "../lib/date";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Skeleton, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const STATUS_TONE: Record<TaskStatus, "green" | "blue" | "red"> = {
  prepared: "green", partial: "blue", unprepared: "red",
};
const STATUS_LABEL: Record<TaskStatus, string> = {
  prepared: "подготовлен", partial: "частичный", unprepared: "не подготовлен",
};
const TASK_LABEL: Record<string, string> = {
  classification: "classification", regression: "regression", auto: "auto", unknown: "unknown",
};
const METRIC_GOAL_LABEL: Record<string, string> = {
  max: "maximize",
  min: "minimize",
};

function statusOf(d: Task): TaskStatus {
  return d.status ?? (d.prepared ? "prepared" : d.hasTrain ? "partial" : "unprepared");
}

function splitTag(ok?: boolean, label?: string) {
  return <Tag tone={ok ? "green" : "neutral"}>{label}</Tag>;
}

function sourceText(d: Task) {
  const value = d.source && d.source !== "-" ? d.source : d.sources?.[0]?.name || d.sources?.[0]?.url || "-";
  return !value || value === "source" ? "-" : value;
}

function TaskCard({ d, dateFormat, onOpen, selecting, isSelected, onToggle }: { d: Task; dateFormat: "mdy" | "dmy"; onOpen: () => void; selecting: boolean; isSelected: boolean; onToggle: () => void }) {
  const status = statusOf(d);
  const taskLabel = TASK_LABEL[d.taskType ?? d.task] ?? d.task;
  return (
    <Card className={`ds-card task-card-rich${selecting && isSelected ? " row-selected" : ""}`} hover onClick={() => selecting ? onToggle() : onOpen()}>
      <div className="spread">
        <div style={{ minWidth: 0 }}>
          <div className="ds-title">{d.name}</div>
        </div>
        <Tag tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Tag>
      </div>
      {d.desc && <div className="muted clamp-2">{d.desc}</div>}
      <div className="run-meta-line" style={{ margin: 0 }}>
        <Tag tone={d.taskType === "regression" ? "blue" : d.taskType === "classification" ? "accent" : "neutral"}>{taskLabel}</Tag>
        <Tag>{METRIC_GOAL_LABEL[d.metricGoal ?? "max"] ?? d.metricGoal}</Tag>
        {(d.tags ?? []).slice(0, 3).map((tag) => <Tag key={tag} tone="neutral">{tag}</Tag>)}
        {d.warningsCount ? <Tag tone="red">{d.warningsCount} warnings</Tag> : null}
      </div>
      <div className="ds-stats rich">
        <div className="ds-stat"><span className="k">rows</span><span className="v">{d.rows ? d.rows.toLocaleString() : "-"}</span></div>
        <div className="ds-stat"><span className="k">features</span><span className="v">{d.cols || "-"}</span></div>
        <div className="ds-stat"><span className="k">target</span><span className="v">{d.target}</span></div>
        <div className="ds-stat"><span className="k">metric</span><span className="v">{d.metric}</span></div>
        <div className="ds-stat"><span className="k">seed</span><span className="v">{d.seed ?? 42}</span></div>
        <div className="ds-stat"><span className="k">source</span><span className="v">{sourceText(d)}</span></div>
      </div>
      <div className="split-pills">
        {splitTag(d.hasTrain, "train")}
        {splitTag(d.hasVal, "val")}
        {splitTag(d.hasTest, "test")}
      </div>
      <div className="ds-stats task-dates">
        {d.createdAt && <div className="ds-stat"><span className="k">создана</span><span className="v">{formatDateOnly(d.createdAt, dateFormat)}</span></div>}
        {d.updatedAt && <div className="ds-stat"><span className="k">обновлена</span><span className="v">{formatDateOnly(d.updatedAt, dateFormat)}</span></div>}
      </div>
    </Card>
  );
}

function ConfirmModal({ count, onConfirm, onCancel, busy }: { count: number; onConfirm: () => void; onCancel: () => void; busy: boolean }) {
  return createPortal(
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">Вернуть задачи?</h3>
        <p className="modal-desc">
          {count === 1 ? "1 задача будет возвращена из архива." : `${count} задач будут возвращены из архива.`}
        </p>
        <div className="modal-actions">
          <Button variant="secondary" onClick={onCancel} disabled={busy}>Отменить</Button>
          <Button variant="primary" onClick={onConfirm} disabled={busy}>{busy ? "Возврат…" : "Вернуть"}</Button>
        </div>
      </div>
    </div>,
    document.body
  );
}

export default function DatasetsArchive() {
  const nav = useNavigate();
  const { data: tasks, loading, reload } = useAsync(() => api.listArchivedTasks(), []);
  const { data: settings } = useAsync(() => api.getSettings(), []);
  const [q, setQ] = useState("");
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState(false);
  const [restoring, setRestoring] = useState(false);

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (tasks ?? []).filter((d) =>
      !term || `${d.name} ${d.task} ${d.metric}`.toLowerCase().includes(term)
    );
  }, [tasks, q]);

  function toggleSelect(id: string) {
    setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  const allFilteredSelected = filtered.length > 0 && filtered.every((d) => selected.has(d.id));

  function toggleSelectAll() {
    setSelected((s) => {
      const n = new Set(s);
      if (allFilteredSelected) filtered.forEach((d) => n.delete(d.id));
      else filtered.forEach((d) => n.add(d.id));
      return n;
    });
  }

  function cancelSelect() { setSelecting(false); setSelected(new Set()); }

  async function doRestore() {
    setRestoring(true);
    try {
      await api.unarchiveTasks([...selected]);
      setConfirm(false);
      cancelSelect();
      reload();
    } finally { setRestoring(false); }
  }

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button className="archive-link" onClick={() => nav("/problems")}>
          <Icon name="chevronLeft" size={15} /> Назад к задачам
        </button>
      </div>

      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={17} />
            <input className="input" placeholder="Поиск по названию, задаче, метрике…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          {selecting ? (
            <>
              <Button variant="secondary" onClick={toggleSelectAll}>{allFilteredSelected ? "Снять выделение" : "Выбрать все"}</Button>
              <Button variant="secondary" onClick={cancelSelect}>Отмена</Button>
            </>
          ) : (
            <Button variant="secondary" onClick={() => setSelecting(true)}>Выбрать</Button>
          )}
        </div>
      </Card>

      {loading && !tasks ? (
        <Skeleton h={200} />
      ) : filtered.length ? (
        <div className="task-grid">
          {filtered.map((d) => (
            <TaskCard
              key={d.id}
              d={d}
              dateFormat={settings?.date_format ?? "mdy"}
              onOpen={() => nav(`/problems/${d.id}`)}
              selecting={selecting}
              isSelected={selected.has(d.id)}
              onToggle={() => toggleSelect(d.id)}
            />
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState icon="archive" title="Архив пуст" text="Архивированные задачи появятся здесь." />
        </Card>
      )}

      {selecting && selected.size > 0 && createPortal(
        <div className="selection-bar">
          <span className="selection-bar-label">Выбрано {selected.size} задач{selected.size === 1 ? "а" : selected.size < 5 ? "и" : ""}</span>
          <Button variant="primary" onClick={() => setConfirm(true)}>
            <Icon name="undo" size={15} /> Вернуть
          </Button>
          <Button variant="secondary" onClick={cancelSelect}>Отменить</Button>
        </div>,
        document.body
      )}

      {confirm && <ConfirmModal count={selected.size} onConfirm={doRestore} onCancel={() => setConfirm(false)} busy={restoring} />}
    </div>
  );
}
