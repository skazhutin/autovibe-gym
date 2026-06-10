import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Skeleton } from "../components/ui";
import { Icon } from "../components/Icon";
import { TaskCard } from "../components/tasks/TaskCard";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { SelectionBar } from "../components/SelectionBar";

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

  const toolbarEl = document.getElementById("toolbar-portal");

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button className="archive-link" onClick={() => nav("/problems")}>
          <Icon name="chevronLeft" size={15} /> Назад к задачам
        </button>
      </div>

      {toolbarEl && createPortal(<Card className="dataset-toolbar">
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
      </Card>, toolbarEl)}

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

      {selecting && selected.size > 0 && (
        <SelectionBar
          count={selected.size}
          noun="задача"
          actionLabel="Вернуть"
          actionIcon="undo"
          busy={restoring}
          onAction={() => setConfirm(true)}
          onCancel={cancelSelect}
        />
      )}

      {confirm && (
        <ConfirmDialog
          title="Вернуть задачи?"
          description={selected.size === 1 ? "1 задача будет возвращена из архива." : `${selected.size} задач будут возвращены из архива.`}
          confirmLabel="Вернуть"
          busy={restoring}
          onConfirm={doRestore}
          onCancel={() => setConfirm(false)}
        />
      )}
    </div>
  );
}
