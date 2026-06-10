import { useEffect, useMemo, useState } from "react";
import { useNavigate, useOutletContext } from "react-router-dom";
import type { SetHeaderAction } from "../components/Layout";
import { api, type Task } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { createPortal } from "react-dom";
import { Button, Card, EmptyState, Field, SelectDropdown, Skeleton } from "../components/ui";
import { Icon } from "../components/Icon";
import { TaskWizard } from "../components/tasks/TaskWizard";
import { TaskCard } from "../components/tasks/TaskCard";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { SelectionBar } from "../components/SelectionBar";
import { statusOf } from "../lib/taskUtils";

type SortField = "updated" | "created" | "az";
type SortDir = "asc" | "desc";

function textBlob(d: Task) {
  return [
    d.name, d.id, d.target, d.metric, d.source, d.desc,
    ...(d.tags ?? []),
    ...(d.sources ?? []).flatMap((s) => [s.name, s.url, s.license, s.organization]),
  ].filter(Boolean).join(" ").replace(/[_-]+/g, " ").toLowerCase();
}

export default function Tasks() {
  const setHeaderAction = useOutletContext<SetHeaderAction>();
  const nav = useNavigate();
  const { data, loading, reload } = useAsync(() => api.listTasks(), []);
  const { data: settings } = useAsync(() => api.getSettings(), []);
  const [wizardOpen, setWizardOpen] = useState(false);

  useEffect(() => {
    setHeaderAction({ label: "Новая задача", icon: "plus", onClick: () => setWizardOpen(true) });
    return () => setHeaderAction(null);
  }, [setHeaderAction]);
  const [toArchive, setToArchive] = useState<Task | null>(null);
  const [busyArchive, setBusyArchive] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [busyBulk, setBusyBulk] = useState(false);
  const [query, setQuery] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const [taskFilter, setTaskFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [metricFilter, setMetricFilter] = useState("all");
  const [sortField, setSortField] = useState<SortField>("updated");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [notice, setNotice] = useState<string | null>(null);

  const metrics = useMemo(() => Array.from(new Set((data ?? []).map((d) => d.metric).filter(Boolean))).sort(), [data]);
  const filtered = useMemo(() => {
    const q = query.trim().replace(/[_-]+/g, " ").toLowerCase();
    const rows = (data ?? []).filter((d) => {
      const status = statusOf(d);
      if (q && !textBlob(d).includes(q)) return false;
      if (taskFilter !== "all" && (d.taskType ?? "unknown") !== taskFilter) return false;
      if (statusFilter !== "all" && status !== statusFilter) return false;
      if (metricFilter !== "all" && d.metric !== metricFilter) return false;
      return true;
    });
    const time = (v?: string | null) => (v ? new Date(v).getTime() : 0);
    const asc = sortDir === "asc";
    return [...rows].sort((a, b) => {
      if (sortField === "az") return asc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
      if (sortField === "created") return asc ? time(a.createdAt) - time(b.createdAt) : time(b.createdAt) - time(a.createdAt);
      return asc ? time(a.updatedAt) - time(b.updatedAt) : time(b.updatedAt) - time(a.updatedAt);
    });
  }, [data, metricFilter, query, sortField, sortDir, statusFilter, taskFilter]);

  const allFilteredSelected = filtered.length > 0 && filtered.every((d) => selected.has(d.id));
  function toggleSelect(id: string) { setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; }); }
  function toggleSelectAll() { setSelected((s) => { const n = new Set(s); if (allFilteredSelected) filtered.forEach((d) => n.delete(d.id)); else filtered.forEach((d) => n.add(d.id)); return n; }); }
  function cancelSelect() { setSelecting(false); setSelected(new Set()); }

  async function doBulkArchive() {
    setBusyBulk(true);
    try { await api.archiveTasks([...selected]); setConfirmBulk(false); cancelSelect(); reload(); }
    finally { setBusyBulk(false); }
  }

  async function doArchive() {
    if (!toArchive) return;
    setBusyArchive(true);
    try {
      await api.archiveTasks([toArchive.id]);
      setNotice(`Задача ${toArchive.name} перемещена в архив.`);
      setToArchive(null);
      reload();
    } finally {
      setBusyArchive(false);
    }
  }

  function created(ds: Task) {
    setWizardOpen(false);
    setNotice(`Датасет ${ds.name} создан.`);
    reload();
    nav(`/problems/${ds.id}`);
  }

  if (loading && !data) {
    return <div className="grid-3">{[0, 1, 2].map((i) => <Card key={i}><Skeleton h={180} /></Card>)}</div>;
  }

  const toolbarEl = document.getElementById("toolbar-portal");

  return (
    <div className="stack" style={{ gap: 18 }}>
      {notice && <div className="success-line"><Icon name="check" size={15} /> {notice}</div>}

      {toolbarEl && createPortal(<Card className="dataset-toolbar">
        <div className="filters tasks-toolbar" style={{ marginBottom: 0 }}>
          <div className="search tasks-toolbar-search">
            <Icon name="search" size={16} />
            <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Поиск по задачам..." />
          </div>
          <div className="tasks-toolbar-controls">
            <Button variant={sortOpen ? "primary" : "ghost"} icon="arrowUpDown" onClick={() => { setSortOpen((v) => !v); setFiltersOpen(false); }}>Сортировка</Button>
            <Button variant={filtersOpen ? "primary" : "ghost"} icon="sliders" onClick={() => { setFiltersOpen((v) => !v); setSortOpen(false); }}>Фильтры</Button>
            <Button variant={selecting ? "primary" : "secondary"} onClick={() => setSelecting((v) => !v)} style={{ width: 96 }}>{selecting ? "Готово" : "Выбрать"}</Button>
          </div>
        </div>
        {sortOpen && (
          <div className="dataset-filters-grid">
            <div style={{ gridColumn: "1 / -1" }}><Field label="Сортировать по:">
              <div className="sort-tabs" style={{ width: "100%" }}>
                {(["updated", "created", "az"] as SortField[]).map((f) => (
                  <button key={f} className={`sort-tab${sortField === f ? " active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                    onClick={() => { if (sortField === f) setSortDir((d) => d === "asc" ? "desc" : "asc"); else { setSortField(f); setSortDir("desc"); } }}>
                    {f === "updated" ? "Обновление" : f === "created" ? "Создание" : "Алфавит"}
                    {sortField === f && <Icon name={sortDir === "desc" ? "arrowDown" : "arrowUp"} size={13} />}
                  </button>
                ))}
              </div>
            </Field></div>
          </div>
        )}
        {filtersOpen && (
          <div className="dataset-filters-grid">
            <Field label="Задача">
              <SelectDropdown
                value={taskFilter}
                options={[
                  { value: "all", label: "Все" },
                  { value: "classification", label: "Classification" },
                  { value: "regression", label: "Regression" },
                  { value: "unknown", label: "Unknown" },
                ]}
                onChange={setTaskFilter}
              />
            </Field>
            <Field label="Статус">
              <SelectDropdown
                value={statusFilter}
                options={[
                  { value: "all", label: "Все" },
                  { value: "prepared", label: "Подготовлен" },
                  { value: "partial", label: "Частичный" },
                  { value: "unprepared", label: "Не подготовлен" },
                ]}
                onChange={setStatusFilter}
              />
            </Field>
            <Field label="Метрика">
              <SelectDropdown
                value={metricFilter}
                options={[{ value: "all", label: "Все" }, ...metrics.map((m) => ({ value: m, label: m }))]}
                onChange={setMetricFilter}
              />
            </Field>
          </div>
        )}
        {selecting && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 14, borderTop: "1px solid var(--border)", marginTop: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-dim)", letterSpacing: "0.04em", textTransform: "uppercase" }}>Режим редактирования</div>
            <div style={{ display: "flex", gap: 10 }}>
              <Button variant="secondary" onClick={toggleSelectAll}>{allFilteredSelected ? "Снять выделение" : "Выбрать все"}</Button>
              <Button variant="secondary" onClick={() => setSelected(new Set())}>Сбросить все</Button>
            </div>
          </div>
        )}
      </Card>, toolbarEl)}

      {!filtered.length ? (
        <EmptyState
          icon="database"
          title="Датасеты не найдены"
          text="Измените поиск или фильтры либо создайте новый датасет из исходной таблицы или подготовленных файлов."
          action={<Button variant="primary" icon="plus" onClick={() => setWizardOpen(true)}>Добавить датасет</Button>}
        />
      ) : (
        <div className="task-grid">
          {filtered.map((d) => (
            <TaskCard key={d.id} d={d} dateFormat={settings?.date_format ?? "mdy"} onOpen={() => nav(`/problems/${d.id}`)} selecting={selecting} isSelected={selected.has(d.id)} onToggle={() => toggleSelect(d.id)} />
          ))}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button className="archive-link" onClick={() => nav("/problems/archive")}>
          <Icon name="archive" size={15} /> Архив
        </button>
      </div>

      {selecting && (
        <SelectionBar
          count={selected.size}
          noun="задача"
          actionLabel="Архивировать"
          actionIcon="archive"
          busy={busyBulk}
          onAction={() => setConfirmBulk(true)}
          onCancel={cancelSelect}
        />
      )}

      {confirmBulk && (
        <ConfirmDialog
          title="Архивировать задачи?"
          description={`${selected.size === 1 ? "1 задача будет перемещена в архив." : `${selected.size} задач будут перемещены в архив.`} Вернуть можно из раздела «Архив».`}
          confirmLabel="Архивировать"
          busy={busyBulk}
          onConfirm={doBulkArchive}
          onCancel={() => setConfirmBulk(false)}
        />
      )}

      {wizardOpen && <TaskWizard onClose={() => setWizardOpen(false)} onCreated={created} />}
      {toArchive && (
        <ConfirmDialog
          title="Архивировать задачу?"
          description={`"${toArchive.name}" будет перемещена в архив. Вернуть можно из раздела «Архив».`}
          confirmLabel="Архивировать"
          busy={busyArchive}
          onConfirm={doArchive}
          onCancel={() => setToArchive(null)}
        />
      )}
    </div>
  );
}
