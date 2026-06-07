import { useEffect, useMemo, useState } from "react";
import { useNavigate, useOutletContext } from "react-router-dom";
import type { SetHeaderAction } from "../components/Layout";
import { api, type Dataset, type DatasetStatus } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Modal, SelectDropdown, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { DatasetWizard } from "../components/datasets/DatasetWizard";

type SortField = "updated" | "created" | "az";
type SortDir = "asc" | "desc";

const STATUS_TONE: Record<DatasetStatus, "green" | "blue" | "red"> = {
  prepared: "green",
  partial: "blue",
  unprepared: "red",
};
const STATUS_LABEL: Record<DatasetStatus, string> = {
  prepared: "prepared",
  partial: "partial",
  unprepared: "unprepared",
};
const TASK_LABEL: Record<string, string> = {
  classification: "classification",
  regression: "regression",
  auto: "auto",
  unknown: "unknown",
};
const METRIC_GOAL_LABEL: Record<string, string> = {
  max: "maximize",
  min: "minimize",
};

function statusOf(d: Dataset): DatasetStatus {
  return d.status ?? (d.prepared ? "prepared" : d.hasTrain ? "partial" : "unprepared");
}

function splitTag(ok?: boolean, label?: string) {
  return <Tag tone={ok ? "green" : "neutral"} mono>{label}</Tag>;
}

function sourceText(d: Dataset) {
  const value = d.source && d.source !== "-" ? d.source : d.sources?.[0]?.name || d.sources?.[0]?.url || "-";
  return !value || value === "source" ? "-" : value;
}

function textBlob(d: Dataset) {
  return [
    d.name,
    d.id,
    d.target,
    d.metric,
    d.source,
    d.desc,
    ...(d.tags ?? []),
    ...(d.sources ?? []).flatMap((s) => [s.name, s.url, s.license, s.organization]),
  ]
    .filter(Boolean)
    .join(" ")
    .replace(/[_-]+/g, " ")
    .toLowerCase();
}

function DatasetCard({ d, onOpen, onArchive }: { d: Dataset; onOpen: () => void; onArchive: () => void }) {
  const status = statusOf(d);
  const taskLabel = TASK_LABEL[d.taskType ?? d.task] ?? d.task;
  return (
    <Card className="ds-card dataset-card-rich" hover onClick={onOpen}>
      <div className="spread">
        <div style={{ minWidth: 0 }}>
          <div className="ds-title">{d.name}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Tag tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Tag>
          <button className="icon-btn ds-delete-btn" title="Архивировать" onClick={(e) => { e.stopPropagation(); onArchive(); }}>
            <Icon name="archive" size={15} />
          </button>
        </div>
      </div>
      {d.desc && <div className="muted clamp-2">{d.desc}</div>}
      <div className="run-meta-line" style={{ margin: 0 }}>
        <Tag tone={d.taskType === "regression" ? "blue" : d.taskType === "classification" ? "accent" : "neutral"}>{taskLabel}</Tag>
        <Tag mono>{METRIC_GOAL_LABEL[d.metricGoal ?? "max"] ?? d.metricGoal}</Tag>
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
      <div className="faint dataset-dates">
        {d.createdAt && <span>создан {new Date(d.createdAt).toLocaleString()}</span>}
        {d.updatedAt && <span>обновлен {new Date(d.updatedAt).toLocaleString()}</span>}
      </div>
    </Card>
  );
}

export default function Datasets() {
  const setHeaderAction = useOutletContext<SetHeaderAction>();
  const nav = useNavigate();
  const { data, loading, reload } = useAsync(() => api.listDatasets(), []);
  const [wizardOpen, setWizardOpen] = useState(false);

  useEffect(() => {
    setHeaderAction({ label: "Новая проблема", icon: "plus", onClick: () => setWizardOpen(true) });
    return () => setHeaderAction(null);
  }, [setHeaderAction]);
  const [toArchive, setToArchive] = useState<Dataset | null>(null);
  const [busyArchive, setBusyArchive] = useState(false);
  const [query, setQuery] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
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

  async function doArchive() {
    if (!toArchive) return;
    setBusyArchive(true);
    try {
      await api.archiveDatasets([toArchive.id]);
      setNotice(`Проблема ${toArchive.name} перемещена в архив.`);
      setToArchive(null);
      reload();
    } finally {
      setBusyArchive(false);
    }
  }

  function created(ds: Dataset) {
    setWizardOpen(false);
    setNotice(`Датасет ${ds.name} создан.`);
    reload();
    nav(`/problems/${ds.id}`);
  }

  if (loading && !data) {
    return <div className="grid-3">{[0, 1, 2].map((i) => <Card key={i}><Skeleton h={180} /></Card>)}</div>;
  }

  return (
    <div className="stack" style={{ gap: 18 }}>
      {notice && <div className="success-line"><Icon name="check" size={15} /> {notice}</div>}

      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Поиск по проблемам..." />
          </div>
          <div className="sort-tabs">
            {(["updated", "created", "az"] as SortField[]).map((f) => (
              <button
                key={f}
                className={`sort-tab${sortField === f ? " active" : ""}`}
                onClick={() => { if (sortField === f) setSortDir((d) => d === "asc" ? "desc" : "asc"); else { setSortField(f); setSortDir("desc"); } }}
              >
                {f === "updated" ? "Обновление" : f === "created" ? "Создание" : "Алфавит"}
                {sortField === f && <Icon name={sortDir === "desc" ? "arrowDown" : "arrowUp"} size={13} />}
              </button>
            ))}
          </div>
          <Button variant="ghost" icon="sliders" onClick={() => setFiltersOpen((v) => !v)}>Фильтры</Button>
        </div>
        {filtersOpen && (
          <div className="dataset-filters-grid">
            <Field label="Задача">
              <SelectDropdown
                value={taskFilter}
                options={[
                  { value: "all", label: "все" },
                  { value: "classification", label: "classification" },
                  { value: "regression", label: "regression" },
                  { value: "unknown", label: "unknown" },
                ]}
                onChange={setTaskFilter}
              />
            </Field>
            <Field label="Статус">
              <SelectDropdown
                value={statusFilter}
                options={[
                  { value: "all", label: "все" },
                  { value: "prepared", label: "подготовлен" },
                  { value: "partial", label: "частичный" },
                  { value: "unprepared", label: "не подготовлен" },
                ]}
                onChange={setStatusFilter}
              />
            </Field>
            <Field label="Метрика">
              <SelectDropdown
                value={metricFilter}
                options={[{ value: "all", label: "все" }, ...metrics.map((m) => ({ value: m, label: m }))]}
                onChange={setMetricFilter}
              />
            </Field>
          </div>
        )}
      </Card>

      {!filtered.length ? (
        <EmptyState
          icon="database"
          title="Датасеты не найдены"
          text="Измените поиск или фильтры либо создайте новый датасет из исходной таблицы или подготовленных файлов."
          action={<Button variant="primary" icon="plus" onClick={() => setWizardOpen(true)}>Добавить датасет</Button>}
        />
      ) : (
        <div className="dataset-grid">
          {filtered.map((d) => (
            <DatasetCard key={d.id} d={d} onOpen={() => nav(`/problems/${d.id}`)} onArchive={() => setToArchive(d)} />
          ))}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button className="archive-link" onClick={() => nav("/problems/archive")}>
          <Icon name="archive" size={15} /> Архив
        </button>
      </div>

      {wizardOpen && <DatasetWizard onClose={() => setWizardOpen(false)} onCreated={created} />}
      {toArchive && (
        <Modal
          title="Архивировать проблему?"
          width={420}
          onClose={() => setToArchive(null)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setToArchive(null)}>Отмена</Button>
              <Button variant="primary" onClick={doArchive} disabled={busyArchive}>{busyArchive ? <Spinner /> : "Архивировать"}</Button>
            </>
          }
        >
          <p className="modal-desc">
            <strong>{toArchive.name}</strong> будет перемещена в архив. Вернуть можно из раздела «Архив».
          </p>
        </Modal>
      )}
    </div>
  );
}
