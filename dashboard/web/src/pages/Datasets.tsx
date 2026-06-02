import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Dataset, type DatasetStatus } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Modal, SelectDropdown, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { DatasetWizard } from "../components/datasets/DatasetWizard";

type SortKey = "az" | "za" | "createdDesc" | "createdAsc" | "updatedDesc" | "updatedAsc" | "rowsDesc" | "rowsAsc" | "colsDesc" | "colsAsc";

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

function DatasetCard({ d, onOpen, onDelete }: { d: Dataset; onOpen: () => void; onDelete: () => void }) {
  const status = statusOf(d);
  const taskLabel = TASK_LABEL[d.taskType ?? d.task] ?? d.task;
  return (
    <Card className="ds-card dataset-card-rich">
      <div className="spread">
        <div style={{ minWidth: 0 }}>
          <div className="ds-title">{d.name}</div>
        </div>
        <Tag tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Tag>
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
      <div className="ds-actions">
        <Button size="sm" icon="external" onClick={onOpen}>Открыть</Button>
        <Button size="sm" variant="ghost" icon="trash" onClick={onDelete}>Удалить</Button>
      </div>
    </Card>
  );
}

export default function Datasets() {
  const nav = useNavigate();
  const { data, loading, reload } = useAsync(() => api.listDatasets(), []);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [toDelete, setToDelete] = useState<Dataset | null>(null);
  const [busyDel, setBusyDel] = useState(false);
  const [query, setQuery] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [taskFilter, setTaskFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [metricFilter, setMetricFilter] = useState("all");
  const [sort, setSort] = useState<SortKey>("updatedDesc");
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
    return [...rows].sort((a, b) => {
      if (sort === "az") return a.name.localeCompare(b.name);
      if (sort === "za") return b.name.localeCompare(a.name);
      if (sort === "createdDesc") return time(b.createdAt) - time(a.createdAt);
      if (sort === "createdAsc") return time(a.createdAt) - time(b.createdAt);
      if (sort === "updatedDesc") return time(b.updatedAt) - time(a.updatedAt);
      if (sort === "updatedAsc") return time(a.updatedAt) - time(b.updatedAt);
      if (sort === "rowsDesc") return (b.rows ?? 0) - (a.rows ?? 0);
      if (sort === "rowsAsc") return (a.rows ?? 0) - (b.rows ?? 0);
      if (sort === "colsDesc") return (b.cols ?? 0) - (a.cols ?? 0);
      return (a.cols ?? 0) - (b.cols ?? 0);
    });
  }, [data, metricFilter, query, sort, statusFilter, taskFilter]);

  async function del() {
    if (!toDelete) return;
    setBusyDel(true);
    try {
      await api.deleteDataset(toDelete.id);
      setNotice(`Датасет ${toDelete.name} удален.`);
      setToDelete(null);
      reload();
    } finally {
      setBusyDel(false);
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
      <div className="spread dataset-center-head">
        <div />
        <Button variant="primary" icon="plus" onClick={() => setWizardOpen(true)}>Новая проблема</Button>
      </div>

      {notice && <div className="success-line"><Icon name="check" size={15} /> {notice}</div>}

      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Поиск по проблемам..." />
          </div>
          <SelectDropdown
            value={sort}
            onChange={setSort}
            options={[
              { value: "updatedDesc", label: "Сначала обновлённые" },
              { value: "updatedAsc",  label: "Давно не обновлялись" },
              { value: "createdDesc", label: "Сначала новые" },
              { value: "createdAsc",  label: "Сначала старые" },
              { value: "az",          label: "А → Я" },
              { value: "za",          label: "Я → А" },
              { value: "rowsDesc",    label: "Больше строк" },
              { value: "rowsAsc",     label: "Меньше строк" },
              { value: "colsDesc",    label: "Больше признаков" },
              { value: "colsAsc",     label: "Меньше признаков" },
            ]}
          />
          <Button variant="ghost" icon="sliders" onClick={() => setFiltersOpen((v) => !v)}>Фильтры</Button>
        </div>
        {filtersOpen && (
          <div className="dataset-filters-grid">
            <Field label="Задача">
              <select className="input" value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)}>
                <option value="all">все</option>
                <option value="classification">classification</option>
                <option value="regression">regression</option>
                <option value="unknown">unknown</option>
              </select>
            </Field>
            <Field label="Статус">
              <select className="input" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                <option value="all">все</option>
                <option value="prepared">подготовлен</option>
                <option value="partial">частичный</option>
                <option value="unprepared">не подготовлен</option>
              </select>
            </Field>
            <Field label="Метрика">
              <select className="input" value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)}>
                <option value="all">все</option>
                {metrics.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
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
            <DatasetCard key={d.id} d={d} onOpen={() => nav(`/problems/${d.id}`)} onDelete={() => setToDelete(d)} />
          ))}
        </div>
      )}

      {wizardOpen && <DatasetWizard onClose={() => setWizardOpen(false)} onCreated={created} />}
      {toDelete && (
        <Modal
          title="Удалить датасет"
          width={420}
          onClose={() => setToDelete(null)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setToDelete(null)}>Отмена</Button>
              <Button variant="danger" onClick={del} disabled={busyDel}>{busyDel ? <Spinner /> : "Удалить"}</Button>
            </>
          }
        >
          <p style={{ margin: 0 }}>
            Удалить <strong>{toDelete.name}</strong> и все исходные/подготовленные файлы? Это действие нельзя отменить.
          </p>
        </Modal>
      )}
    </div>
  );
}
