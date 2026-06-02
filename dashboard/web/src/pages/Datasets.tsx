import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Dataset, type DatasetStatus } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Modal, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { DatasetWizard } from "../components/datasets/DatasetWizard";

type SortKey = "az" | "za" | "createdDesc" | "createdAsc" | "updatedDesc" | "updatedAsc" | "rowsDesc" | "rowsAsc" | "colsDesc" | "colsAsc";

const STATUS_TONE: Record<DatasetStatus, "green" | "blue" | "red"> = {
  prepared: "green",
  partial: "blue",
  unprepared: "red",
};

function statusOf(d: Dataset): DatasetStatus {
  return d.status ?? (d.prepared ? "prepared" : d.hasTrain ? "partial" : "unprepared");
}

function splitTag(ok?: boolean, label?: string) {
  return <Tag tone={ok ? "green" : "neutral"} mono>{label}</Tag>;
}

function sourceText(d: Dataset) {
  return d.source && d.source !== "-" ? d.source : d.sources?.[0]?.name || d.sources?.[0]?.url || "-";
}

function textBlob(d: Dataset) {
  return [
    d.name,
    d.id,
    d.target,
    d.metric,
    d.source,
    d.desc,
    d.suite,
    ...(d.tags ?? []),
    ...(d.sources ?? []).flatMap((s) => [s.name, s.url, s.license, s.organization]),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function DatasetCard({ d, onOpen, onDelete }: { d: Dataset; onOpen: () => void; onDelete: () => void }) {
  const status = statusOf(d);
  return (
    <Card className="ds-card dataset-card-rich">
      <div className="spread">
        <div style={{ minWidth: 0 }}>
          <div className="ds-title">{d.name}</div>
          <div className="mono faint dataset-id">{d.id}</div>
        </div>
        <Tag tone={STATUS_TONE[status]}>{status}</Tag>
      </div>
      {d.desc && <div className="muted clamp-2">{d.desc}</div>}
      <div className="run-meta-line" style={{ margin: 0 }}>
        <Tag tone={d.taskType === "regression" ? "blue" : d.taskType === "classification" ? "accent" : "neutral"}>{d.task}</Tag>
        <Tag mono>{d.metricGoal ?? "max"}</Tag>
        {(d.tags ?? []).slice(0, 3).map((tag) => <Tag key={tag} tone="neutral">{tag}</Tag>)}
        {d.warningsCount ? <Tag tone="red">{d.warningsCount} warning{d.warningsCount === 1 ? "" : "s"}</Tag> : null}
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
        {d.createdAt && <span>created {new Date(d.createdAt).toLocaleString()}</span>}
        {d.updatedAt && <span>updated {new Date(d.updatedAt).toLocaleString()}</span>}
      </div>
      <div className="ds-actions">
        <Button size="sm" icon="external" onClick={onOpen}>Open</Button>
        <Button size="sm" variant="ghost" icon="trash" onClick={onDelete}>Delete</Button>
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
  const [splitFilter, setSplitFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("");
  const [warningsFilter, setWarningsFilter] = useState("all");
  const [sort, setSort] = useState<SortKey>("updatedDesc");
  const [notice, setNotice] = useState<string | null>(null);

  const metrics = useMemo(() => Array.from(new Set((data ?? []).map((d) => d.metric).filter(Boolean))).sort(), [data]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = (data ?? []).filter((d) => {
      const status = statusOf(d);
      if (q && !textBlob(d).includes(q)) return false;
      if (taskFilter !== "all" && (d.taskType ?? "unknown") !== taskFilter) return false;
      if (statusFilter !== "all" && status !== statusFilter) return false;
      if (metricFilter !== "all" && d.metric !== metricFilter) return false;
      if (splitFilter === "train" && !d.hasTrain) return false;
      if (splitFilter === "val" && !d.hasVal) return false;
      if (splitFilter === "test" && !d.hasTest) return false;
      if (sourceFilter.trim() && !sourceText(d).toLowerCase().includes(sourceFilter.trim().toLowerCase())) return false;
      if (warningsFilter === "with" && !(d.warningsCount && d.warningsCount > 0)) return false;
      if (warningsFilter === "without" && (d.warningsCount ?? 0) > 0) return false;
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
  }, [data, metricFilter, query, sort, sourceFilter, splitFilter, statusFilter, taskFilter, warningsFilter]);

  async function del() {
    if (!toDelete) return;
    setBusyDel(true);
    try {
      await api.deleteDataset(toDelete.id);
      setNotice(`Deleted ${toDelete.name}.`);
      setToDelete(null);
      reload();
    } finally {
      setBusyDel(false);
    }
  }

  function created(ds: Dataset) {
    setWizardOpen(false);
    setNotice(`Created ${ds.name}.`);
    reload();
    nav(`/datasets/${ds.id}`);
  }

  if (loading && !data) {
    return <div className="grid-3">{[0, 1, 2].map((i) => <Card key={i}><Skeleton h={180} /></Card>)}</div>;
  }

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="spread dataset-center-head">
        <div>
          <h2 className="page-title">Dataset Center</h2>
          <div className="muted">Manage raw files, prepared train/val/test splits, metadata, sources, and agent-facing notes for AutoML Gym.</div>
        </div>
        <Button variant="primary" icon="plus" onClick={() => setWizardOpen(true)}>Add dataset</Button>
      </div>

      {notice && <div className="success-line"><Icon name="check" size={15} /> {notice}</div>}

      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={16} />
            <input className="input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search name, id, target, metric, source, tags..." />
          </div>
          <select className="select-sm" value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="updatedDesc">Updated newest</option>
            <option value="updatedAsc">Updated oldest</option>
            <option value="createdDesc">Created newest</option>
            <option value="createdAsc">Created oldest</option>
            <option value="az">A-Z</option>
            <option value="za">Z-A</option>
            <option value="rowsDesc">Rows high-low</option>
            <option value="rowsAsc">Rows low-high</option>
            <option value="colsDesc">Features high-low</option>
            <option value="colsAsc">Features low-high</option>
          </select>
          <Button variant="ghost" icon="sliders" onClick={() => setFiltersOpen((v) => !v)}>Filters</Button>
        </div>
        {filtersOpen && (
          <div className="dataset-filters-grid">
            <Field label="Task">
              <select className="input" value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)}>
                <option value="all">all</option>
                <option value="classification">classification</option>
                <option value="regression">regression</option>
                <option value="unknown">unknown</option>
              </select>
            </Field>
            <Field label="Status">
              <select className="input" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                <option value="all">all</option>
                <option value="prepared">prepared</option>
                <option value="partial">partial</option>
                <option value="unprepared">unprepared</option>
              </select>
            </Field>
            <Field label="Metric">
              <select className="input" value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)}>
                <option value="all">all</option>
                {metrics.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            <Field label="Required split">
              <select className="input" value={splitFilter} onChange={(e) => setSplitFilter(e.target.value)}>
                <option value="all">all</option>
                <option value="train">has train</option>
                <option value="val">has val</option>
                <option value="test">has test</option>
              </select>
            </Field>
            <Field label="Source contains"><input className="input" value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)} /></Field>
            <Field label="Warnings">
              <select className="input" value={warningsFilter} onChange={(e) => setWarningsFilter(e.target.value)}>
                <option value="all">all</option>
                <option value="with">with warnings</option>
                <option value="without">no warnings</option>
              </select>
            </Field>
          </div>
        )}
      </Card>

      {!filtered.length ? (
        <EmptyState
          icon="database"
          title="No datasets match"
          text="Adjust the search or filters, or create a new dataset from raw or prepared files."
          action={<Button variant="primary" icon="plus" onClick={() => setWizardOpen(true)}>Add dataset</Button>}
        />
      ) : (
        <div className="grid-3">
          {filtered.map((d) => (
            <DatasetCard key={d.id} d={d} onOpen={() => nav(`/datasets/${d.id}`)} onDelete={() => setToDelete(d)} />
          ))}
        </div>
      )}

      {wizardOpen && <DatasetWizard onClose={() => setWizardOpen(false)} onCreated={created} />}
      {toDelete && (
        <Modal
          title="Delete dataset"
          width={420}
          onClose={() => setToDelete(null)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setToDelete(null)}>Cancel</Button>
              <Button variant="danger" onClick={del} disabled={busyDel}>{busyDel ? <Spinner /> : "Delete"}</Button>
            </>
          }
        >
          <p style={{ margin: 0 }}>
            Delete <strong>{toDelete.name}</strong> and all raw/prepared files? This cannot be undone.
          </p>
        </Modal>
      )}
    </div>
  );
}
