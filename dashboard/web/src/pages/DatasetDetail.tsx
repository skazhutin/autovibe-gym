import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, EmptyState, Field, Skeleton, Spinner, Tabs, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { MiniHist } from "../components/charts";

const TABS = [
  { id: "preview", label: "Превью данных", icon: "table" },
  { id: "columns", label: "Статистика колонок", icon: "sliders" },
  { id: "meta", label: "Метаданные", icon: "settings" },
];

function PreviewTab({ id }: { id: string }) {
  const { data, loading } = useAsync(() => api.datasetPreview(id, "train", 50), [id]);
  if (loading && !data) return <Skeleton h={240} />;
  if (!data || !data.columns.length) return <EmptyState icon="table" title="Нет данных" text="train.csv недоступен." />;
  return (
    <Card style={{ padding: 0 }}>
      <div className="table-wrap">
        <table className="data">
          <thead><tr>{data.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr key={i}>{row.map((v, j) => <td key={j} className="mono" style={{ fontSize: 12.5 }}>{v === null ? <span className="faint">∅</span> : String(v)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="faint" style={{ padding: "10px 14px", fontSize: 12 }}>показано {data.shown} из {data.total.toLocaleString()} строк</div>
    </Card>
  );
}

function ColumnsTab({ id }: { id: string }) {
  const { data, loading } = useAsync(() => api.datasetColumns(id, "train"), [id]);
  if (loading && !data) return <Skeleton h={240} />;
  if (!data || !data.length) return <EmptyState icon="sliders" title="Нет статистики" />;
  return (
    <Card style={{ padding: 0 }}>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Колонка</th><th>Тип</th><th>Вид</th><th>Пропуски</th><th>Уникальных</th><th>Распределение</th></tr></thead>
          <tbody>
            {data.map((c) => (
              <tr key={c.name}>
                <td className="mono">{c.name}</td>
                <td className="mono faint">{c.dtype}</td>
                <td><Tag tone={c.kind === "numeric" ? "blue" : "neutral"}>{c.kind === "numeric" ? "число" : "категория"}</Tag></td>
                <td className="mono" style={{ color: c.missingPct > 0 ? "var(--orange)" : "var(--text-dim)" }}>{c.missingPct}%</td>
                <td className="mono faint">{c.unique}</td>
                <td><MiniHist data={c.hist} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function MetaTab({ id, onSaved }: { id: string; onSaved: () => void }) {
  const { data, loading } = useAsync(() => api.getDataset(id), [id]);
  const [form, setForm] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState(false);
  if (loading && !data) return <Skeleton h={200} />;
  if (!data) return null;
  const val = (k: keyof typeof data, f: string) => form[f] ?? String((data as unknown as Record<string, unknown>)[k] ?? "");

  async function save() {
    setBusy(true); setOk(false);
    try {
      await api.updateDataset(id, {
        name: val("name", "name"), target: val("target", "target"),
        metric: val("metric", "metric"), desc: val("desc", "desc"),
      });
      setOk(true); onSaved();
    } finally { setBusy(false); }
  }

  return (
    <Card>
      <div className="stack" style={{ gap: 14, maxWidth: 520 }}>
        <Field label="Имя"><input className="input" value={val("name", "name")} onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))} /></Field>
        <Field label="Target-колонка"><input className="input mono" value={val("target", "target")} onChange={(e) => setForm((s) => ({ ...s, target: e.target.value }))} /></Field>
        <Field label="Метрика"><input className="input mono" value={val("metric", "metric")} onChange={(e) => setForm((s) => ({ ...s, metric: e.target.value }))} /></Field>
        <Field label="Описание"><textarea className="input" rows={3} value={val("desc", "desc")} onChange={(e) => setForm((s) => ({ ...s, desc: e.target.value }))} /></Field>
        <div className="row">
          <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить"}</Button>
          {ok && <span style={{ color: "var(--green)", fontSize: 13 }}><Icon name="check" size={14} /> сохранено</span>}
        </div>
      </div>
    </Card>
  );
}

export default function DatasetDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [tab, setTab] = useState("preview");
  const { data, loading, reload } = useAsync(() => api.getDataset(id), [id]);

  if (loading && !data) return <Skeleton h={300} />;
  if (!data) return <EmptyState icon="alert" title="Датасет не найден" action={<Button onClick={() => nav("/datasets")}>К датасетам</Button>} />;

  return (
    <div>
      <button className="back-link" onClick={() => nav("/datasets")}><Icon name="chevronLeft" size={16} /> Все датасеты</button>
      <Card style={{ marginBottom: 20 }}>
        <div className="spread">
          <div>
            <div className="ds-title" style={{ fontSize: 18 }}>{data.name}</div>
            <div className="run-meta-line" style={{ margin: "10px 0 0" }}>
              <Tag tone={data.task === "Регрессия" ? "blue" : "neutral"}>{data.task}</Tag>
              <span className="mono faint">метрика: {data.metric}</span>
              <span className="mono faint">target: {data.target}</span>
              {!data.prepared && <Tag tone="red">не подготовлен</Tag>}
            </div>
          </div>
          {data.prepared && (
            <div className="chip-metrics">
              <div className="chip-metric"><div><div className="cm-label">строк</div><div className="cm-val">{data.rows.toLocaleString()}</div></div></div>
              <span className="vline" />
              <div className="chip-metric"><div><div className="cm-label">признаков</div><div className="cm-val">{data.cols}</div></div></div>
            </div>
          )}
        </div>
      </Card>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "preview" && <PreviewTab id={id} />}
      {tab === "columns" && <ColumnsTab id={id} />}
      {tab === "meta" && <MetaTab id={id} onSaved={reload} />}
    </div>
  );
}
