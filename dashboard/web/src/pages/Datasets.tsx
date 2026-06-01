import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Dataset } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Field, Modal, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

function DropZone({ label, file, onPick }: { label: string; file: File | null; onPick: (f: File | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div className={`dropzone${file ? " has" : ""}`} onClick={() => ref.current?.click()}>
      <input ref={ref} type="file" accept=".csv" hidden onChange={(e) => onPick(e.target.files?.[0] ?? null)} />
      <Icon name={file ? "check" : "upload"} size={18} />
      <div style={{ marginTop: 4 }}>{file ? file.name : label}</div>
    </div>
  );
}

function UploadModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [name, setName] = useState("");
  const [target, setTarget] = useState("");
  const [metric, setMetric] = useState("");
  const [seed, setSeed] = useState("42");
  const [desc, setDesc] = useState("");
  const [files, setFiles] = useState<{ train: File | null; val: File | null; test: File | null }>({ train: null, val: null, test: null });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!name || !files.train) { setErr("Укажите имя и хотя бы train.csv"); return; }
    setBusy(true); setErr(null);
    const fd = new FormData();
    fd.append("name", name);
    if (target) fd.append("target", target);
    if (metric) fd.append("metric", metric);
    if (seed) fd.append("seed", seed);
    if (desc) fd.append("desc", desc);
    (["train", "val", "test"] as const).forEach((k) => files[k] && fd.append(k, files[k]!));
    try { await api.uploadDataset(fd); onDone(); } catch (e) { setErr(e instanceof Error ? e.message : "Ошибка загрузки"); setBusy(false); }
  }

  return (
    <Modal title="Загрузить датасет" width={560} onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Отмена</Button>
        <Button variant="primary" onClick={submit} disabled={busy}>{busy ? <Spinner /> : "Создать"}</Button>
      </>}>
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <DropZone label="train.csv" file={files.train} onPick={(f) => setFiles((s) => ({ ...s, train: f }))} />
        <DropZone label="val.csv" file={files.val} onPick={(f) => setFiles((s) => ({ ...s, val: f }))} />
        <DropZone label="test.csv" file={files.test} onPick={(f) => setFiles((s) => ({ ...s, test: f }))} />
      </div>
      <div className="stack" style={{ gap: 14 }}>
        <Field label="Имя датасета"><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="my_dataset" /></Field>
        <div className="grid-2">
          <Field label="Target-колонка"><input className="input mono" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" /></Field>
          <Field label="Метрика" hint="f1_weighted, f1_macro, neg_rmse"><input className="input mono" value={metric} onChange={(e) => setMetric(e.target.value)} placeholder="f1_weighted" /></Field>
        </div>
        <div className="grid-2">
          <Field label="Seed"><input className="input mono" value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>
          <Field label="Описание"><input className="input" value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
        </div>
        {err && <div style={{ color: "var(--red)", fontSize: 13 }}>{err}</div>}
      </div>
    </Modal>
  );
}

export default function Datasets() {
  const nav = useNavigate();
  const { data, loading, reload } = useAsync(() => api.listDatasets(), []);
  const [uploading, setUploading] = useState(false);
  const [toDelete, setToDelete] = useState<Dataset | null>(null);
  const [busyDel, setBusyDel] = useState(false);

  async function del() {
    if (!toDelete) return;
    setBusyDel(true);
    try { await api.deleteDataset(toDelete.id); setToDelete(null); reload(); } finally { setBusyDel(false); }
  }

  if (loading && !data) return <div className="grid-3">{[0, 1, 2].map((i) => <Card key={i}><Skeleton h={140} /></Card>)}</div>;

  return (
    <div>
      <div className="grid-3">
        {(data ?? []).map((d) => (
          <Card key={d.id} className="ds-card">
            <div className="spread">
              <span className="ds-title">{d.name}</span>
              <Tag tone={d.task === "Регрессия" ? "blue" : "neutral"}>{d.task}</Tag>
            </div>
            {d.desc && <div className="muted" style={{ fontSize: 13 }}>{d.desc}</div>}
            {d.prepared ? (
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">строк</span> <span className="v">{d.rows.toLocaleString()}</span></div>
                <div className="ds-stat"><span className="k">признаков</span> <span className="v">{d.cols}</span></div>
                <div className="ds-stat"><span className="k">метрика</span> <span className="v">{d.metric}</span></div>
                <div className="ds-stat"><span className="k">target</span> <span className="v">{d.target}</span></div>
              </div>
            ) : (
              <Tag tone="red">не подготовлен (нет train/val/test)</Tag>
            )}
            <div className="ds-actions">
              <Button size="sm" icon="external" onClick={() => nav(`/datasets/${d.id}`)}>Открыть</Button>
              <Button size="sm" variant="ghost" icon="trash" onClick={() => setToDelete(d)}>Удалить</Button>
            </div>
          </Card>
        ))}
        <div className="upload-tile" onClick={() => setUploading(true)}>
          <div>
            <Icon name="upload" size={26} />
            <div style={{ marginTop: 10, fontWeight: 600 }}>Загрузить датасет</div>
            <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>train / val / test CSV + метаданные</div>
          </div>
        </div>
      </div>

      {uploading && <UploadModal onClose={() => setUploading(false)} onDone={() => { setUploading(false); reload(); }} />}
      {toDelete && (
        <Modal title="Удалить датасет" width={420} onClose={() => setToDelete(null)}
          footer={<>
            <Button variant="ghost" onClick={() => setToDelete(null)}>Отмена</Button>
            <Button variant="danger" onClick={del} disabled={busyDel}>{busyDel ? <Spinner /> : "Удалить"}</Button>
          </>}>
          <p style={{ margin: 0 }}>Удалить датасет <strong>{toDelete.name}</strong> и все его файлы? Действие необратимо.</p>
        </Modal>
      )}
    </div>
  );
}
