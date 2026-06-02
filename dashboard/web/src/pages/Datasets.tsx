import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Dataset } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Field, Modal, Skeleton, Spinner, Tag } from "../components/ui";

function CreateDatasetModal({ onClose, onDone }: { onClose: () => void; onDone: (id: string) => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!name.trim()) {
      setErr("Укажите имя датасета.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const created = await api.createDataset(name.trim());
      onDone(created.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Не удалось создать датасет.");
      setBusy(false);
    }
  }

  return (
    <Modal
      title="Новый датасет"
      width={460}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Отмена</Button>
          <Button variant="primary" onClick={submit} disabled={busy}>{busy ? <Spinner /> : "Создать"}</Button>
        </>
      }
    >
      <div className="stack" style={{ gap: 14 }}>
        <Field label="Имя датасета" hint="Будет использовано как имя папки в datasets/">
          <input className="input mono" value={name} onChange={(e) => setName(e.target.value)} placeholder="uk_road_safety" />
        </Field>
        <div className="muted" style={{ fontSize: 13 }}>
          После создания вы попадёте в редактор конфигурации, где можно загрузить файлы, скачать их по URL, настроить joins и подготовить splits.
        </div>
        {err && <div style={{ color: "var(--red)", fontSize: 13 }}>{err}</div>}
      </div>
    </Modal>
  );
}

export default function Datasets() {
  const nav = useNavigate();
  const { data, loading, reload } = useAsync(() => api.listDatasets(), []);
  const [creating, setCreating] = useState(false);
  const [toDelete, setToDelete] = useState<Dataset | null>(null);
  const [busyDel, setBusyDel] = useState(false);

  async function del() {
    if (!toDelete) return;
    setBusyDel(true);
    try {
      await api.deleteDataset(toDelete.id);
      setToDelete(null);
      reload();
    } finally {
      setBusyDel(false);
    }
  }

  if (loading && !data) {
    return <div className="grid-3">{[0, 1, 2].map((i) => <Card key={i}><Skeleton h={140} /></Card>)}</div>;
  }

  return (
    <div>
      <div className="spread" style={{ marginBottom: 18, gap: 12, flexWrap: "wrap" }}>
        <div className="muted" style={{ maxWidth: 720 }}>
          Единый менеджер датасетов: raw/pre-split ingestion, загрузка файлов, URL downloads, multi-table joins, валидация и подготовка в совместимый `prepared/` формат.
        </div>
        <Button variant="primary" icon="plus" onClick={() => setCreating(true)}>Новый датасет</Button>
      </div>

      <div className="grid-3">
        {(data ?? []).map((d) => (
          <Card key={d.id} className="ds-card">
            <div className="spread">
              <span className="ds-title">{d.name}</span>
              <Tag tone={d.task === "Регрессия" ? "blue" : "neutral"}>{d.task}</Tag>
            </div>
            {d.desc && <div className="muted" style={{ fontSize: 13 }}>{d.desc}</div>}
            <div className="pick-chips" style={{ marginTop: 10, marginBottom: 12 }}>
              <Tag mono>{d.ingestionMode || "—"}</Tag>
              <Tag mono>{d.metric}</Tag>
              <Tag mono>{d.target}</Tag>
            </div>
            {d.prepared ? (
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">строк</span> <span className="v">{d.rows.toLocaleString()}</span></div>
                <div className="ds-stat"><span className="k">признаков</span> <span className="v">{d.cols}</span></div>
                <div className="ds-stat"><span className="k">suite</span> <span className="v">{d.suite ?? "—"}</span></div>
                <div className="ds-stat"><span className="k">seed</span> <span className="v">{d.seed ?? "—"}</span></div>
              </div>
            ) : (
              <Tag tone="red">ещё не подготовлен</Tag>
            )}
            <div className="ds-actions">
              <Button size="sm" icon="external" onClick={() => nav(`/datasets/${d.id}`)}>Открыть</Button>
              <Button size="sm" variant="ghost" icon="trash" onClick={() => setToDelete(d)}>Удалить</Button>
            </div>
          </Card>
        ))}
      </div>

      {creating && (
        <CreateDatasetModal
          onClose={() => setCreating(false)}
          onDone={(id) => {
            setCreating(false);
            reload();
            nav(`/datasets/${id}`);
          }}
        />
      )}

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
            Удалить датасет <strong>{toDelete.name}</strong> и все его файлы? Действие необратимо.
          </p>
        </Modal>
      )}
    </div>
  );
}
