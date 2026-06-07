import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { api, type ModelRec } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Dot, EmptyState, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const PROVIDERS = ["OpenAI-совместимый", "vLLM", "Gemini", "LiteLLM"];

function ConfirmModal({ count, onConfirm, onCancel, busy }: { count: number; onConfirm: () => void; onCancel: () => void; busy: boolean }) {
  return createPortal(
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">Вернуть модели?</h3>
        <p className="modal-desc">
          {count === 1 ? "1 модель будет возвращена из архива." : `${count} моделей будут возвращены из архива.`}
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

export default function ModelsArchive() {
  const nav = useNavigate();
  const { data: models, loading, reload } = useAsync(() => api.listArchivedModels(), []);
  const [q, setQ] = useState("");
  const [providerFilter, setProviderFilter] = useState("all");

  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState(false);
  const [restoring, setRestoring] = useState(false);

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (models ?? []).filter((m) => {
      if (providerFilter !== "all" && m.provider !== providerFilter) return false;
      if (term && !`${m.name} ${m.provider} ${m.baseUrl ?? ""}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [models, q, providerFilter]);

  function toggleSelect(id: string) {
    setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  const allFilteredSelected = filtered.length > 0 && filtered.every((m) => selected.has(m.id));

  function toggleSelectAll() {
    setSelected((s) => {
      const n = new Set(s);
      if (allFilteredSelected) filtered.forEach((m) => n.delete(m.id));
      else filtered.forEach((m) => n.add(m.id));
      return n;
    });
  }

  function cancelSelect() { setSelecting(false); setSelected(new Set()); }

  async function doRestore() {
    setRestoring(true);
    try {
      await api.unarchiveModels([...selected]);
      setConfirm(false);
      cancelSelect();
      reload();
    } finally { setRestoring(false); }
  }

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button className="archive-link" onClick={() => nav("/models")}>
          <Icon name="chevronLeft" size={15} /> Назад к моделям
        </button>
      </div>

      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={17} />
            <input className="input" placeholder="Поиск по имени, провайдеру…" value={q} onChange={(e) => setQ(e.target.value)} />
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

      {loading && !models ? (
        <Skeleton h={200} />
      ) : filtered.length ? (
        <div className="model-grid">
          {filtered.map((m) => (
            <Card key={m.id} className={`model-card${selecting && selected.has(m.id) ? " row-selected" : ""}`}
              onClick={() => selecting && toggleSelect(m.id)} style={{ cursor: selecting ? "pointer" : undefined }}>
              <div className="spread" style={{ marginBottom: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flex: 1 }}>
                  {selecting && <input type="checkbox" className="row-checkbox" checked={selected.has(m.id)} onChange={() => toggleSelect(m.id)} onClick={(e) => e.stopPropagation()} />}
                  <div className="ds-title" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</div>
                </div>
              </div>
              <div className="run-meta-line" style={{ margin: "0 0 10px" }}>
                <Tag>{m.provider}</Tag>
                <span className="row" style={{ gap: 5 }}>
                  <Dot tone={m.online === false ? "red" : m.online ? "green" : "gray"} />
                  <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{m.online === false ? "офлайн" : m.online ? "онлайн" : "не проверено"}</span>
                </span>
              </div>
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">Input limit</span><span className="v">{(m.ctx / 1024).toFixed(0)}k</span></div>
                <div className="ds-stat"><span className="k">Output limit</span><span className="v">{m.maxTokens ? `${(m.maxTokens / 1024).toFixed(0)}k` : "—"}</span></div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState icon="archive" title="Архив пуст" text="Архивированные модели появятся здесь." />
        </Card>
      )}

      {selecting && selected.size > 0 && createPortal(
        <div className="selection-bar">
          <span className="selection-bar-label">Выбрано {selected.size} модел{selected.size === 1 ? "ь" : selected.size < 5 ? "и" : "ей"}</span>
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
