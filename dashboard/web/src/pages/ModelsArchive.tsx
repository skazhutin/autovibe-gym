import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { api, type ModelRec } from "../lib/api";
import { ModelModal } from "./Models";
import { useAsync } from "../lib/hooks";
import { Button, Card, Dot, EmptyState, Skeleton, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { SelectionBar } from "../components/SelectionBar";

export default function ModelsArchive() {
  const nav = useNavigate();
  const { data: models, loading, reload } = useAsync(() => api.listArchivedModels(), []);
  const [q, setQ] = useState("");
  const [providerFilter] = useState("all");

  const [edit, setEdit] = useState<ModelRec | null>(null);
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

  const toolbarEl = document.getElementById("toolbar-portal");

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button className="archive-link" onClick={() => nav("/models")}>
          <Icon name="chevronLeft" size={15} /> Назад к моделям
        </button>
      </div>

      {toolbarEl && createPortal(<Card className="dataset-toolbar">
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
      </Card>, toolbarEl)}

      {loading && !models ? (
        <Skeleton h={200} />
      ) : filtered.length ? (
        <div className="model-grid">
          {filtered.map((m) => (
            <Card key={m.id} className={`model-card${selecting && selected.has(m.id) ? " row-selected" : ""}`}
              onClick={() => selecting ? toggleSelect(m.id) : setEdit(m)} style={{ cursor: "pointer" }} hover={!selecting}>
              <div className="spread" style={{ marginBottom: 5 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flex: 1 }}>
                  <div className="ds-title" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</div>
                </div>
                <Tag><Dot tone="gray" />не проверено</Tag>
              </div>
              <div className="run-meta-line" style={{ margin: "0 0 8px" }}>
                <Tag>{m.provider}</Tag>
                {m.hasApiKey && <Tag>API key</Tag>}
              </div>
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">Input limit</span><span className="v">{(m.ctx / 1024).toFixed(0)}k</span></div>
                <div className="ds-stat"><span className="k">Output limit</span><span className="v">{m.maxTokens ? `${(m.maxTokens / 1024).toFixed(0)}k` : "—"}</span></div>
                <div className={`ds-stat${(m.baseUrl || "").length > 35 ? " span-full" : ""}`}><span className="k">Base URL</span><span className="v" style={{ fontSize: 11.5 }}>{m.baseUrl || "—"}</span></div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState icon="archive" title="Архив пуст" text="Архивированные модели появятся здесь." />
        </Card>
      )}

      {selecting && selected.size > 0 && (
        <SelectionBar
          count={selected.size}
          noun="модель"
          actionLabel="Вернуть"
          actionIcon="undo"
          busy={restoring}
          onAction={() => setConfirm(true)}
          onCancel={cancelSelect}
        />
      )}

      {confirm && (
        <ConfirmDialog
          title="Вернуть модели?"
          description={selected.size === 1 ? "1 модель будет возвращена из архива." : `${selected.size} моделей будут возвращены из архива.`}
          confirmLabel="Вернуть"
          busy={restoring}
          onConfirm={doRestore}
          onCancel={() => setConfirm(false)}
        />
      )}
      {edit && <ModelModal initial={edit} onClose={() => setEdit(null)} onDone={() => { setEdit(null); reload(); }}
        onUnarchive={async () => { await api.unarchiveModels([edit.id]); setEdit(null); reload(); }} />}
    </div>
  );
}
