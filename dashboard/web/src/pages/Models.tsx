import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate, useOutletContext } from "react-router-dom";
import { api, type ModelRec } from "../lib/api";
import type { SetHeaderAction } from "../components/Layout";
import { useAsync } from "../lib/hooks";
import { Button, Card, Dot, EmptyState, FieldInfo, Field, Modal, SelectDropdown, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";
import { formatDateOnly } from "../lib/date";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { SelectionBar } from "../components/SelectionBar";

const PROVIDERS = ["OpenAI-совместимый", "vLLM", "Gemini", "LiteLLM"];
const needsBaseUrl = (provider: string) => provider === "OpenAI-совместимый" || provider === "vLLM";


export type ModelDraft = { name: string; provider: string; baseUrl: string; apiKey: string; ctx: number | string; maxTokens: number | string };
export const defaultDraft = (): ModelDraft => ({ name: "", provider: PROVIDERS[0], baseUrl: "", apiKey: "", ctx: 32768, maxTokens: 8192 });

export function ModelModal({ initial, draft, onDraftChange, onClose, onDone, onUnarchive }: {
  initial?: ModelRec; draft?: ModelDraft; onDraftChange?: (d: ModelDraft) => void; onClose: () => void; onDone: () => void; onUnarchive?: () => void;
}) {
  const [f, setF] = useState<ModelDraft>(draft ?? (initial ? {
    name: initial.name,
    provider: initial.provider,
    baseUrl: initial.baseUrl ?? "",
    apiKey: "",
    ctx: initial.ctx,
    maxTokens: initial.maxTokens ?? 8192,
  } : defaultDraft()));
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmArchive, setConfirmArchive] = useState(false);
  const set = (k: string, v: string | number) => setF((s) => { const n = { ...s, [k]: v }; onDraftChange?.(n); return n; });
  const showBaseUrl = needsBaseUrl(f.provider);

  async function save() {
    if (!f.name) return;
    setBusy(true);
    const payload = {
      ...f,
      baseUrl: showBaseUrl ? f.baseUrl : "",
      ctx: Number(f.ctx),
      maxTokens: Number(f.maxTokens),
      apiKey: f.apiKey || undefined,
    };
    try {
      if (initial) await api.updateModel(initial.id, payload);
      else await api.createModel(payload);
      onDone();
    } finally { setBusy(false); }
  }

  async function check() {
    if (!initial) { setTest("Сохраните модель, затем проверьте связь."); return; }
    setTest("…");
    const r = await api.checkModel(initial.id);
    setTest(r.online ? "Соединение есть" : `Недоступно${r.error ? `: ${r.error}` : r.status ? ` (${r.status})` : ""}`);
  }

  return (
    <>
    <Modal title={initial ? "Редактировать модель" : "Добавить модель"} width={720} onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={check}>Проверить связь</Button>
        {initial && onUnarchive && <Button variant="secondary" onClick={onUnarchive} disabled={busy}>Вернуть из архива</Button>}
        {initial && !onUnarchive && <Button variant="secondary" onClick={() => setConfirmArchive(true)} disabled={busy}>Архивировать</Button>}
        {initial && <Button variant="danger" onClick={() => setConfirmDelete(true)} disabled={busy}>Удалить</Button>}
        <div style={{ flex: 1 }} />
        <Button variant="ghost" onClick={onClose}>Отмена</Button>
        <Button variant="primary" onClick={save} disabled={busy || !f.name}>{busy ? <Spinner /> : "Сохранить"}</Button>
      </>}>
      <div className="stack" style={{ gap: 14 }}>
        <FieldInfo label="Имя модели" info="Название модели как у провайдера, напр. anthropic/claude-opus-4-5, gemini-2.5-flash"><input className="input mono" value={f.name} onChange={(e) => set("name", e.target.value)} /></FieldInfo>
        <div className="grid-2">
          <FieldInfo label="Input token limit" info="Максимум токенов в запросе (контекстное окно модели). Если превышено — прогон завершится с ошибкой. Напр. 32768 для большинства моделей, 128000 для GPT-4o."><input className="input mono" value={f.ctx} onChange={(e) => set("ctx", e.target.value.replace(/\D/g, ""))} /></FieldInfo>
        </div>
        <FieldInfo label="Провайдер" info="Тип API: OpenAI-совместимый и vLLM требуют Base URL, Gemini — Google API Key, LiteLLM — любой провайдер через litellm"><SelectDropdown value={f.provider} options={PROVIDERS.map((p) => ({ value: p, label: p }))} onChange={(v) => set("provider", v)} /></FieldInfo>
        {showBaseUrl && <FieldInfo label="URL" info="Базовый адрес API. Для vLLM/локального сервера: http://host:8000/v1. Для OpenRouter: https://openrouter.ai/api/v1"><input className="input mono" value={f.baseUrl} onChange={(e) => set("baseUrl", e.target.value)} placeholder="http://host:8000/v1" /></FieldInfo>}
        <FieldInfo label="API-ключ" info={initial?.hasApiKey ? "Ключ уже сохранён — оставьте пустым, чтобы не менять" : "Ключ авторизации у провайдера. Для локального vLLM можно оставить пустым."}><input className="input" type="password" value={f.apiKey} onChange={(e) => set("apiKey", e.target.value)} placeholder="••••••••" /></FieldInfo>
        <FieldInfo label="Output token limit" info="Максимум токенов в одном ответе. Если модель упирается в этот лимит — ответ обрезается и прогон завершается с ошибкой."><input className="input mono" value={f.maxTokens} onChange={(e) => set("maxTokens", e.target.value.replace(/\D/g, ""))} /></FieldInfo>
        {test && <div style={{ fontSize: 13, color: test === "Соединение есть" ? "var(--green)" : test === "…" ? "var(--text-dim)" : "var(--red)" }}>{test === "…" ? <Spinner /> : test}</div>}
      </div>
    </Modal>
    {confirmArchive && initial && (
      <ConfirmDialog
        title="Архивировать модель?"
        description="Модель будет перемещена в архив. Вернуть её можно из раздела «Архив»."
        confirmLabel="Архивировать"
        busy={busy}
        onConfirm={async () => { setBusy(true); try { await api.archiveModels([initial.id]); onDone(); } finally { setBusy(false); } }}
        onCancel={() => setConfirmArchive(false)}
      />
    )}
    {confirmDelete && initial && (
      <ConfirmDialog
        title="Удалить модель?"
        description="Модель будет удалена безвозвратно. Это действие нельзя отменить."
        confirmLabel="Удалить"
        confirmVariant="danger"
        busy={busy}
        onConfirm={async () => { setBusy(true); try { await api.deleteModel(initial.id); onDone(); } finally { setBusy(false); } }}
        onCancel={() => setConfirmDelete(false)}
      />
    )}
    </>
  );
}


export default function Models() {
  const nav = useNavigate();
  const setHeaderAction = useOutletContext<SetHeaderAction>();
  const { data, loading, reload } = useAsync(() => api.listModels(), []);
  const { data: settings } = useAsync(() => api.getSettings(), []);
  const [edit, setEdit] = useState<ModelRec | null>(null);
  const [adding, setAdding] = useState(false);
  const [newDraft, setNewDraft] = useState<ModelDraft>(defaultDraft());
  const [checking, setChecking] = useState<string | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [archiving, setArchiving] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [q, setQ] = useState("");
  const [providerFilter, setProviderFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const [sortField, setSortField] = useState<"date" | "az">("date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    setHeaderAction({ label: "Новая модель", icon: "plus", onClick: () => setAdding(true) });
    return () => setHeaderAction(null);
  }, [setHeaderAction]);

  async function check(id: string) {
    setChecking(id);
    try { await api.checkModel(id); reload(); } finally { setChecking(null); }
  }

  function toggleSelect(id: string) {
    setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  function cancelSelect() { setSelecting(false); setSelected(new Set()); }

  async function doArchive() {
    setArchiving(true);
    try { await api.archiveModels([...selected]); setConfirm(false); cancelSelect(); reload(); }
    finally { setArchiving(false); }
  }

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    const list = (data ?? []).filter((m) => {
      if (term && !`${m.name} ${m.baseUrl ?? ""} ${m.provider}`.toLowerCase().includes(term)) return false;
      if (providerFilter !== "all" && m.provider !== providerFilter) return false;
      if (statusFilter === "online" && !m.online) return false;
      if (statusFilter === "offline" && m.online !== false) return false;
      if (statusFilter === "unchecked" && m.online !== null && m.online !== undefined) return false;
      return true;
    });
    const t = (v?: string | null) => (v ? new Date(v).getTime() : 0);
    const asc = sortDir === "asc";
    return [...list].sort((a, b) => {
      if (sortField === "az") return asc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
      return asc ? t(a.createdAt) - t(b.createdAt) : t(b.createdAt) - t(a.createdAt);
    });
  }, [data, q, providerFilter, statusFilter, sortField, sortDir]);

  const allFilteredSelected = filtered.length > 0 && filtered.every((m) => selected.has(m.id));

  function toggleSelectAll() {
    setSelected((s) => {
      const n = new Set(s);
      if (allFilteredSelected) filtered.forEach((m) => n.delete(m.id));
      else filtered.forEach((m) => n.add(m.id));
      return n;
    });
  }

  const toolbarEl = document.getElementById("toolbar-portal");

  return (
    <div className="stack" style={{ gap: 18 }}>
      {toolbarEl && createPortal(<Card className="dataset-toolbar">
        <div className="filters models-toolbar" style={{ marginBottom: 0 }}>
          <div className="search models-toolbar-search">
            <Icon name="search" size={17} />
            <input className="input" placeholder="Поиск по имени, URL…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <div className="models-toolbar-controls">
            <Button variant={sortOpen ? "primary" : "ghost"} icon="arrowUpDown" onClick={() => { setSortOpen((v) => !v); setFiltersOpen(false); }}>Сортировка</Button>
            <Button variant={filtersOpen ? "primary" : "ghost"} icon="sliders" onClick={() => { setFiltersOpen((v) => !v); setSortOpen(false); }}>Фильтры</Button>
            <Button variant={selecting ? "primary" : "secondary"} onClick={() => setSelecting((v) => !v)} style={{ width: 96 }}>{selecting ? "Готово" : "Выбрать"}</Button>
          </div>
        </div>
        {sortOpen && (
          <div className="dataset-filters-grid">
            <div style={{ gridColumn: "1 / -1" }}><Field label="Сортировать по:">
              <div className="sort-tabs" style={{ width: "100%" }}>
                {(["date", "az"] as const).map((f) => (
                  <button key={f} className={`sort-tab${sortField === f ? " active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                    onClick={() => { if (sortField === f) setSortDir((d) => d === "asc" ? "desc" : "asc"); else { setSortField(f); setSortDir("desc"); } }}>
                    {f === "date" ? "Дата" : "Алфавит"}
                    {sortField === f && <Icon name={sortDir === "desc" ? "arrowDown" : "arrowUp"} size={13} />}
                  </button>
                ))}
              </div>
            </Field></div>
          </div>
        )}
        {filtersOpen && (
          <div className="dataset-filters-grid">
            <Field label="Провайдер">
              <SelectDropdown
                value={providerFilter}
                options={[{ value: "all", label: "Все" }, ...PROVIDERS.map((p) => ({ value: p, label: p }))]}
                onChange={setProviderFilter}
              />
            </Field>
            <Field label="Статус">
              <SelectDropdown
                value={statusFilter}
                options={[
                  { value: "all", label: "Все" },
                  { value: "online", label: "Онлайн" },
                  { value: "offline", label: "Офлайн" },
                  { value: "unchecked", label: "Не проверено" },
                ]}
                onChange={setStatusFilter}
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
      {loading && !data ? (
        <Skeleton h={200} />
      ) : filtered.length ? (
        <div className="model-grid">
          {filtered.map((m) => (
            <Card key={m.id} className={`model-card${selecting && selected.has(m.id) ? " row-selected" : ""}`}
              hover={!selecting} style={{ cursor: selecting ? "pointer" : undefined }}
              onClick={() => selecting ? toggleSelect(m.id) : setEdit(m)}>
              <div className="spread" style={{ marginBottom: 5 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flex: 1 }}>
                  <div className="ds-title" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</div>
                </div>
                <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                  <Tag tone={m.online === false ? "red" : m.online ? "green" : undefined}>
                    <Dot tone={m.online === false ? "red" : m.online ? "green" : "gray"} />
                    {m.online === false ? "офлайн" : m.online ? "онлайн" : "не проверено"}
                  </Tag>
                  {!selecting && (
                    <button className="icon-btn" title="Проверить связь" onClick={(e) => { e.stopPropagation(); check(m.id); }}>
                      {checking === m.id ? <Spinner /> : <Icon name="refresh" size={15} />}
                    </button>
                  )}
                </div>
              </div>
              <div className="run-meta-line" style={{ margin: "0 0 8px" }}>
                <Tag>{m.provider}</Tag>
                {m.hasApiKey && <Tag>API key</Tag>}
              </div>
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">Input token limit</span><span className="v">{(m.ctx / 1024).toFixed(0)}k</span></div>
                <div className="ds-stat"><span className="k">Output token limit</span><span className="v">{m.maxTokens ? `${(m.maxTokens / 1024).toFixed(0)}k` : "—"}</span></div>
                <div className="ds-stat span-full"><span className="k">URL</span><span className="v" style={{ fontSize: 11.5 }}>{m.baseUrl || "—"}</span></div>
                {m.createdAt && <div className="ds-stat span-full"><span className="k">Создана</span><span className="v">{formatDateOnly(m.createdAt, settings?.date_format ?? "mdy")}</span></div>}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState
            icon="cpu"
            title={data?.length ? "Ничего не найдено" : "Нет моделей"}
            text={data?.length ? "Измените поиск или фильтры." : "Добавьте LLM-эндпоинт, чтобы запускать прогоны."}
            action={!data?.length ? <Button variant="primary" icon="plus" onClick={() => setAdding(true)}>Добавить модель</Button> : undefined}
          />
        </Card>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button className="archive-link" onClick={() => nav("/models/archive")}>
          <Icon name="archive" size={15} /> Архив
        </button>
      </div>

      {selecting && (
        <SelectionBar
          count={selected.size}
          noun="модель"
          actionLabel="Архивировать"
          actionIcon="archive"
          busy={archiving}
          onAction={() => setConfirm(true)}
          onCancel={cancelSelect}
        />
      )}

      {confirm && (
        <ConfirmDialog
          title="Архивировать модели?"
          description={`${selected.size === 1 ? "1 модель будет перемещена в архив." : `${selected.size} моделей будут перемещены в архив.`} Вернуть можно из раздела «Архив».`}
          confirmLabel="Архивировать"
          busy={archiving}
          onConfirm={doArchive}
          onCancel={() => setConfirm(false)}
        />
      )}

      {adding && <ModelModal draft={newDraft} onDraftChange={setNewDraft} onClose={() => setAdding(false)} onDone={() => { setAdding(false); setNewDraft(defaultDraft()); reload(); }} />}
      {edit && <ModelModal initial={edit} onClose={() => setEdit(null)} onDone={() => { setEdit(null); reload(); }} />}
    </div>
  );
}
