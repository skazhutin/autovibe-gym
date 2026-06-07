import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate, useOutletContext } from "react-router-dom";
import { api, type ModelRec } from "../lib/api";
import type { SetHeaderAction } from "../components/Layout";
import { useAsync } from "../lib/hooks";
import { Button, Card, Dot, EmptyState, Field, Modal, SelectDropdown, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const PROVIDERS = ["OpenAI-совместимый", "vLLM", "Gemini", "LiteLLM"];
const needsBaseUrl = (provider: string) => provider === "OpenAI-совместимый" || provider === "vLLM";

function Info({ text }: { text: string }) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const dotRef = useRef<HTMLSpanElement>(null);
  function show() {
    const rect = dotRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ top: rect.top, left: rect.left + rect.width / 2 });
    setVisible(true);
  }
  return (
    <>
      <span ref={dotRef} className="info-dot" aria-label={text} onMouseEnter={show} onMouseLeave={() => setVisible(false)}>?</span>
      {visible && createPortal(<div className="tooltip-portal" style={{ top: pos.top, left: pos.left }}>{text}</div>, document.body)}
    </>
  );
}

function FI({ label, info, children }: { label: string; info: string; children: ReactNode }) {
  return (
    <Field label={<span className="field-info-label">{label}<Info text={info} /></span>}>
      {children}
    </Field>
  );
}

function ModelModal({ initial, onClose, onDone }: { initial?: ModelRec; onClose: () => void; onDone: () => void }) {
  const [f, setF] = useState({
    name: initial?.name ?? "",
    provider: initial?.provider ?? PROVIDERS[0],
    baseUrl: initial?.baseUrl ?? "",
    apiKey: "",
    ctx: initial?.ctx ?? 32768,
    temp: initial?.temp ?? 0.4,
    maxTokens: initial?.maxTokens ?? 8192,
  });
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<string | null>(null);
  const set = (k: string, v: string | number) => setF((s) => ({ ...s, [k]: v }));
  const showBaseUrl = needsBaseUrl(f.provider);

  async function save() {
    if (!f.name) return;
    setBusy(true);
    const payload = {
      ...f,
      baseUrl: showBaseUrl ? f.baseUrl : "",
      ctx: Number(f.ctx),
      temp: Number(f.temp),
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
    <Modal title={initial ? "Редактировать модель" : "Добавить модель"} width={520} onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={check}>Проверить связь</Button>
        {initial && <Button variant="danger" onClick={() => api.deleteModel(initial.id).then(onDone)} disabled={busy}>Удалить</Button>}
        <div style={{ flex: 1 }} />
        <Button variant="ghost" onClick={onClose}>Отмена</Button>
        <Button variant="primary" onClick={save} disabled={busy || !f.name}>{busy ? <Spinner /> : "Сохранить"}</Button>
      </>}>
      <div className="stack" style={{ gap: 14 }}>
        <FI label="Имя модели" info="Название модели как у провайдера, напр. anthropic/claude-opus-4-5, gemini-2.5-flash"><input className="input mono" value={f.name} onChange={(e) => set("name", e.target.value)} /></FI>
        <div className="grid-2">
          <FI label="Провайдер" info="Тип API: OpenAI-совместимый и vLLM требуют Base URL, Gemini — Google API Key, LiteLLM — любой провайдер через litellm"><SelectDropdown value={f.provider} options={PROVIDERS.map((p) => ({ value: p, label: p }))} onChange={(v) => set("provider", v)} /></FI>
          <FI label="Input limit" info="Максимум токенов в запросе (контекстное окно модели). Если превышено — прогон завершится с ошибкой. Напр. 32768 для большинства моделей, 128000 для GPT-4o."><input className="input mono" value={f.ctx} onChange={(e) => set("ctx", e.target.value.replace(/\D/g, ""))} /></FI>
        </div>
        {showBaseUrl && <FI label="Base URL" info="Базовый адрес API. Для vLLM/локального сервера: http://host:8000/v1. Для OpenRouter: https://openrouter.ai/api/v1"><input className="input mono" value={f.baseUrl} onChange={(e) => set("baseUrl", e.target.value)} placeholder="http://host:8000/v1" /></FI>}
        <FI label="API-ключ" info={initial?.hasApiKey ? "Ключ уже сохранён — оставьте пустым, чтобы не менять" : "Ключ авторизации у провайдера. Для локального vLLM можно оставить пустым."}><input className="input" type="password" value={f.apiKey} onChange={(e) => set("apiKey", e.target.value)} placeholder="••••••••" /></FI>
        <div className="grid-2">
          <FI label="Температура" info="Случайность ответов модели: 0 — всегда одинаково, 1 — очень вариативно. Рекомендуется 0.3–0.6 для кода."><input className="input mono" value={f.temp} onChange={(e) => set("temp", e.target.value)} /></FI>
          <FI label="Output limit" info="Максимум токенов в одном ответе. Если модель упирается в этот лимит — ответ обрезается и прогон завершается с ошибкой."><input className="input mono" value={f.maxTokens} onChange={(e) => set("maxTokens", e.target.value.replace(/\D/g, ""))} /></FI>
        </div>
        {test && <div style={{ fontSize: 13, color: test === "Соединение есть" ? "var(--green)" : test === "…" ? "var(--text-dim)" : "var(--red)" }}>{test === "…" ? <Spinner /> : test}</div>}
      </div>
    </Modal>
  );
}

export default function Models() {
  const nav = useNavigate();
  const setHeaderAction = useOutletContext<SetHeaderAction>();
  const { data, loading, reload } = useAsync(() => api.listModels(), []);
  const [edit, setEdit] = useState<ModelRec | null>(null);
  const [adding, setAdding] = useState(false);
  const [checking, setChecking] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [providerFilter, setProviderFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    setHeaderAction({ label: "Новая модель", icon: "plus", onClick: () => setAdding(true) });
    return () => setHeaderAction(null);
  }, [setHeaderAction]);

  async function check(id: string) {
    setChecking(id);
    try { await api.checkModel(id); reload(); } finally { setChecking(null); }
  }

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (data ?? []).filter((m) => {
      if (term && !`${m.name} ${m.baseUrl ?? ""} ${m.provider}`.toLowerCase().includes(term)) return false;
      if (providerFilter !== "all" && m.provider !== providerFilter) return false;
      if (statusFilter === "online" && !m.online) return false;
      if (statusFilter === "offline" && m.online !== false) return false;
      if (statusFilter === "unchecked" && m.online !== null && m.online !== undefined) return false;
      return true;
    });
  }, [data, q, providerFilter, statusFilter]);

  return (
    <div className="stack" style={{ gap: 18 }}>
      <Card className="dataset-toolbar">
        <div className="filters" style={{ marginBottom: 0 }}>
          <div className="search">
            <Icon name="search" size={17} />
            <input className="input" placeholder="Поиск по имени, URL…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <SelectDropdown
            value={providerFilter}
            options={[{ value: "all", label: "Все провайдеры" }, ...PROVIDERS.map((p) => ({ value: p, label: p }))]}
            onChange={setProviderFilter}
          />
          <SelectDropdown
            value={statusFilter}
            options={[
              { value: "all", label: "Все статусы" },
              { value: "online", label: "Онлайн" },
              { value: "offline", label: "Офлайн" },
              { value: "unchecked", label: "Не проверено" },
            ]}
            onChange={setStatusFilter}
          />
        </div>
      </Card>
      {loading && !data ? (
        <Skeleton h={200} />
      ) : filtered.length ? (
        <div className="model-grid">
          {filtered.map((m) => (
            <Card key={m.id} className="model-card" hover onClick={() => setEdit(m)}>
              <div className="spread" style={{ marginBottom: 6 }}>
                <div className="ds-title" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{m.name}</div>
                <div className="row" style={{ gap: 4 }}>
                  <button className="icon-btn" title="Проверить связь" onClick={(e) => { e.stopPropagation(); check(m.id); }}>
                    {checking === m.id ? <Spinner /> : <Icon name="refresh" size={15} />}
                  </button>
                  <button className="icon-btn model-card-del" title="Архивировать" onClick={(e) => { e.stopPropagation(); api.archiveModels([m.id]).then(reload); }}>
                    <Icon name="archive" size={15} />
                  </button>
                </div>
              </div>
              <div className="run-meta-line" style={{ margin: "0 0 10px" }}>
                <Tag>{m.provider}</Tag>
                <span className="row" style={{ gap: 5 }}>
                  <Dot tone={m.online === false ? "red" : m.online ? "green" : "gray"} />
                  <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{m.online === false ? "офлайн" : m.online ? "онлайн" : "не проверено"}</span>
                </span>
                {m.hasApiKey && <Tag tone="green">API key</Tag>}
              </div>
              <div className="ds-stats">
                <div className="ds-stat"><span className="k">Input limit</span><span className="v">{(m.ctx / 1024).toFixed(0)}k</span></div>
                <div className="ds-stat"><span className="k">Output limit</span><span className="v">{m.maxTokens ? `${(m.maxTokens / 1024).toFixed(0)}k` : "—"}</span></div>
                <div className="ds-stat" style={{ gridColumn: "1 / -1" }}><span className="k">Base URL</span><span className="v" style={{ fontSize: 11.5 }}>{m.baseUrl || "—"}</span></div>
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

      {adding && <ModelModal onClose={() => setAdding(false)} onDone={() => { setAdding(false); reload(); }} />}
      {edit && <ModelModal initial={edit} onClose={() => setEdit(null)} onDone={() => { setEdit(null); reload(); }} />}
    </div>
  );
}
