import { useState } from "react";
import { api, type ModelRec } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Dot, EmptyState, Field, Modal, Skeleton, Spinner, Tag } from "../components/ui";
import { Icon } from "../components/Icon";

const PROVIDERS = ["OpenAI-совместимый", "vLLM", "Gemini", "LiteLLM"];
const needsBaseUrl = (provider: string) => provider === "OpenAI-совместимый" || provider === "vLLM";

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
        <div style={{ flex: 1 }} />
        <Button variant="ghost" onClick={onClose}>Отмена</Button>
        <Button variant="primary" onClick={save} disabled={busy || !f.name}>{busy ? <Spinner /> : "Сохранить"}</Button>
      </>}>
      <div className="stack" style={{ gap: 14 }}>
        <Field label="Имя модели" hint="как в --model (напр. Qwen/Qwen3-32B)"><input className="input mono" value={f.name} onChange={(e) => set("name", e.target.value)} /></Field>
        <div className="grid-2">
          <Field label="Провайдер"><select className="input" value={f.provider} onChange={(e) => set("provider", e.target.value)}>{PROVIDERS.map((p) => <option key={p}>{p}</option>)}</select></Field>
          <Field label="Контекст (токены)"><input className="input mono" value={f.ctx} onChange={(e) => set("ctx", e.target.value.replace(/\D/g, ""))} /></Field>
        </div>
        {showBaseUrl && <Field label="Base URL"><input className="input mono" value={f.baseUrl} onChange={(e) => set("baseUrl", e.target.value)} placeholder="http://host:8000/v1" /></Field>}
        <Field label="API-ключ" hint={initial?.hasApiKey ? "оставьте пустым, чтобы не менять" : "необязательно"}><input className="input" type="password" value={f.apiKey} onChange={(e) => set("apiKey", e.target.value)} placeholder="••••••••" /></Field>
        <div className="grid-2">
          <Field label="Температура по умолч."><input className="input mono" value={f.temp} onChange={(e) => set("temp", e.target.value)} /></Field>
          <Field label="Макс. токенов"><input className="input mono" value={f.maxTokens} onChange={(e) => set("maxTokens", e.target.value.replace(/\D/g, ""))} /></Field>
        </div>
        {test && <div style={{ fontSize: 13, color: test === "Соединение есть" ? "var(--green)" : test === "…" ? "var(--text-dim)" : "var(--red)" }}>{test === "…" ? <Spinner /> : test}</div>}
      </div>
    </Modal>
  );
}

export default function Models() {
  const { data, loading, reload } = useAsync(() => api.listModels(), []);
  const [edit, setEdit] = useState<ModelRec | null>(null);
  const [adding, setAdding] = useState(false);
  const [checking, setChecking] = useState<string | null>(null);

  async function check(id: string) {
    setChecking(id);
    try { await api.checkModel(id); reload(); } finally { setChecking(null); }
  }

  return (
    <div>
      <div className="spread" style={{ marginBottom: 16 }}>
        <span className="muted" style={{ fontSize: 13.5 }}>Реестр LLM-эндпоинтов. Имя передаётся раннеру через <span className="mono">--model</span>.</span>
        <Button variant="primary" icon="plus" onClick={() => setAdding(true)}>Добавить модель</Button>
      </div>
      <Card style={{ padding: 0 }}>
        {loading && !data ? (
          <div style={{ padding: 20 }}><Skeleton h={160} /></div>
        ) : data && data.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Модель</th><th>Провайдер</th><th>Base URL</th><th>Контекст</th><th>Ключ</th><th>Статус</th><th></th></tr></thead>
              <tbody>
                {data.map((m) => (
                  <tr key={m.id}>
                    <td className="mono">{m.name}</td>
                    <td><Tag>{m.provider}</Tag></td>
                    <td className="mono faint">{m.baseUrl || "—"}</td>
                    <td className="mono faint">{(m.ctx / 1024).toFixed(0)}k</td>
                    <td>{m.hasApiKey ? <Tag tone="green">есть</Tag> : <span className="faint">—</span>}</td>
                    <td><span className="row" style={{ gap: 7 }}><Dot tone={m.online === false ? "red" : m.online ? "green" : "gray"} />{m.online === false ? "офлайн" : m.online ? "онлайн" : "не проверено"}</span></td>
                    <td>
                      <div className="row" style={{ gap: 4, justifyContent: "flex-end" }}>
                        <button className="icon-btn" title="Проверить связь" onClick={() => check(m.id)}>{checking === m.id ? <Spinner /> : <Icon name="refresh" size={16} />}</button>
                        <button className="icon-btn" title="Редактировать" onClick={() => setEdit(m)}><Icon name="edit" size={16} /></button>
                        <button className="icon-btn" title="Удалить" onClick={() => api.deleteModel(m.id).then(reload)}><Icon name="trash" size={16} /></button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon="cpu" title="Нет моделей" text="Добавьте LLM-эндпоинт, чтобы запускать прогоны." action={<Button variant="primary" icon="plus" onClick={() => setAdding(true)}>Добавить модель</Button>} />
        )}
      </Card>

      {adding && <ModelModal onClose={() => setAdding(false)} onDone={() => { setAdding(false); reload(); }} />}
      {edit && <ModelModal initial={edit} onClose={() => setEdit(null)} onDone={() => { setEdit(null); reload(); }} />}
    </div>
  );
}
