import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Spinner } from "../components/ui";
import { Icon } from "../components/Icon";
import { applyAppearance, loadAppearance, type Appearance } from "../lib/theme";

const ACCENTS = ["#FFDD2D", "#FF8A3D", "#4BBE85", "#3B6CC7", "#9B4DCA"];

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

function Row({ label, info, children }: { label: string; info?: string; children: ReactNode }) {
  return (
    <div className="set-row">
      <div>
        <div className="label" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          {label}
          {info && <Info text={info} />}
        </div>
      </div>
      <div className="control">{children}</div>
    </div>
  );
}

export default function Settings() {
  const { data } = useAsync(() => api.getSettings(), []);
  const [form, setForm] = useState<Record<string, string>>({});
  const [remoteOn, setRemoteOn] = useState(false);
  const [appearance, setAppearance] = useState<Appearance>(loadAppearance());
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [validErr, setValidErr] = useState<string | null>(null);
  const [check, setCheck] = useState<{ msg: string; ok: boolean } | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (!data) return;
    setForm({
      mlflow_tracking_uri: data.mlflow_tracking_uri, datasets_dir: data.datasets_dir,
      remote_ssh: data.remote_ssh ?? "", remote_ssh_opts: data.remote_ssh_opts ?? "",
      remote_repo: data.remote_repo ?? "", remote_python: data.remote_python ?? "",
      remote_runs_dir: data.remote_runs_dir ?? "",
      remote_password: data.remote_has_password ? "********" : "",
    });
    setRemoteOn(!!data.remote_enabled);
    setDirty(false);
  }, [data]);

  const set = (k: string, v: string) => { setForm((s) => ({ ...s, [k]: v })); setDirty(true); };
  const toggleRemote = (v: boolean) => { setRemoteOn(v); setDirty(true); };

  function setAppr(patch: Partial<Appearance>) {
    const next = { ...appearance, ...patch };
    setAppearance(next);
    applyAppearance(next);
    setDirty(true);
  }

  function validate(): string | null {
    if (!form.mlflow_tracking_uri?.trim()) return "MLflow tracking URI не заполнен";
    if (!form.datasets_dir?.trim()) return "Каталог датасетов не заполнен";
    if (remoteOn && !form.remote_ssh?.trim()) return "SSH (user@host) не заполнен — обязательно при включённом сервере";
    return null;
  }

  async function save() {
    const err = validate();
    if (err) { setValidErr(err); return; }
    setValidErr(null);
    setBusy(true);
    const payload: Record<string, unknown> = { ...form, remote_enabled: remoteOn, ...appearance };
    if (payload.remote_password === "********") delete payload.remote_password;
    try {
      await api.saveSettings(payload);
      setDirty(false);
    } finally { setBusy(false); }
  }

  async function runCheck() {
    setChecking(true); setCheck(null);
    try {
      await api.saveSettings({ ...form, remote_enabled: remoteOn, remote_password: form.remote_password === "********" ? undefined : form.remote_password } as never);
      const r = await api.remoteCheck();
      setCheck({ ok: r.ok, msg: r.ok ? `Связь есть · repo ${r.repo ? "✓" : "✗"} · gym ${r.gym ? "✓" : "✗"}` : (r.error || r.output || "недоступно") });
    } catch (e) {
      setCheck({ ok: false, msg: e instanceof Error ? e.message : "ошибка" });
    } finally { setChecking(false); }
  }

  return (
    <div className="stack" style={{ maxWidth: 820 }}>
      <Card>
        <h2 className="section-title">Сервер и подключения</h2>
        <Row label="MLflow tracking URI" info="Адрес MLflow-сервера, где хранятся прогоны. Обычно file:./mlruns или http://localhost:5000">
          <input className="input mono" value={form.mlflow_tracking_uri ?? ""} onChange={(e) => set("mlflow_tracking_uri", e.target.value)} />
        </Row>
        <Row label="Каталог датасетов" info="Путь к папке datasets/ в корне проекта. Все датасеты читаются и сохраняются туда.">
          <input className="input mono" value={form.datasets_dir ?? ""} onChange={(e) => set("datasets_dir", e.target.value)} />
        </Row>
      </Card>

      <Card>
        <h2 className="section-title">Выполнение на сервере (SSH)</h2>
        <div className="muted" style={{ fontSize: 13, marginTop: -8, marginBottom: 8 }}>
          Когда включено, gym и обучение моделей выполняются на сервере по SSH, а сайт остаётся
          локальным и лишь подтягивает результаты — компьютер не нагружается. Рекомендуется
          настроить SSH-ключ (<span className="mono">ssh-copy-id</span>); пароль — запасной вариант.
        </div>
        <Row label="Запускать прогоны на сервере" info="Если выключено, прогоны запускаются локально на этой машине.">
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{remoteOn ? "Сервер" : "Локально"}</span>
            <div className={`toggle${remoteOn ? " on" : ""}`} onClick={() => toggleRemote(!remoteOn)} />
          </div>
        </Row>
        {remoteOn && (
          <>
            <Row label="SSH (user@host)" info="Адрес сервера в формате user@host или user@ip. Например: booml@10.8.52.11">
              <input className="input mono" placeholder="booml@10.8.52.11" value={form.remote_ssh ?? ""} onChange={(e) => set("remote_ssh", e.target.value)} />
            </Row>
            <Row label="Доп. SSH-опции" info="Дополнительные флаги для ssh. Например: -p 2222 для нестандартного порта.">
              <input className="input mono" value={form.remote_ssh_opts ?? ""} onChange={(e) => set("remote_ssh_opts", e.target.value)} />
            </Row>
            <Row label="Путь к репозиторию на сервере" info="Абсолютный путь к autovibe-gym на сервере. Например: /home/booml/autovibe-gym-current">
              <input className="input mono" placeholder="/home/booml/autovibe-gym-current" value={form.remote_repo ?? ""} onChange={(e) => set("remote_repo", e.target.value)} />
            </Row>
            <Row label="Python сервера (venv)" info="Путь к python из виртуального окружения на сервере.">
              <input className="input mono" placeholder="/home/booml/autovibe-gym/.venv/bin/python" value={form.remote_python ?? ""} onChange={(e) => set("remote_python", e.target.value)} />
            </Row>
            <Row label="Каталог прогонов на сервере" info="Рабочие папки эпизодов на сервере. Например: /home/booml/dash_runs">
              <input className="input mono" placeholder="/home/booml/dash_runs" value={form.remote_runs_dir ?? ""} onChange={(e) => set("remote_runs_dir", e.target.value)} />
            </Row>
            <Row label="Пароль SSH (необязательно)" info="Используйте только как запасной вариант. Рекомендуется SSH-ключ (ssh-copy-id). Пароль хранится только локально в data/.">
              <input className="input" type="password" value={form.remote_password ?? ""} onChange={(e) => set("remote_password", e.target.value)} placeholder="••••••••" />
            </Row>
            <div className="row" style={{ marginTop: 6 }}>
              <Button variant="secondary" onClick={runCheck} disabled={checking}>{checking ? <Spinner /> : <Icon name="refresh" size={16} />} Проверить связь</Button>
              {check && <span style={{ fontSize: 13, color: check.ok ? "var(--green)" : "var(--red)" }}>{check.msg}</span>}
            </div>
          </>
        )}
      </Card>

      <Card>
        <h2 className="section-title">Внешний вид</h2>
        <Row label="Тема" info="Светлая или тёмная тема интерфейса на базе цвета #333.">
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{appearance.theme === "dark" ? "Тёмная" : "Светлая"}</span>
            <div className={`toggle${appearance.theme === "dark" ? " on" : ""}`} onClick={() => setAppr({ theme: appearance.theme === "dark" ? "light" : "dark" })} />
          </div>
        </Row>
        <Row label="Акцентный цвет" info="Основной цвет интерфейса — кнопки, ссылки, активные элементы.">
          <div className="swatch-row" style={{ justifyContent: "flex-end" }}>
            {ACCENTS.map((c) => (
              <span key={c} className={`swatch${appearance.accent.toLowerCase() === c.toLowerCase() ? " active" : ""}`} style={{ background: c }} onClick={() => setAppr({ accent: c })} />
            ))}
          </div>
        </Row>
        <Row label={`Скругление: ${appearance.radius}px`} info="Радиус скругления углов карточек и кнопок. От 8px (квадратный) до 24px (круглый).">
          <input className="range" type="range" min={8} max={24} value={appearance.radius} onChange={(e) => setAppr({ radius: Number(e.target.value) })} />
        </Row>
      </Card>

      {dirty && createPortal(
        <div className="settings-save-bar">
          {validErr && (
            <div className="settings-save-error">
              <Icon name="x" size={14} /> {validErr}
            </div>
          )}
          <div className="settings-save-row">
            <span className="settings-unsaved-label">У вас есть несохранённые изменения</span>
            <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить"}</Button>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
