import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, Spinner } from "../components/ui";
import { Icon } from "../components/Icon";
import { applyAppearance, loadAppearance, type Appearance } from "../lib/theme";

const ACCENTS = ["#FFDD2D", "#FF8A3D", "#4BBE85", "#3B6CC7", "#9B4DCA"];

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="set-row">
      <div>
        <div className="label">{label}</div>
        {hint && <div className="hint">{hint}</div>}
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
  const [ok, setOk] = useState(false);
  const [check, setCheck] = useState<{ msg: string; ok: boolean } | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (!data) return;
    setForm({
      mlflow_tracking_uri: data.mlflow_tracking_uri, datasets_dir: data.datasets_dir,
      default_mode: data.default_mode,
      remote_ssh: data.remote_ssh ?? "", remote_ssh_opts: data.remote_ssh_opts ?? "",
      remote_repo: data.remote_repo ?? "", remote_python: data.remote_python ?? "",
      remote_runs_dir: data.remote_runs_dir ?? "",
      remote_password: data.remote_has_password ? "********" : "",
    });
    setRemoteOn(!!data.remote_enabled);
  }, [data]);

  const set = (k: string, v: string) => setForm((s) => ({ ...s, [k]: v }));

  function setAppr(patch: Partial<Appearance>) {
    const next = { ...appearance, ...patch };
    setAppearance(next);
    applyAppearance(next);
  }

  async function save() {
    setBusy(true); setOk(false);
    const payload: Record<string, unknown> = { ...form, remote_enabled: remoteOn, ...appearance };
    if (payload.remote_password === "********") delete payload.remote_password; // keep stored one
    try {
      await api.saveSettings(payload);
      setOk(true);
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
        <Row label="MLflow tracking URI" hint="где хранятся прогоны">
          <input className="input mono" value={form.mlflow_tracking_uri ?? ""} onChange={(e) => set("mlflow_tracking_uri", e.target.value)} />
        </Row>
        <Row label="Каталог датасетов" hint="datasets/ в корне проекта">
          <input className="input mono" value={form.datasets_dir ?? ""} onChange={(e) => set("datasets_dir", e.target.value)} />
        </Row>
        <Row label="Режим бюджета по умолчанию">
          <select className="input" value={form.default_mode ?? "local"} onChange={(e) => set("default_mode", e.target.value)}>
            <option value="local">local</option><option value="cloud">cloud</option>
          </select>
        </Row>
      </Card>

      <Card>
        <h2 className="section-title">Выполнение на сервере (SSH)</h2>
        <div className="muted" style={{ fontSize: 13, marginTop: -8, marginBottom: 8 }}>
          Когда включено, gym и обучение моделей выполняются на сервере по SSH, а сайт остаётся
          локальным и лишь подтягивает результаты — компьютер не нагружается. Рекомендуется
          настроить SSH-ключ (<span className="mono">ssh-copy-id</span>); пароль — запасной вариант.
        </div>
        <Row label="Запускать прогоны на сервере" hint="иначе — локально на этой машине">
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{remoteOn ? "Сервер" : "Локально"}</span>
            <div className={`toggle${remoteOn ? " on" : ""}`} onClick={() => setRemoteOn((v) => !v)} />
          </div>
        </Row>
        <Row label="SSH (user@host)"><input className="input mono" placeholder="booml@10.8.52.11" value={form.remote_ssh ?? ""} onChange={(e) => set("remote_ssh", e.target.value)} /></Row>
        <Row label="Доп. SSH-опции" hint="напр. -p 2222"><input className="input mono" value={form.remote_ssh_opts ?? ""} onChange={(e) => set("remote_ssh_opts", e.target.value)} /></Row>
        <Row label="Путь к репозиторию на сервере"><input className="input mono" placeholder="/home/booml/autovibe-gym-current" value={form.remote_repo ?? ""} onChange={(e) => set("remote_repo", e.target.value)} /></Row>
        <Row label="Python сервера (venv)"><input className="input mono" placeholder="/home/booml/autovibe-gym/.venv/bin/python" value={form.remote_python ?? ""} onChange={(e) => set("remote_python", e.target.value)} /></Row>
        <Row label="Каталог прогонов на сервере" hint="рабочие папки эпизодов"><input className="input mono" placeholder="/home/booml/dash_runs" value={form.remote_runs_dir ?? ""} onChange={(e) => set("remote_runs_dir", e.target.value)} /></Row>
        <Row label="Пароль SSH (необязательно)" hint="лучше ключ; хранится только локально в data/">
          <input className="input" type="password" value={form.remote_password ?? ""} onChange={(e) => set("remote_password", e.target.value)} placeholder="••••••••" />
        </Row>
        <div className="row" style={{ marginTop: 6 }}>
          <Button variant="secondary" onClick={runCheck} disabled={checking}>{checking ? <Spinner /> : <Icon name="refresh" size={16} />} Проверить связь</Button>
          {check && <span style={{ fontSize: 13, color: check.ok ? "var(--green)" : "var(--red)" }}>{check.msg}</span>}
        </div>
      </Card>

      <Card>
        <h2 className="section-title">Внешний вид</h2>
        <Row label="Тема" hint="светлая / тёмная на базе #333">
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{appearance.theme === "dark" ? "Тёмная" : "Светлая"}</span>
            <div className={`toggle${appearance.theme === "dark" ? " on" : ""}`} onClick={() => setAppr({ theme: appearance.theme === "dark" ? "light" : "dark" })} />
          </div>
        </Row>
        <Row label="Акцентный цвет">
          <div className="swatch-row" style={{ justifyContent: "flex-end" }}>
            {ACCENTS.map((c) => (
              <span key={c} className={`swatch${appearance.accent.toLowerCase() === c.toLowerCase() ? " active" : ""}`} style={{ background: c }} onClick={() => setAppr({ accent: c })} />
            ))}
          </div>
        </Row>
        <Row label={`Скругление: ${appearance.radius}px`}>
          <input className="range" type="range" min={8} max={24} value={appearance.radius} onChange={(e) => setAppr({ radius: Number(e.target.value) })} />
        </Row>
      </Card>

      <div className="row">
        <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : "Сохранить настройки"}</Button>
        {ok && <span style={{ color: "var(--green)", fontSize: 13 }}><Icon name="check" size={14} /> сохранено</span>}
      </div>
    </div>
  );
}
