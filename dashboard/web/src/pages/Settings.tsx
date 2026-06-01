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
  const [appearance, setAppearance] = useState<Appearance>(loadAppearance());
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState(false);

  useEffect(() => {
    if (data) setForm({ mlflow_tracking_uri: data.mlflow_tracking_uri, datasets_dir: data.datasets_dir, default_mode: data.default_mode, default_episode: data.default_episode });
  }, [data]);

  function setAppr(patch: Partial<Appearance>) {
    const next = { ...appearance, ...patch };
    setAppearance(next);
    applyAppearance(next);
  }

  async function save() {
    setBusy(true); setOk(false);
    try {
      await api.saveSettings({ ...form, ...appearance });
      setOk(true);
    } finally { setBusy(false); }
  }

  return (
    <div className="stack" style={{ maxWidth: 760 }}>
      <Card>
        <h2 className="section-title">Сервер и подключения</h2>
        <Row label="MLflow tracking URI" hint="где хранятся прогоны">
          <input className="input mono" value={form.mlflow_tracking_uri ?? ""} onChange={(e) => setForm((s) => ({ ...s, mlflow_tracking_uri: e.target.value }))} />
        </Row>
        <Row label="Каталог датасетов" hint="datasets/ в корне проекта">
          <input className="input mono" value={form.datasets_dir ?? ""} onChange={(e) => setForm((s) => ({ ...s, datasets_dir: e.target.value }))} />
        </Row>
        <Row label="Режим бюджета по умолчанию">
          <select className="input" value={form.default_mode ?? "local"} onChange={(e) => setForm((s) => ({ ...s, default_mode: e.target.value }))}>
            <option value="local">local</option><option value="cloud">cloud</option>
          </select>
        </Row>
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
