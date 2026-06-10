import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Button, Card, SelectDropdown, Spinner } from "../components/ui";
import { Icon } from "../components/Icon";
import { applyAppearance, loadAppearance, type Appearance } from "../lib/theme";
import { useI18n } from "../lib/i18n";

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
  const { language, setLanguage, t } = useI18n();
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
      date_format: data.date_format ?? "mdy",
      language: data.language ?? "ru",
      remote_ssh: data.remote_ssh ?? "", remote_ssh_opts: data.remote_ssh_opts ?? "",
      remote_repo: data.remote_repo ?? "", remote_python: data.remote_python ?? "",
      remote_runs_dir: data.remote_runs_dir ?? "",
      remote_password: data.remote_has_password ? "********" : "",
    });
    setAppearance({
      theme: data.theme === "dark" ? "dark" : "light",
      accent: data.accent,
      radius: data.radius,
      animations: data.animations ?? "on",
      overlayOpacity: data.overlay_opacity ?? loadAppearance().overlayOpacity,
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
    if (!form.mlflow_tracking_uri?.trim()) return t("settings.error.mlflow");
    if (!form.datasets_dir?.trim()) return t("settings.error.datasets");
    if (remoteOn && !form.remote_ssh?.trim()) return t("settings.error.remote");
    return null;
  }

  async function save() {
    const err = validate();
    if (err) { setValidErr(err); return; }
    setValidErr(null);
    setBusy(true);
    const payload: Record<string, unknown> = { ...form, remote_enabled: remoteOn, ...appearance, overlay_opacity: appearance.overlayOpacity };
    if (payload.remote_password === "********") delete payload.remote_password;
    try {
      await api.saveSettings(payload);
      if (form.language === "ru" || form.language === "en") setLanguage(form.language);
      setDirty(false);
    } finally { setBusy(false); }
  }

  async function runCheck() {
    setChecking(true); setCheck(null);
    try {
      await api.saveSettings({ ...form, remote_enabled: remoteOn, remote_password: form.remote_password === "********" ? undefined : form.remote_password } as never);
      const r = await api.remoteCheck();
      setCheck({ ok: r.ok, msg: r.ok ? `SSH ok · repo ${r.repo ? "✓" : "✗"} · runtime ${r.runtime ? "✓" : "✗"}` : (r.error || r.output || "unavailable") });
    } catch (e) {
      setCheck({ ok: false, msg: e instanceof Error ? e.message : "error" });
    } finally { setChecking(false); }
  }

  return (
    <div className="stack" style={{ maxWidth: 820 }}>
      <Card>
        <h2 className="section-title">{t("settings.server")}</h2>
        <Row label={t("settings.mlflow")} info={t("settings.info.mlflow")}>
          <input className="input mono" value={form.mlflow_tracking_uri ?? ""} onChange={(e) => set("mlflow_tracking_uri", e.target.value)} />
        </Row>
        <Row label={t("settings.datasetsDir")} info={t("settings.info.datasetsDir")}>
          <input className="input mono" value={form.datasets_dir ?? ""} onChange={(e) => set("datasets_dir", e.target.value)} />
        </Row>
      </Card>

      <Card>
        <h2 className="section-title">{t("settings.remote")}</h2>
        <div className="muted" style={{ fontSize: 13, marginTop: -8, marginBottom: 8 }}>
          {t("settings.remoteBlurb")}
        </div>
        <Row label={t("settings.remoteEnabled")} info={t("settings.info.remoteEnabled")}>
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{remoteOn ? t("settings.remoteEnabledState.server") : t("settings.remoteEnabledState.local")}</span>
            <div className={`toggle${remoteOn ? " on" : ""}`} onClick={() => toggleRemote(!remoteOn)} />
          </div>
        </Row>
        <div style={{ display: "grid", gridTemplateRows: remoteOn ? "1fr" : "0fr", transition: "grid-template-rows var(--dur-slow) var(--ease-standard)" }}>
          <div style={{ overflow: "hidden" }}>
            <Row label={t("settings.remoteSsh")} info={t("settings.info.remoteSsh")}>
              <input className="input mono" placeholder="user@host.example.com" value={form.remote_ssh ?? ""} onChange={(e) => set("remote_ssh", e.target.value)} />
            </Row>
            <Row label={t("settings.remoteSshOpts")} info={t("settings.info.remoteSshOpts")}>
              <input className="input mono" value={form.remote_ssh_opts ?? ""} onChange={(e) => set("remote_ssh_opts", e.target.value)} />
            </Row>
            <Row label={t("settings.remoteRepo")} info={t("settings.info.remoteRepo")}>
              <input className="input mono" placeholder="/home/user/autovibe-gym" value={form.remote_repo ?? ""} onChange={(e) => set("remote_repo", e.target.value)} />
            </Row>
            <Row label={t("settings.remotePython")} info={t("settings.info.remotePython")}>
              <input className="input mono" placeholder="/home/user/autovibe-gym/.venv/bin/python" value={form.remote_python ?? ""} onChange={(e) => set("remote_python", e.target.value)} />
            </Row>
            <Row label={t("settings.remoteRunsDir")} info={t("settings.info.remoteRunsDir")}>
              <input className="input mono" placeholder="/home/user/runs" value={form.remote_runs_dir ?? ""} onChange={(e) => set("remote_runs_dir", e.target.value)} />
            </Row>
            <Row label={t("settings.remotePassword")} info={t("settings.info.remotePassword")}>
              <input className="input" type="password" value={form.remote_password ?? ""} onChange={(e) => set("remote_password", e.target.value)} placeholder="••••••••" />
            </Row>
            <div className="row" style={{ marginTop: 6, marginBottom: 4 }}>
              <Button variant="secondary" onClick={runCheck} disabled={checking}>{checking ? <Spinner /> : <Icon name="refresh" size={16} />} {t("settings.checkConnection")}</Button>
              {check && <span style={{ fontSize: 13, color: check.ok ? "var(--green)" : "var(--red)" }}>{check.msg}</span>}
            </div>
          </div>
        </div>
      </Card>

      <Card>
        <h2 className="section-title">{t("settings.dashboard")}</h2>
        <Row label={t("settings.language")} info={t("settings.info.language")}>
          <SelectDropdown
            value={form.language ?? language}
            options={[
              { value: "ru", label: t("settings.language.ru") },
              { value: "en", label: t("settings.language.en") },
            ]}
            onChange={(v) => set("language", v)}
          />
        </Row>
        <Row label={t("settings.dateFormat")} info={t("settings.info.dateFormat")}>
          <SelectDropdown
            value={form.date_format ?? "mdy"}
            options={[
              { value: "mdy", label: "mm/dd/yyyy" },
              { value: "dmy", label: "dd.mm.yyyy" },
            ]}
            onChange={(v) => set("date_format", v)}
          />
        </Row>
        <Row label={t("settings.theme")} info={t("settings.info.theme")}>
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{appearance.theme === "dark" ? t("settings.theme.dark") : t("settings.theme.light")}</span>
            <div className={`toggle${appearance.theme === "dark" ? " on" : ""}`} onClick={() => setAppr({ theme: appearance.theme === "dark" ? "light" : "dark" })} />
          </div>
        </Row>
        <Row label={t("settings.accent")} info={t("settings.info.accent")}>
          <div className="swatch-row" style={{ justifyContent: "flex-end" }}>
            {ACCENTS.map((c) => (
              <span key={c} className={`swatch${appearance.accent.toLowerCase() === c.toLowerCase() ? " active" : ""}`} style={{ background: c }} onClick={() => setAppr({ accent: c })} />
            ))}
          </div>
        </Row>
        <Row label={t("settings.animations")} info={t("settings.info.animations")}>
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <span className="faint" style={{ fontSize: 13 }}>{appearance.animations === "on" ? t("settings.animations.on") : t("settings.animations.off")}</span>
            <div className={`toggle${appearance.animations === "on" ? " on" : ""}`} onClick={() => setAppr({ animations: appearance.animations === "on" ? "off" : "on" })} />
          </div>
        </Row>
        <Row label={`${t("settings.rounding")}: ${appearance.radius}px`} info={t("settings.info.rounding")}>
          <input className="range" type="range" min={8} max={24} value={appearance.radius} onChange={(e) => setAppr({ radius: Number(e.target.value) })} />
        </Row>
        <Row label={`${t("settings.overlay")}: ${appearance.overlayOpacity}%`} info={t("settings.info.overlay")}>
          <input className="range" type="range" min={0} max={100} step={5} value={appearance.overlayOpacity} onChange={(e) => setAppr({ overlayOpacity: Number(e.target.value) })} />
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
            <span className="settings-unsaved-label">{t("settings.unsaved")}</span>
            <Button variant="primary" onClick={save} disabled={busy}>{busy ? <Spinner /> : t("settings.save")}</Button>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
