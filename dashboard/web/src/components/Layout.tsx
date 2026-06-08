import { useEffect, useRef, useState, useCallback } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Icon } from "./Icon";
import { Button, Dot } from "./ui";
import { useAsync } from "../lib/hooks";
import { api } from "../lib/api";
import { useI18n } from "../lib/i18n";

const NAV = [
  { to: "/", labelKey: "nav.dashboard", icon: "dashboard", end: true },
  { to: "/new", labelKey: "nav.new", icon: "play" },
  { to: "/runs", labelKey: "nav.runs", icon: "runs" },
  { to: "/compare", labelKey: "nav.compare", icon: "compare" },
  { to: "/problems", labelKey: "nav.tasks", icon: "check2" },
  { to: "/models", labelKey: "nav.models", icon: "cpu" },
  { to: "/settings", labelKey: "nav.settings", icon: "settings" },
];

// Route -> header title/subtitle.
const META: { match: (p: string) => boolean; titleKey: string; subKey: string }[] = [
  { match: (p) => p === "/", titleKey: "meta.dashboard.title", subKey: "meta.dashboard.sub" },
  { match: (p) => p.startsWith("/new"), titleKey: "meta.new.title", subKey: "meta.new.sub" },
  { match: (p) => p === "/runs/archive", titleKey: "meta.runsArchive.title", subKey: "meta.runsArchive.sub" },
  { match: (p) => p.startsWith("/runs/"), titleKey: "meta.run.title", subKey: "meta.run.sub" },
  { match: (p) => p.startsWith("/runs"), titleKey: "meta.runs.title", subKey: "meta.runs.sub" },
  { match: (p) => p.startsWith("/compare"), titleKey: "meta.compare.title", subKey: "meta.compare.sub" },
  { match: (p) => p.startsWith("/problems/"), titleKey: "meta.task.title", subKey: "meta.task.sub" },
  { match: (p) => p.startsWith("/problems"), titleKey: "meta.tasks.title", subKey: "meta.tasks.sub" },
  { match: (p) => p.startsWith("/models"), titleKey: "meta.models.title", subKey: "meta.models.sub" },
  { match: (p) => p.startsWith("/settings"), titleKey: "meta.settings.title", subKey: "meta.settings.sub" },
];

function HeaderStatus() {
  const { t } = useI18n();
  // Reflects whether the LLM server (gemma/deepseek host) is reachable.
  const { data, error } = useAsync(() => api.serverHealth(), [], 15000);
  const online = !error && data?.online;
  const title = error
    ? t("status.dashboardBackendDown")
    : !data?.configured
    ? t("status.serverNotConfigured")
    : data.servers.map((s) => `${s.baseUrl} — ${s.online ? "онлайн" : s.error ?? s.status ?? "офлайн"}`).join("\n");
  return (
    <span className="status-pill" title={title}>
      <Dot tone={online ? "green" : "red"} pulse={false} />
      {online ? t("status.serverOnline") : t("status.serverOffline")}
    </span>
  );
}

export interface HeaderAction { label: string; icon?: string; onClick: () => void }
export type SetHeaderAction = (a: HeaderAction | null) => void;

export default function Layout() {
  const { t } = useI18n();
  const loc = useLocation();
  const nav = useNavigate();
  const meta = META.find((m) => m.match(loc.pathname)) ?? META[0];
  const [collapsed, setCollapsed] = useState(false);
  const [headerAction, setHeaderActionRaw] = useState<HeaderAction | null>(null);
  const setHeaderAction: SetHeaderAction = useCallback((a) => setHeaderActionRaw(a), []);
  const headerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    function measure() {
      if (headerRef.current) {
        document.documentElement.style.setProperty("--header-h", `${headerRef.current.offsetHeight}px`);
      }
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const isRuns = loc.pathname === "/runs" || loc.pathname.startsWith("/runs") && !loc.pathname.startsWith("/runs/");
  const isCompare = loc.pathname.startsWith("/compare");
  const showNewRun = (isRuns || isCompare) && loc.pathname !== "/new";

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [loc.pathname]);

  return (
    <div className={`app${collapsed ? " sidebar-collapsed" : ""}`}>
      <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
        <div className="sidebar-logo">
          <span className="mark">
            <svg className="brand-dumbbell" viewBox="0 0 30 30" aria-hidden="true">
              {/* clean symmetric dumbbell, rotated to a -45° diagonal */}
              <g transform="rotate(-45 15 15)">
                <rect x="12" y="13.4" width="6" height="3.2" rx="1.6" />
                <rect x="9.4" y="10" width="3.2" height="10" rx="1.5" />
                <rect x="17.4" y="10" width="3.2" height="10" rx="1.5" />
                <rect x="6.6" y="11.6" width="2.6" height="6.8" rx="1.3" />
                <rect x="20.8" y="11.6" width="2.6" height="6.8" rx="1.3" />
              </g>
            </svg>
          </span>
          <span className="sidebar-logo-text">AutoVibe Gym</span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`} title={collapsed ? t(n.labelKey) : undefined}>
              <Icon name={n.icon} size={19} strokeWidth={1.9} />
              <span className="nav-item-label">{t(n.labelKey)}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
      <button className={`sidebar-toggle${collapsed ? " collapsed" : ""}`} onClick={() => setCollapsed((v) => !v)} title={collapsed ? t("sidebar.expand") : t("sidebar.collapse")}>
        <Icon name={collapsed ? "chevronRight" : "chevronLeft"} size={14} strokeWidth={2} />
      </button>

      <div className="main">
        <header className="header" ref={headerRef}>
          <div>
            <h1>{t(meta.titleKey)}</h1>
            <div className="sub">{t(meta.subKey)}</div>
          </div>
          <div className="header-right">
            <HeaderStatus />
            {showNewRun && (
              <Button variant="primary" icon="plus" onClick={() => nav("/new")}>{t("header.newRun")}</Button>
            )}
            {!showNewRun && headerAction && (
              <Button variant="primary" icon={headerAction.icon ?? "plus"} onClick={headerAction.onClick}>
                {headerAction.label}
              </Button>
            )}
          </div>
        </header>
        <main className="content">
          <Outlet context={setHeaderAction} />
        </main>
      </div>
    </div>
  );
}
