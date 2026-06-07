import { useEffect, useState, useCallback } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Icon } from "./Icon";
import { Button, Dot } from "./ui";
import { useAsync } from "../lib/hooks";
import { api } from "../lib/api";

const NAV = [
  { to: "/", label: "Дашборд", icon: "dashboard", end: true },
  { to: "/new", label: "Новый прогон", icon: "play" },
  { to: "/runs", label: "Прогоны", icon: "runs" },
  { to: "/compare", label: "Сравнение", icon: "compare" },
  { to: "/problems", label: "Проблемы", icon: "database" },
  { to: "/models", label: "Модели", icon: "cpu" },
  { to: "/settings", label: "Настройки", icon: "settings" },
];

// Route -> header title/subtitle.
const META: { match: (p: string) => boolean; title: string; sub: string }[] = [
  { match: (p) => p === "/", title: "Дашборд", sub: "Обзор активности спортзала" },
  { match: (p) => p.startsWith("/new"), title: "Новый прогон", sub: "Настройте и запустите LLM-агента" },
  { match: (p) => p === "/runs/archive", title: "Архив прогонов", sub: "Архивированные прогоны" },
  { match: (p) => p.startsWith("/runs/"), title: "Прогон", sub: "Решение агента, метрики и диагностика" },
  { match: (p) => p.startsWith("/runs"), title: "Прогоны", sub: "История всех запусков" },
  { match: (p) => p.startsWith("/compare"), title: "Сравнение", sub: "Сопоставление прогонов по метрикам" },
  { match: (p) => p.startsWith("/problems/"), title: "Проблема", sub: "Датасет, статистика и метаданные" },
  { match: (p) => p.startsWith("/problems"), title: "Проблемы", sub: "Датасеты для экспериментов LLM-агента" },
  { match: (p) => p.startsWith("/models"), title: "Модели", sub: "Реестр LLM-эндпоинтов" },
  { match: (p) => p.startsWith("/settings"), title: "Настройки", sub: "Подключения и внешний вид" },
];

function HeaderStatus() {
  // Reflects whether the LLM server (gemma/deepseek host) is reachable.
  const { data, error } = useAsync(() => api.serverHealth(), [], 15000);
  const online = !error && data?.online;
  const title = error
    ? "Дашборд-бэкенд недоступен"
    : !data?.configured
    ? "LLM-сервер не настроен (добавьте модель)"
    : data.servers.map((s) => `${s.baseUrl} — ${s.online ? "онлайн" : s.error ?? s.status ?? "офлайн"}`).join("\n");
  return (
    <span className="status-pill" title={title}>
      <Dot tone={online ? "green" : "red"} pulse={false} />
      {online ? "Сервер онлайн" : "Сервер офлайн"}
    </span>
  );
}

export interface HeaderAction { label: string; icon?: string; onClick: () => void }
export type SetHeaderAction = (a: HeaderAction | null) => void;

export default function Layout() {
  const loc = useLocation();
  const nav = useNavigate();
  const meta = META.find((m) => m.match(loc.pathname)) ?? META[0];
  const [collapsed, setCollapsed] = useState(false);
  const [headerAction, setHeaderActionRaw] = useState<HeaderAction | null>(null);
  const setHeaderAction: SetHeaderAction = useCallback((a) => setHeaderActionRaw(a), []);

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
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`} title={collapsed ? n.label : undefined}>
              <Icon name={n.icon} size={19} strokeWidth={1.9} />
              <span className="nav-item-label">{n.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
      <button className={`sidebar-toggle${collapsed ? " collapsed" : ""}`} onClick={() => setCollapsed((v) => !v)} title={collapsed ? "Развернуть" : "Свернуть"}>
        <Icon name={collapsed ? "chevronRight" : "chevronLeft"} size={14} strokeWidth={2} />
      </button>

      <div className="main">
        <header className="header">
          <div>
            <h1>{meta.title}</h1>
            <div className="sub">{meta.sub}</div>
          </div>
          <div className="header-right">
            <HeaderStatus />
            {showNewRun && (
              <Button variant="primary" icon="plus" onClick={() => nav("/new")}>Новый прогон</Button>
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
