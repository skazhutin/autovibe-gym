import { useEffect } from "react";
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
  { to: "/datasets", label: "Датасеты", icon: "database" },
  { to: "/models", label: "Модели", icon: "cpu" },
  { to: "/settings", label: "Настройки", icon: "settings" },
];

// Route -> header title/subtitle.
const META: { match: (p: string) => boolean; title: string; sub: string }[] = [
  { match: (p) => p === "/", title: "Дашборд", sub: "Обзор активности спортзала" },
  { match: (p) => p.startsWith("/new"), title: "Новый прогон", sub: "Настройте и запустите LLM-агента" },
  { match: (p) => p.startsWith("/runs/"), title: "Прогон", sub: "Решение агента, метрики и диагностика" },
  { match: (p) => p.startsWith("/runs"), title: "Прогоны", sub: "История всех запусков" },
  { match: (p) => p.startsWith("/compare"), title: "Сравнение", sub: "Сопоставление прогонов по метрикам" },
  { match: (p) => p.startsWith("/datasets/"), title: "Датасет", sub: "Данные, статистика и метаданные" },
  { match: (p) => p.startsWith("/datasets"), title: "Датасеты", sub: "Управление наборами данных" },
  { match: (p) => p.startsWith("/models"), title: "Модели", sub: "Реестр LLM-эндпоинтов" },
  { match: (p) => p.startsWith("/settings"), title: "Настройки", sub: "Подключения и внешний вид" },
];

function HeaderStatus() {
  const { data } = useAsync(() => api.health(), [], 15000);
  const online = data?.status === "online";
  return (
    <span className="status-pill">
      <Dot tone={online ? "green" : "gray"} />
      {online ? "Бэкенд онлайн" : "Бэкенд офлайн"}
    </span>
  );
}

export default function Layout() {
  const loc = useLocation();
  const nav = useNavigate();
  const meta = META.find((m) => m.match(loc.pathname)) ?? META[0];

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [loc.pathname]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="mark">A</span>
          AutoVibe Gym
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
              <Icon name={n.icon} size={19} strokeWidth={1.9} />
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="avatar">AV</span>
          <div style={{ lineHeight: 1.25 }}>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Команда</div>
            <div className="faint" style={{ fontSize: 11.5 }}>локальный режим</div>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="header">
          <div>
            <h1>{meta.title}</h1>
            <div className="sub">{meta.sub}</div>
          </div>
          <div className="header-right">
            <HeaderStatus />
            {loc.pathname !== "/new" && (
              <Button variant="primary" icon="plus" onClick={() => nav("/new")}>
                Новый прогон
              </Button>
            )}
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
