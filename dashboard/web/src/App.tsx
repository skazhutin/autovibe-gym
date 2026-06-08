import { useEffect } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { applyAppearance, loadAppearance } from "./lib/theme";
import { I18nProvider } from "./lib/i18n";
import { useAsync } from "./lib/hooks";
import { api } from "./lib/api";
import "./styles/app.css";

import Dashboard from "./pages/Dashboard";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import RunsArchive from "./pages/RunsArchive";
import RunDetail from "./pages/RunDetail";
import Compare from "./pages/Compare";
import Tasks from "./pages/Tasks";
import TaskDetail from "./pages/TaskDetail";
import Models from "./pages/Models";
import ModelsArchive from "./pages/ModelsArchive";
import TasksArchive from "./pages/TasksArchive";
import SettingsPage from "./pages/Settings";

export default function App() {
  const { data } = useAsync(() => api.getSettings(), []);

  useEffect(() => {
    applyAppearance(loadAppearance());
  }, []);

  useEffect(() => {
    if (!data) return;
    applyAppearance({
      theme: data.theme === "dark" ? "dark" : "light",
      accent: data.accent,
      radius: data.radius,
      animations: data.animations ?? "on",
      overlayOpacity: (data as unknown as Record<string, unknown>).overlay_opacity as number ?? 78,
    });
  }, [data]);

  return (
    <I18nProvider key={data?.language ?? "ru"} initialLanguage={data?.language ?? "ru"}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="new" element={<NewRun />} />
          <Route path="runs" element={<Runs />} />
          <Route path="runs/archive" element={<RunsArchive />} />
          <Route path="runs/:id" element={<RunDetail />} />
          <Route path="compare" element={<Compare />} />
          <Route path="tasks" element={<Tasks />} />
          <Route path="tasks/:id" element={<TaskDetail />} />
          <Route path="problems" element={<Tasks />} />
          <Route path="problems/:id" element={<TaskDetail />} />
          <Route path="models" element={<Models />} />
          <Route path="models/archive" element={<ModelsArchive />} />
          <Route path="problems/archive" element={<TasksArchive />} />
          <Route path="tasks/archive" element={<TasksArchive />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </I18nProvider>
  );
}
