import { useEffect } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { applyAppearance, loadAppearance } from "./lib/theme";
import "./styles/app.css";

import Dashboard from "./pages/Dashboard";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import RunsArchive from "./pages/RunsArchive";
import RunDetail from "./pages/RunDetail";
import Compare from "./pages/Compare";
import Datasets from "./pages/Datasets";
import DatasetDetail from "./pages/DatasetDetail";
import Models from "./pages/Models";
import ModelsArchive from "./pages/ModelsArchive";
import DatasetsArchive from "./pages/DatasetsArchive";
import SettingsPage from "./pages/Settings";

export default function App() {
  useEffect(() => {
    applyAppearance(loadAppearance());
  }, []);

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="new" element={<NewRun />} />
        <Route path="runs" element={<Runs />} />
        <Route path="runs/archive" element={<RunsArchive />} />
        <Route path="runs/:id" element={<RunDetail />} />
        <Route path="compare" element={<Compare />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="datasets/:id" element={<DatasetDetail />} />
        <Route path="problems" element={<Datasets />} />
        <Route path="problems/:id" element={<DatasetDetail />} />
        <Route path="models" element={<Models />} />
        <Route path="models/archive" element={<ModelsArchive />} />
        <Route path="problems/archive" element={<DatasetsArchive />} />
        <Route path="datasets/archive" element={<DatasetsArchive />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
