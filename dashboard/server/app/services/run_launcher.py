"""Launch / track / stop gym experiment subprocesses.

UI mode -> runner:
  single    -> experiments.run_baseline
  repeated  -> experiments.run_multishot   (--shots)
  iterative -> experiments.run_gym --episode-mode iterative_no_checklist  (--max-steps)
  gym       -> experiments.run_gym --episode-mode gym_with_checklist       (--max-steps)

Runs are spawned with the project venv, cwd=repo root, and MLflow tracking
pointed at the dashboard's store so finished runs show up in the history list.
Each launch gets a stable local id and a unique run-name (``dash_<id>``); once
the process exits we resolve the MLflow run it produced and serve its artifacts.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import get_settings
from . import model_store

# local_id -> {"proc": Popen, "meta": dict}
_ACTIVE: dict[str, dict[str, Any]] = {}

_MODE_TO_RUNNER = {
    "single": ("experiments.run_baseline", None),
    "repeated": ("experiments.run_multishot", None),
    "iterative": ("experiments.run_gym", "iterative_no_checklist"),
    "gym": ("experiments.run_gym", "gym_with_checklist"),
}


def _run_dir(local_id: str) -> Path:
    s = get_settings()
    d = s.runs_dir / local_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_meta(meta: dict[str, Any]) -> None:
    d = _run_dir(meta["id"])
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")


def _read_meta(local_id: str) -> dict[str, Any] | None:
    p = get_settings().runs_dir / local_id / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _build_command(cfg: dict[str, Any]) -> list[str]:
    s = get_settings()
    mode = cfg["mode"]
    module, episode = _MODE_TO_RUNNER[mode]
    cmd = [s.python_bin, "-m", module, "--dataset-dir", cfg["datasetDir"]]
    cmd += ["--mode", cfg.get("budgetMode", "local")]
    if cfg.get("model"):
        cmd += ["--model", cfg["model"]]
    if cfg.get("maxTokens"):
        cmd += ["--max-tokens", str(cfg["maxTokens"])]
    if cfg.get("seed") is not None:
        cmd += ["--seed", str(cfg["seed"])]
    if episode:
        cmd += ["--episode-mode", episode]
        if cfg.get("maxSteps"):
            cmd += ["--max-steps", str(cfg["maxSteps"])]
        # Notebook modes flush artifacts after every step into this dir, so the
        # dashboard can read the in-flight notebook/trajectory/checklist live.
        if cfg.get("workspaceDir"):
            cmd += ["--workspace-dir", cfg["workspaceDir"]]
    if mode == "repeated" and cfg.get("shots"):
        cmd += ["--shots", str(cfg["shots"])]
    cmd += ["--experiment-name", cfg.get("experimentName", "autovibe-dashboard")]
    cmd += ["--run-name", cfg["runName"]]
    return cmd


def _build_env(cfg: dict[str, Any]) -> dict[str, str]:
    s = get_settings()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(s.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["MLFLOW_TRACKING_URI"] = s.mlflow_tracking_uri
    env["PYTHONUNBUFFERED"] = "1"
    # Override LLM connection from the selected model (load_dotenv won't override
    # vars we set explicitly here).
    model = model_store.get_model(cfg["modelId"]) if cfg.get("modelId") else None
    if model:
        if model.get("name"):
            env["LLM_MODEL"] = model["name"]
        if model.get("baseUrl"):
            env["LLM_BASE_URL"] = model["baseUrl"]
        api_key = model.get("apiKey") or os.getenv(model.get("apiKeyEnv") or "LLM_API_KEY", "")
        if api_key:
            env["LLM_API_KEY"] = api_key
    if cfg.get("temp") is not None:
        env["LLM_TEMPERATURE"] = str(cfg["temp"])
    return env


def launch(cfg: dict[str, Any]) -> dict[str, Any]:
    s = get_settings()
    if cfg["mode"] not in _MODE_TO_RUNNER:
        raise ValueError(f"Unsupported mode: {cfg['mode']}")
    local_id = "live_" + uuid.uuid4().hex[:8]
    cfg["runName"] = f"dash_{local_id}"
    run_dir = _run_dir(local_id)
    log_path = run_dir / "process.log"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    cfg["workspaceDir"] = str(workspace)

    model = model_store.get_model(cfg["modelId"]) if cfg.get("modelId") else None
    meta = {
        "id": local_id,
        "shortId": local_id,
        "runName": cfg["runName"],
        "model": (model or {}).get("name") or cfg.get("model") or "—",
        "modelId": cfg.get("modelId"),
        "mode": cfg["mode"],
        "dataset": cfg.get("dataset") or Path(cfg["datasetDir"]).name,
        "datasetDir": cfg["datasetDir"],
        "status": "running",
        "score": None,
        "checklist": 0,
        "checklistTotal": 12,
        "errors": 0,
        "step": 0,
        "steps": cfg.get("maxSteps"),
        "tokIn": 0,
        "tokOut": 0,
        "startedMs": int(time.time() * 1000),
        "endedMs": 0,
        "dur": None,
        "seed": cfg.get("seed"),
        "temp": cfg.get("temp"),
        "budgetMode": cfg.get("budgetMode", "local"),
        "workspaceDir": str(workspace),
        "source": "live",
        "mlflowId": None,
    }
    cmd = _build_command(cfg)
    meta["command"] = " ".join(cmd)
    env = _build_env(cfg)

    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(s.repo_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group so we can stop the whole tree
    )
    meta["pid"] = proc.pid
    _ACTIVE[local_id] = {"proc": proc, "meta": meta, "log": log_file}
    _write_meta(meta)
    return meta


def _resolve_mlflow_id(run_name: str) -> str | None:
    from . import mlflow_store

    for r in mlflow_store.list_runs():
        if r.get("runName") == run_name:
            return r["id"]
    return None


def _refresh(local_id: str) -> dict[str, Any] | None:
    """Update a live run's status by polling its process; link MLflow when done."""
    entry = _ACTIVE.get(local_id)
    if entry is None:
        return _read_meta(local_id)
    proc: subprocess.Popen = entry["proc"]
    meta = entry["meta"]
    rc = proc.poll()
    if rc is None:
        meta["dur"] = round((time.time() * 1000 - meta["startedMs"]) / 1000)
        return meta
    # finished
    try:
        entry["log"].close()
    except Exception:
        pass
    meta["endedMs"] = int(time.time() * 1000)
    meta["dur"] = round((meta["endedMs"] - meta["startedMs"]) / 1000)
    mlflow_id = _resolve_mlflow_id(meta["runName"])
    meta["mlflowId"] = mlflow_id
    if mlflow_id:
        from . import mlflow_store

        rec = mlflow_store.get_run(mlflow_id)
        if rec:
            for key in ("status", "score", "metric", "checklist", "checklistCoverage",
                        "checklistTotal", "errors", "step", "steps", "tokIn", "tokOut"):
                if key in rec and rec[key] is not None:
                    meta[key] = rec[key]
    if meta["status"] == "running":
        meta["status"] = "null" if rc == 0 else "failed"
        if rc != 0:
            meta["failReason"] = f"Процесс завершился с кодом {rc}. См. вкладку «Логи»."
    _write_meta(meta)
    del _ACTIVE[local_id]
    return meta


def list_live() -> list[dict[str, Any]]:
    s = get_settings()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for local_id in list(_ACTIVE.keys()):
        m = _refresh(local_id)
        if m:
            out.append(m)
            seen.add(local_id)
    # finished live runs persisted on disk (process registry lost on restart)
    if s.runs_dir.exists():
        for d in s.runs_dir.iterdir():
            if d.name in seen:
                continue
            m = _read_meta(d.name)
            if m:
                out.append(m)
    out.sort(key=lambda r: r.get("startedMs", 0), reverse=True)
    return out


def get_live(local_id: str) -> dict[str, Any] | None:
    if local_id in _ACTIVE:
        return _refresh(local_id)
    return _read_meta(local_id)


def workspace_dir(local_id: str) -> "Path | None":
    meta = get_live(local_id)
    wd = (meta or {}).get("workspaceDir")
    p = Path(wd) if wd else None
    return p if p and p.exists() else None


def read_log(local_id: str, tail: int = 400) -> str:
    p = get_settings().runs_dir / local_id / "process.log"
    if not p.exists():
        return ""
    lines = p.read_text("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:])


def stop(local_id: str) -> bool:
    entry = _ACTIVE.get(local_id)
    if entry is None:
        return False
    proc: subprocess.Popen = entry["proc"]
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    meta = entry["meta"]
    meta["status"] = "failed"
    meta["failReason"] = "Прогон остановлен пользователем."
    meta["endedMs"] = int(time.time() * 1000)
    _write_meta(meta)
    try:
        entry["log"].close()
    except Exception:
        pass
    _ACTIVE.pop(local_id, None)
    return True
