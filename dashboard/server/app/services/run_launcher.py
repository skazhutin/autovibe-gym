"""Launch / track / stop gym experiment subprocesses.

UI mode -> runner:
  single    -> experiments.run_baseline
  repeated  -> experiments.run_multishot   (--shots)
  iterative -> experiments.run_gym --episode-mode iterative_no_checklist  (--max-steps)
  gym       -> experiments.run_gym --episode-mode gym_with_checklist       (--max-steps)
  fixed     -> experiments.run_fixed       (--max-steps)
  batch     -> experiments.run --modes ... (selected product-mode batch)

Runs are spawned with the project venv, cwd=repo root, and MLflow tracking
pointed at the dashboard's store so finished runs show up in the history list.
Each launch gets a stable local id and a unique run-name (``dash_<id>``); once
the process exits we resolve the MLflow run it produced and serve its artifacts.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import get_settings
from . import model_store, remote_exec
from experiments.modes import ALL_PRODUCT_MODES, BATCH_REQUESTED_MODE, MODE_BY_DASHBOARD_MODE

# local_id -> {"proc": Popen, "meta": dict}
_ACTIVE: dict[str, dict[str, Any]] = {}

_MODE_TO_RUNNER = {
    "single": ("experiments.run_baseline", None),
    "repeated": ("experiments.run_multishot", None),
    "iterative": ("experiments.run_gym", "iterative_no_checklist"),
    "gym": ("experiments.run_gym", "gym_with_checklist"),
    "fixed": ("experiments.run_fixed", None),
    BATCH_REQUESTED_MODE: ("experiments.run", None),
}


def _selected_modes(cfg: dict[str, Any]) -> list[str]:
    raw_modes = cfg.get("modes") or [cfg.get("mode")]
    modes: list[str] = []
    for raw_mode in raw_modes:
        if not raw_mode or raw_mode == BATCH_REQUESTED_MODE:
            continue
        mode = str(raw_mode)
        if mode not in MODE_BY_DASHBOARD_MODE:
            raise ValueError(f"Unsupported mode in batch: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("At least one run mode must be selected")
    if len(modes) > 5:
        raise ValueError("Select at most 5 run modes")
    return modes


def _normalize_cfg_modes(cfg: dict[str, Any]) -> list[str]:
    modes = _selected_modes(cfg)
    cfg["modes"] = modes
    if len(modes) > 1:
        cfg["mode"] = BATCH_REQUESTED_MODE
    elif cfg.get("mode") == BATCH_REQUESTED_MODE:
        cfg["mode"] = modes[0]
    return modes


def _selected_product_keys(cfg: dict[str, Any]) -> list[str]:
    return [MODE_BY_DASHBOARD_MODE[mode].key for mode in _selected_modes(cfg)]


def _planned_steps(cfg: dict[str, Any]) -> int | None:
    if cfg["mode"] == BATCH_REQUESTED_MODE:
        return len(_selected_modes(cfg))
    if cfg["mode"] == "all":
        return len(ALL_PRODUCT_MODES)
    if cfg["mode"] == "single":
        return 1
    if cfg["mode"] == "repeated":
        return cfg.get("shots")
    return cfg.get("maxSteps")


def _python_available(python_bin: str) -> bool:
    return Path(python_bin).exists() or shutil.which(python_bin) is not None


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


def _runner_args(cfg: dict[str, Any]) -> list[str]:
    """Runner args shared by local and remote launches (without python binary,
    --dataset-dir and --workspace-dir, which are added per environment)."""
    mode = cfg["mode"]
    module, episode = _MODE_TO_RUNNER[mode]
    if mode == BATCH_REQUESTED_MODE:
        args = [
            "-m", module,
            "--modes", *_selected_product_keys(cfg),
            "--budget-mode", cfg.get("budgetMode", "local"),
        ]
    else:
        args = ["-m", module, "--mode", cfg.get("budgetMode", "local")]
    if cfg.get("model"):
        args += ["--model", cfg["model"]]
    if cfg.get("maxTokens"):
        args += ["--max-tokens", str(cfg["maxTokens"])]
    if cfg.get("seed") is not None:
        args += ["--seed", str(cfg["seed"])]
    if episode:
        args += ["--episode-mode", episode]
        if cfg.get("maxSteps"):
            args += ["--max-steps", str(cfg["maxSteps"])]
        # Persistent agent scratchpad — only the notebook (gym/iterative) modes
        # support it (multi-turn, so notes can be re-shown to the agent).
        if cfg.get("enableThoughts"):
            args += ["--enable-thoughts"]
    elif mode == "fixed" and cfg.get("maxSteps"):
        args += ["--max-steps", str(cfg["maxSteps"])]
    elif mode == BATCH_REQUESTED_MODE and cfg.get("maxSteps"):
        args += ["--max-steps", str(cfg["maxSteps"])]
    if mode == "repeated" and cfg.get("shots"):
        args += ["--shots", str(cfg["shots"])]
    if mode == BATCH_REQUESTED_MODE and cfg.get("shots"):
        args += ["--shots", str(cfg["shots"])]
    # The legacy single-shot/repeated runners default to a 60s execution timeout,
    # too short for local thread-capped model training (→ "no candidate" on kill).
    # Give them a generous timeout (env AUTOVIBE_DASHBOARD_TIMEOUT).
    if mode in ("single", "repeated", BATCH_REQUESTED_MODE):
        args += ["--sandbox-timeout", os.getenv("AUTOVIBE_DASHBOARD_TIMEOUT", "300")]
    args += ["--experiment-name", cfg.get("experimentName", "autovibe-dashboard")]
    args += ["--run-name", cfg["runName"]]
    return args


def _build_command(cfg: dict[str, Any]) -> list[str]:
    s = get_settings()
    cmd = [s.python_bin, *_runner_args(cfg), "--dataset-dir", cfg["datasetDir"]]
    # Runs write episode artifacts here so the dashboard can show the
    # notebook/trajectory/checklist. Notebook modes flush per step; single-shot
    # and repeated emit a synthesized episode at the end.
    if cfg.get("workspaceDir"):
        cmd += ["--workspace-dir", cfg["workspaceDir"]]
    return cmd


def _llm_env(cfg: dict[str, Any]) -> dict[str, str]:
    """LLM connection + thread caps for the selected model (used by both envs)."""
    out: dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": _THREADS, "OPENBLAS_NUM_THREADS": _THREADS,
        "MKL_NUM_THREADS": _THREADS, "NUMEXPR_NUM_THREADS": _THREADS,
        "VECLIB_MAXIMUM_THREADS": _THREADS, "AUTOVIBE_SANDBOX_THREADS": _THREADS,
        "JOBLIB_MULTIPROCESSING": "0", "LOKY_MAX_CPU_COUNT": _THREADS,
    }
    model = model_store.get_model(cfg["modelId"]) if cfg.get("modelId") else None
    if model:
        if model.get("name"):
            out["LLM_MODEL"] = model["name"]
        if model.get("baseUrl"):
            out["LLM_BASE_URL"] = model["baseUrl"]
        key = model.get("apiKey") or os.getenv(model.get("apiKeyEnv") or "LLM_API_KEY", "")
        if key:
            out["LLM_API_KEY"] = key
    if cfg.get("temp") is not None:
        out["LLM_TEMPERATURE"] = str(cfg["temp"])
    return out


# Keep local model training (xgboost/sklearn/BLAS) from pegging every core and
# spinning the laptop fans. Overridable via AUTOVIBE_DASHBOARD_THREADS.
_THREADS = os.getenv("AUTOVIBE_DASHBOARD_THREADS", "2")


def _build_env(cfg: dict[str, Any]) -> dict[str, str]:
    s = get_settings()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(s.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["MLFLOW_TRACKING_URI"] = s.mlflow_tracking_uri
    # Local execution has no Docker: force the in-process executor for the legacy
    # single-shot/repeated runners (the .env default is docker) and a local kernel
    # for notebook modes. Overridable via AUTOVIBE_DASHBOARD_EXECUTOR.
    env["AUTOVIBE_EXECUTOR_BACKEND"] = os.getenv("AUTOVIBE_DASHBOARD_EXECUTOR", "subprocess")
    env["AUTOVIBE_KERNEL_BACKEND"] = "local"
    # LLM connection + thread caps (load_dotenv won't override these explicit vars).
    env.update(_llm_env(cfg))
    return env


def launch(cfg: dict[str, Any]) -> dict[str, Any]:
    s = get_settings()
    selected_modes = _normalize_cfg_modes(cfg)
    if cfg["mode"] not in _MODE_TO_RUNNER:
        raise ValueError(f"Unsupported mode: {cfg['mode']}")
    if not _python_available(s.python_bin):
        raise ValueError(f"Python interpreter not found: {s.python_bin}")
    # Per-run execution location: "server" (SSH) | "local" | None (use default).
    exe = cfg.get("execution")
    if exe == "server" and not remote_exec.is_configured():
        raise ValueError("Серверный режим не настроен: Настройки → «Выполнение на сервере (SSH)».")
    want_remote = (exe == "server") or (not exe and remote_exec.is_enabled())
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
        "requestedMode": BATCH_REQUESTED_MODE if cfg["mode"] == BATCH_REQUESTED_MODE else cfg["mode"],
        "batchId": None,
        "productMode": None,
        "modeLabel": ", ".join(selected_modes) if cfg["mode"] == BATCH_REQUESTED_MODE else cfg["mode"],
        "modeOrder": None,
        "selectedModes": selected_modes,
        "dataset": cfg.get("dataset") or Path(cfg["datasetDir"]).name,
        "datasetDir": cfg["datasetDir"],
        "status": "running",
        "score": None,
        "checklist": 0,
        "checklistTotal": 12,
        "errors": 0,
        "step": 0,
        "steps": _planned_steps(cfg),
        "tokIn": 0,
        "tokOut": 0,
        "startedMs": int(time.time() * 1000),
        "endedMs": 0,
        "dur": None,
        "seed": cfg.get("seed"),
        "temp": cfg.get("temp"),
        "budgetMode": cfg.get("budgetMode", "local"),
        "workspaceDir": str(workspace),
        "thoughtsEnabled": bool(cfg.get("enableThoughts")),
        "source": "live",
        "mlflowId": None,
    }

    # Remote mode: run the gym on the server over SSH; the Mac only syncs results.
    if want_remote:
        return _launch_remote(cfg, meta, run_dir)

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
        if meta.get("mode") in (BATCH_REQUESTED_MODE, "all"):
            meta["step"] = meta.get("steps") or len(ALL_PRODUCT_MODES)
            meta["status"] = "success" if rc == 0 else "failed"
        else:
            meta["status"] = "null" if rc == 0 else "failed"
        if rc != 0:
            meta["failReason"] = f"Процесс завершился с кодом {rc}. См. вкладку «Логи»."
    _write_meta(meta)
    del _ACTIVE[local_id]
    return meta


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- remote (SSH) execution -----------------------------------------------

_REMOTE_SYNC_AT: dict[str, float] = {}


def _launch_remote(cfg: dict[str, Any], meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    rc = remote_exec.config()
    runs_dir = rc["runs_dir"].rstrip("/")
    ws_remote = f"{runs_dir}/{meta['id']}/workspace"
    log_remote = f"{runs_dir}/{meta['id']}/process.log"
    dataset_rel = f"datasets/{Path(cfg['datasetDir']).name}"
    args = _runner_args(cfg)
    env = _llm_env(cfg)
    meta.update({
        "remote": True, "remoteSsh": rc["ssh"], "wsRemote": ws_remote, "logRemote": log_remote,
        "command": f"ssh {rc['ssh']} :: {rc['python']} {' '.join(args)} --dataset-dir {dataset_rel}",
    })
    res = remote_exec.launch(
        rc, run_id=meta["id"], runner_args=args, env=env,
        dataset_rel=dataset_rel, workspace_remote=ws_remote, log_remote=log_remote,
    )
    if "error" in res:
        meta["status"] = "failed"
        meta["failReason"] = f"Не удалось запустить на сервере: {res['error']}"
        meta["endedMs"] = _now_ms()
        _write_meta(meta)
        return meta
    meta["remotePid"] = res["pid"]
    _write_meta(meta)
    return meta


def _apply_summary(meta: dict[str, Any], summary: dict[str, Any] | None) -> None:
    if not summary:
        meta["status"] = "failed"
        meta.setdefault("failReason", "Прогон на сервере завершился без итоговой сводки. См. «Логи».")
        return
    score = summary.get("final_test_metric", summary.get("test_metric"))
    cov = summary.get("checklist_coverage")
    meta["score"] = score
    if cov is not None:
        meta["checklistCoverage"] = cov
        meta["checklist"] = round(cov * meta.get("checklistTotal", 12))
    if summary.get("steps_used") is not None:
        meta["step"] = int(summary["steps_used"])
    elif summary.get("attempts_used") is not None:
        meta["step"] = int(summary["attempts_used"])
    if summary.get("error_count") is not None:
        meta["errors"] = int(summary["error_count"])
    meta["tokIn"] = int(summary.get("input_tokens", meta.get("tokIn", 0)) or 0)
    meta["tokOut"] = int(summary.get("output_tokens", meta.get("tokOut", 0)) or 0)
    meta["metric"] = meta.get("metric") or summary.get("metric_name")
    if score is not None and summary.get("valid_submit"):
        meta["status"] = "success"
    else:
        meta["status"] = "null"
        meta["failReason"] = "Агент не дошёл до валидного сабмита (см. «Траектория»/«Логи»)."


def _refresh_remote(meta: dict[str, Any]) -> dict[str, Any]:
    """Sync the remote workspace/log to the local mirror and reconcile status."""
    if meta.get("status") != "running":
        return meta
    rc = remote_exec.config()
    if not rc["ssh"]:
        return meta
    run_id = meta["id"]
    meta["dur"] = round((_now_ms() - meta.get("startedMs", _now_ms())) / 1000)
    # Throttle remote sync to at most ~ every 2.5s across concurrent polls.
    now = time.time()
    if now - _REMOTE_SYNC_AT.get(run_id, 0) >= 2.5:
        _REMOTE_SYNC_AT[run_id] = now
        try:
            remote_exec.sync(rc, workspace_remote=meta["wsRemote"], log_remote=meta["logRemote"],
                             local_dir=get_settings().runs_dir / run_id)
        except Exception:
            pass
        if not remote_exec.alive(rc, meta.get("remotePid", "")):
            summary = remote_exec.parse_summary(read_log(run_id))
            _apply_summary(meta, summary)
            meta["endedMs"] = _now_ms()
            meta["dur"] = round((meta["endedMs"] - meta["startedMs"]) / 1000)
        _write_meta(meta)
    return meta


def _reconcile_orphan(meta: dict[str, Any]) -> dict[str, Any]:
    """A meta still marked 'running' but no longer in this process's registry —
    typically the server was reloaded/restarted mid-run. Reconcile it against
    the MLflow run it produced; if that finished, adopt the final result, else
    keep it running but advance the live duration."""
    if meta.get("status") != "running":
        return meta
    mlflow_id = meta.get("mlflowId") or _resolve_mlflow_id(meta.get("runName", ""))
    if mlflow_id:
        meta["mlflowId"] = mlflow_id
        from . import mlflow_store

        rec = mlflow_store.get_run(mlflow_id)
        if rec and rec.get("status") != "running":
            for key in ("status", "score", "metric", "checklist", "checklistCoverage",
                        "checklistTotal", "errors", "step", "steps", "tokIn", "tokOut", "dur"):
                if rec.get(key) is not None:
                    meta[key] = rec[key]
            meta["endedMs"] = meta.get("endedMs") or _now_ms()
            _write_meta(meta)
            return meta
    # still running (or MLflow run not yet created): keep the clock moving
    meta["dur"] = round((_now_ms() - meta.get("startedMs", _now_ms())) / 1000)
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
    if s.runs_dir.exists():
        for d in s.runs_dir.iterdir():
            if d.name in seen:
                continue
            m = _read_meta(d.name)
            if m:
                if m.get("mode") in (BATCH_REQUESTED_MODE, "all") and m.get("status") != "running":
                    continue
                out.append(_refresh_remote(m) if m.get("remote") else _reconcile_orphan(m))
    out.sort(key=lambda r: r.get("startedMs", 0), reverse=True)
    return out


def get_live(local_id: str) -> dict[str, Any] | None:
    if local_id in _ACTIVE:
        return _refresh(local_id)
    meta = _read_meta(local_id)
    if not meta:
        return None
    if meta.get("remote"):
        return _refresh_remote(meta)
    return _reconcile_orphan(meta)


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
        # Remote run (or orphaned): kill over SSH if applicable, mark stopped.
        meta = _read_meta(local_id)
        if not meta or meta.get("status") != "running":
            return False
        if meta.get("remote"):
            try:
                remote_exec.kill(remote_exec.config(), meta.get("remotePid", ""))
            except Exception:
                pass
            meta["status"] = "failed"
            meta["failReason"] = "Прогон остановлен пользователем."
            meta["endedMs"] = int(time.time() * 1000)
            _write_meta(meta)
            return True
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
