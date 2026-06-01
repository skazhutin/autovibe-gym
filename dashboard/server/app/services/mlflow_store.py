"""Read runs from the project's MLflow store and parse episode artifacts into
the dashboard's Run shape + per-tab detail (notebook, trajectory, checklist,
errors, logs).

Artifact parsing is **directory-based**: every function takes the episode dir
that holds solution.ipynb / notebook_events.json / feedback_trace.json / ... .
That dir is `mlruns/<exp>/<run>/artifacts/episode` for a finished MLflow run, or
the live `data/runs/<id>/workspace` dir for a run that is still in progress — so
the same parsing powers both live and historical views.

Per-item checklist closure is reconstructed by replaying public notebook events
through the gym's own NotebookChecklist (keeps this faithful to env logic).
No caching: live runs rewrite these files after every step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings

_MODE_MAP = {
    "baseline": "single",
    "baseline_single_shot": "single",
    "gym_with_checklist": "gym",
    "iterative_no_checklist": "iterative",
    "multishot": "repeated",
    "single_shot": "single",
    "repeated_single_shot": "repeated",
    "fixed_transitions": "iterative",
}

_CHECK_LABELS: dict[str, str] = {
    "task_understanding": "Понимание задачи",
    "schema_review": "Обзор схемы данных",
    "target_distribution_review": "Распределение таргета",
    "missing_values_audit": "Пропущенные значения",
    "categorical_features_audit": "Категориальные признаки",
    "duplicates_audit": "Дубликаты",
    "suspicious_columns_audit": "Подозрительные колонки",
    "target_exclusion": "Исключение таргета из признаков",
    "baseline_candidate_created": "Baseline-кандидат создан",
    "validation_evaluated": "Оценка на валидации",
    "reproducible_solution": "Воспроизводимость (clean run)",
    "submit_ready_artifact": "Готовность к сабмиту",
}
CHECK_TOTAL = len(_CHECK_LABELS)


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def _f(metrics: dict, *keys: str) -> float | None:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def _bool_metric(metrics: dict, key: str) -> bool | None:
    if key not in metrics or metrics[key] is None:
        return None
    return bool(float(metrics[key]))


def _has_real_test_metric(metrics: dict) -> bool:
    test_metric = _f(metrics, "final_test_metric", "test_metric")
    if test_metric is None:
        return False
    valid_submit = _bool_metric(metrics, "valid_submit")
    has_test = _bool_metric(metrics, "has_test_metric")
    submit_failed = _bool_metric(metrics, "submit_failed")
    if valid_submit is not None or has_test is not None:
        return bool(valid_submit or has_test)
    return submit_failed is not True


def _score(metrics: dict) -> float | None:
    if not _has_real_test_metric(metrics):
        return None
    return _f(metrics, "final_test_metric", "test_metric")


def _derive_status(info_status: str, metrics: dict) -> str:
    if info_status in ("RUNNING", "SCHEDULED"):
        return "running"
    if _has_real_test_metric(metrics):
        return "success"
    submit_failed = _bool_metric(metrics, "submit_failed")
    if submit_failed or info_status == "FAILED":
        return "failed"
    return "null"


def _run_record(run) -> dict[str, Any]:
    params = run.data.params or {}
    metrics = run.data.metrics or {}
    info = run.info
    mode_param = params.get("experiment_type") or params.get("episode_mode") or ""
    ui_mode = _MODE_MAP.get(mode_param, "gym")
    coverage = _f(metrics, "checklist_coverage")
    max_steps = (
        params.get("max_steps")
        or params.get("max_agent_turns")
        or params.get("max_attempts")
    )
    step = _f(metrics, "steps_used", "attempts_used")
    if step is None and ui_mode == "single" and info.status not in ("RUNNING", "SCHEDULED"):
        step = 1
        max_steps = max_steps or 1
    started_ms = info.start_time or 0
    ended_ms = info.end_time or 0
    dur = _f(metrics, "elapsed_seconds")
    if dur is None and started_ms and ended_ms:
        dur = round((ended_ms - started_ms) / 1000)
    return {
        "id": info.run_id,
        "shortId": info.run_id[:8],
        "runName": (run.data.tags or {}).get("mlflow.runName", info.run_id[:8]),
        "model": params.get("model", "—"),
        "mode": ui_mode,
        "dataset": params.get("dataset", "—"),
        "status": _derive_status(info.status, metrics),
        "score": _score(metrics),
        "metric": params.get("metric_name") or params.get("metric"),
        "baseline": _f(metrics, "best_validation_metric"),
        "checklistTotal": CHECK_TOTAL,
        "checklist": round((coverage or 0) * CHECK_TOTAL),
        "checklistCoverage": coverage,
        "errors": int(_f(metrics, "error_count", "errors_count") or 0),
        "step": int(step or 0),
        "steps": int(max_steps) if max_steps else None,
        "tokIn": int(_f(metrics, "input_tokens") or 0),
        "tokOut": int(_f(metrics, "output_tokens") or 0),
        "startedMs": started_ms,
        "endedMs": ended_ms,
        "dur": int(dur) if dur is not None else None,
        "seed": params.get("seed"),
        "temp": params.get("temperature") or params.get("temp"),
        "experimentId": info.experiment_id,
        "source": "mlflow",
    }


def list_runs() -> list[dict[str, Any]]:
    client = _client()
    try:
        experiments = client.search_experiments()
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for exp in experiments:
        try:
            runs = client.search_runs([exp.experiment_id], max_results=500)
        except Exception:
            continue
        for run in runs:
            try:
                records.append(_run_record(run))
            except Exception:
                continue
    records.sort(key=lambda r: r["startedMs"], reverse=True)
    return records


def get_run(run_id: str) -> dict[str, Any] | None:
    try:
        return _run_record(_client().get_run(run_id))
    except Exception:
        return None


# --- artifact directory resolution ---------------------------------------

def mlflow_episode_dir(run_id: str) -> Path | None:
    """Resolve the local `.../artifacts/episode` dir for a finished MLflow run."""
    base = get_settings().mlruns_dir
    if not base.exists():
        return None
    for exp_dir in base.iterdir():
        cand = exp_dir / run_id / "artifacts" / "episode"
        if cand.exists():
            return cand
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _events(episode_dir: Path | None) -> list[dict]:
    if not episode_dir:
        return []
    return _read_json(episode_dir / "notebook_events.json") or []


def _feedback_trace(episode_dir: Path | None) -> list[dict]:
    if not episode_dir:
        return []
    return _read_json(episode_dir / "feedback_trace.json") or []


def _notebook_path(episode_dir: Path | None) -> Path | None:
    if not episode_dir:
        return None
    for name in ("solution.ipynb", "final_notebook.ipynb"):
        p = episode_dir / name
        if p.exists():
            return p
    return None


# --- parsing ---------------------------------------------------------------

def _map_output(out: dict) -> dict | None:
    otype = out.get("output_type")
    if otype == "stream":
        return {"type": "stdout", "text": out.get("text", "")}
    if otype == "error":
        tb = "\n".join(out.get("traceback", []) or [])
        return {"type": "error", "ename": out.get("ename", "Error"), "text": tb or out.get("evalue", "")}
    if otype in ("execute_result", "display_data"):
        data = out.get("data", {}) or {}
        if "text/html" in data:
            html = data["text/html"]
            return {"type": "table", "html": "".join(html) if isinstance(html, list) else html}
        if "text/plain" in data:
            txt = data["text/plain"]
            return {"type": "stdout", "text": "".join(txt) if isinstance(txt, list) else txt}
    return None


def notebook(episode_dir: Path | None) -> dict[str, Any]:
    nb = _read_json(_notebook_path(episode_dir)) if _notebook_path(episode_dir) else None
    if not nb:
        return {"cells": []}
    cells: list[dict] = []
    n = 0
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if ctype == "markdown":
            cells.append({"type": "markdown", "text": src})
            continue
        if ctype != "code":
            continue
        n += 1
        outs = []
        for out in cell.get("outputs", []) or []:
            mapped = _map_output(out)
            if mapped:
                outs.append(mapped)
        cells.append({"type": "code", "n": cell.get("execution_count") or n, "code": src, "outputs": outs})
    return {"cells": cells}


def _channel(item: dict) -> str:
    ch = item.get("channel", "runtime")
    if ch == "checklist" and item.get("visible_to_agent", True) and item.get("severity") == "info":
        return "checklist-hint"
    return ch


def trajectory(episode_dir: Path | None) -> list[dict[str, Any]]:
    steps: list[dict] = []
    for entry in _feedback_trace(episode_dir):
        action = entry.get("action", "code")
        title = {
            "add_cell": "Добавлена ячейка", "edit_cell": "Изменена ячейка",
            "update_cell": "Изменена ячейка", "delete_cell": "Удалена ячейка",
            "run_cell": "Выполнена ячейка", "restart_and_run_all": "Чистый перезапуск",
            "validate": "Валидация кандидата", "submit": "Финальный сабмит",
        }.get(action, action)
        feedback = [{"ch": _channel(it), "text": it.get("message", "")} for it in (entry.get("feedback_items") or [])]
        if entry.get("stderr"):
            feedback.append({"ch": "runtime", "text": entry["stderr"].strip()[:2000]})
        ui_action = "submit" if action == "submit" else ("validate" if action == "validate" else "code")
        steps.append({
            "step": entry.get("step"), "action": ui_action, "title": title,
            "code": entry.get("code") or "", "budgetRemaining": entry.get("budget_remaining"),
            "feedback": feedback,
        })
    return steps


def errors(episode_dir: Path | None) -> list[dict[str, Any]]:
    out: list[dict] = []
    for ev in _events(episode_dir):
        res = ev.get("execution_result") or {}
        if res.get("error_name") or (res.get("success") is False):
            tb = res.get("traceback") or []
            out.append({
                "step": ev.get("step"), "cellId": ev.get("cell_id"),
                "type": res.get("error_name") or "Error", "value": res.get("error_value") or "",
                "traceback": "\n".join(tb) if isinstance(tb, list) else str(tb),
                "stderr": res.get("stderr", ""),
            })
    return out


def logs(episode_dir: Path | None) -> list[dict[str, Any]]:
    msgs: list[dict] = []
    for entry in _feedback_trace(episode_dir):
        code = entry.get("code") or ""
        if code:
            msgs.append({"role": "assistant", "text": code, "action": entry.get("action")})
        parts = []
        if entry.get("stdout"):
            parts.append(entry["stdout"].rstrip())
        if entry.get("stderr"):
            parts.append("stderr:\n" + entry["stderr"].rstrip())
        for it in entry.get("feedback_items") or []:
            parts.append(f"[{it.get('channel')}] {it.get('message')}")
        if parts:
            msgs.append({"role": "tool", "text": "\n\n".join(parts)})
    return msgs


def _replay_checklist(episode_dir: Path | None, target_col: str = "") -> tuple[dict[str, int | None], float | None]:
    """Return ({key: closed_step|None}, coverage) by replaying events through
    the gym's NotebookChecklist."""
    closed_step: dict[str, int | None] = {k: None for k in _CHECK_LABELS}
    coverage: float | None = None
    try:
        from gym.feedback import NotebookChecklist

        events = _events(episode_dir)
        if not events:
            return closed_step, None
        cl = NotebookChecklist(target_col=target_col or "target")
        prev: set[str] = set()
        for ev in events:
            res = ev.get("execution_result") or {}
            cl.record_execution(
                source=ev.get("source_after") or "", stdout=res.get("stdout", ""),
                cell_id=ev.get("cell_id"), step=ev.get("step", 0),
                execution_success=bool(res.get("success", True)),
            )
            for key in set(cl.covered) - prev:
                if key in closed_step and closed_step[key] is None:
                    closed_step[key] = ev.get("step")
            prev = set(cl.covered)
        coverage = cl.coverage()
    except Exception:
        pass
    return closed_step, coverage


def checklist(episode_dir: Path | None, target_col: str = "", fallback_coverage: float | None = None) -> dict[str, Any]:
    closed_step, coverage = _replay_checklist(episode_dir, target_col)
    replay_closed = sum(1 for step in closed_step.values() if step is not None)
    if fallback_coverage is not None and (coverage is None or replay_closed == 0):
        coverage = fallback_coverage
    items_meta = [{"id": k, "label": _CHECK_LABELS[k], "desc": ""} for k in _CHECK_LABELS]
    try:
        from gym.feedback import GENERIC_CHECKLIST_HINTS

        for it in items_meta:
            it["desc"] = GENERIC_CHECKLIST_HINTS.get(it["id"], "")
    except Exception:
        pass
    items = [{**m, "closed": closed_step[m["id"]] is not None, "closedStep": closed_step[m["id"]]} for m in items_meta]
    # Use the env's authoritative coverage metric for the count so it matches the
    # run header; fall back to the per-item replay count only when unavailable.
    closed = round(coverage * len(items)) if coverage is not None else replay_closed
    return {"items": items, "coverage": coverage, "closed": closed, "total": len(items)}


def episode_progress(episode_dir: Path | None, target_col: str = "") -> dict[str, Any]:
    """Live progress derived from in-flight artifacts: current step, error count,
    checklist closed/coverage, and notebook cell count. Used to keep a running
    run's header/chips/ring moving before MLflow metrics exist."""
    events = _events(episode_dir)
    step = max((e.get("step", 0) for e in events), default=0)
    err = sum(1 for e in events if (e.get("execution_result") or {}).get("error_name") or (e.get("execution_result") or {}).get("success") is False)
    closed_step, coverage = _replay_checklist(episode_dir, target_col)
    closed = sum(1 for v in closed_step.values() if v is not None)
    nb = notebook(episode_dir)
    return {
        "step": step,
        "errors": err,
        "checklist": closed,
        "checklistTotal": CHECK_TOTAL,
        "checklistCoverage": coverage,
        "notebookCells": len([c for c in nb["cells"] if c["type"] == "code"]),
    }
