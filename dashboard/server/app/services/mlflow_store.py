"""Read completed/finished runs from the project's MLflow tracking store and
map them to the dashboard's Run shape + per-tab detail (notebook, trajectory,
checklist, errors, logs).

The gym writes episode artifacts under the MLflow run:
  episode/solution.ipynb, episode/final_notebook.ipynb,
  episode/notebook_events.json, episode/feedback_trace.json,
  episode/validation_trajectory.json, episode/episode_summary.json
We parse those into the UI contract. Per-item checklist closure is reconstructed
by replaying the public notebook events through the gym's own NotebookChecklist,
which keeps this dashboard faithful to the real environment logic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import get_settings

# UI mode <- gym param value.
_MODE_MAP = {
    "gym_with_checklist": "gym",
    "iterative_no_checklist": "iterative",
    "single_shot": "single",
    "repeated_single_shot": "repeated",
    "fixed_transitions": "iterative",
}

# Russian labels for the 12 mandatory checklist keys (generic, dataset-agnostic).
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


def _client():
    from mlflow.tracking import MlflowClient

    s = get_settings()
    return MlflowClient(tracking_uri=s.mlflow_tracking_uri)


def _f(metrics: dict, *keys: str) -> float | None:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def _derive_status(info_status: str, metrics: dict) -> str:
    if info_status in ("RUNNING", "SCHEDULED"):
        return "running"
    has_test = _f(metrics, "has_test_metric") or 0
    test_metric = _f(metrics, "final_test_metric", "test_metric")
    valid_submit = _f(metrics, "valid_submit") or 0
    submit_failed = _f(metrics, "submit_failed") or 0
    if test_metric is not None and (valid_submit or has_test):
        return "success"
    if submit_failed or info_status == "FAILED":
        return "failed"
    return "null"


def _run_record(run) -> dict[str, Any]:
    params = run.data.params or {}
    metrics = run.data.metrics or {}
    info = run.info
    mode_param = params.get("experiment_type") or params.get("episode_mode") or params.get("mode_kind") or ""
    ui_mode = _MODE_MAP.get(mode_param, _MODE_MAP.get(params.get("episode_mode", ""), "gym"))
    coverage = _f(metrics, "checklist_coverage")
    max_steps = params.get("max_steps") or params.get("max_agent_turns")
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
        "score": _f(metrics, "final_test_metric", "test_metric"),
        "metric": params.get("metric_name") or params.get("metric"),
        "baseline": _f(metrics, "best_validation_metric"),
        "checklistTotal": len(_CHECK_LABELS),
        "checklist": round((coverage or 0) * len(_CHECK_LABELS)),
        "checklistCoverage": coverage,
        "errors": int(_f(metrics, "error_count", "errors_count") or 0),
        "step": int(_f(metrics, "steps_used") or 0),
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
    client = _client()
    try:
        run = client.get_run(run_id)
    except Exception:
        return None
    return _run_record(run)


# --- artifacts -------------------------------------------------------------

def _artifact_dir(run_id: str) -> Path | None:
    """Resolve the local artifact root for a run (file-backed store)."""
    s = get_settings()
    base = s.mlruns_dir
    if not base.exists():
        return None
    for exp_dir in base.iterdir():
        cand = exp_dir / run_id / "artifacts"
        if cand.exists():
            return cand
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@lru_cache(maxsize=64)
def _events(run_id: str) -> list[dict]:
    d = _artifact_dir(run_id)
    if not d:
        return []
    return _read_json(d / "episode" / "notebook_events.json") or []


@lru_cache(maxsize=64)
def _feedback_trace(run_id: str) -> list[dict]:
    d = _artifact_dir(run_id)
    if not d:
        return []
    return _read_json(d / "episode" / "feedback_trace.json") or []


def _notebook_path(run_id: str) -> Path | None:
    d = _artifact_dir(run_id)
    if not d:
        return None
    for name in ("solution.ipynb", "final_notebook.ipynb"):
        p = d / "episode" / name
        if p.exists():
            return p
    return None


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


def notebook(run_id: str) -> dict[str, Any]:
    path = _notebook_path(run_id)
    if not path:
        return {"cells": []}
    nb = _read_json(path)
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
        cells.append(
            {
                "type": "code",
                "n": cell.get("execution_count") or n,
                "code": src,
                "outputs": outs,
            }
        )
    return {"cells": cells}


def _channel(item: dict) -> str:
    ch = item.get("channel", "runtime")
    # A visible checklist nudge is a "hint"; structural checklist notes stay "checklist".
    if ch == "checklist" and item.get("visible_to_agent", True) and item.get("severity") == "info":
        return "checklist-hint"
    return ch


def trajectory(run_id: str) -> list[dict[str, Any]]:
    trace = _feedback_trace(run_id)
    steps: list[dict] = []
    for entry in trace:
        action = entry.get("action", "code")
        title = {
            "add_cell": "Добавлена ячейка",
            "edit_cell": "Изменена ячейка",
            "delete_cell": "Удалена ячейка",
            "run_cell": "Выполнена ячейка",
            "restart_and_run_all": "Чистый перезапуск",
            "validate": "Валидация кандидата",
            "submit": "Финальный сабмит",
        }.get(action, action)
        feedback = [
            {"ch": _channel(it), "text": it.get("message", "")}
            for it in (entry.get("feedback_items") or [])
        ]
        # surface runtime stderr as a runtime feedback item when present
        if entry.get("stderr"):
            feedback.append({"ch": "runtime", "text": entry["stderr"].strip()[:2000]})
        ui_action = "submit" if action == "submit" else ("validate" if action == "validate" else "code")
        steps.append(
            {
                "step": entry.get("step"),
                "action": ui_action,
                "title": title,
                "code": entry.get("code") or "",
                "budgetRemaining": entry.get("budget_remaining"),
                "feedback": feedback,
            }
        )
    return steps


def errors(run_id: str) -> list[dict[str, Any]]:
    out: list[dict] = []
    for ev in _events(run_id):
        res = ev.get("execution_result") or {}
        if res.get("error_name") or (res.get("success") is False):
            tb = res.get("traceback") or []
            out.append(
                {
                    "step": ev.get("step"),
                    "cellId": ev.get("cell_id"),
                    "type": res.get("error_name") or "Error",
                    "value": res.get("error_value") or "",
                    "traceback": "\n".join(tb) if isinstance(tb, list) else str(tb),
                    "stderr": res.get("stderr", ""),
                }
            )
    return out


def logs(run_id: str) -> list[dict[str, Any]]:
    """Reconstruct an agent<->env dialogue from the public feedback trace."""
    trace = _feedback_trace(run_id)
    msgs: list[dict] = []
    for entry in trace:
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


def checklist(run_id: str, target_col: str = "") -> dict[str, Any]:
    """Replay public notebook events through the gym's NotebookChecklist to get
    exact per-item closure + the step each item was first covered."""
    items_meta = [{"id": k, "label": _CHECK_LABELS.get(k, k), "desc": ""} for k in _CHECK_LABELS]
    closed_step: dict[str, int | None] = {k: None for k in _CHECK_LABELS}
    coverage = None
    try:
        from gym.feedback import GENERIC_CHECKLIST_HINTS, NotebookChecklist

        for it in items_meta:
            it["desc"] = GENERIC_CHECKLIST_HINTS.get(it["id"], "")

        cl = NotebookChecklist(target_col=target_col or "target")
        prev_covered: set[str] = set()
        for ev in _events(run_id):
            res = ev.get("execution_result") or {}
            src = ev.get("source_after") or ""
            cl.record_execution(
                source=src,
                stdout=res.get("stdout", ""),
                cell_id=ev.get("cell_id"),
                step=ev.get("step", 0),
                execution_success=bool(res.get("success", True)),
            )
            newly = set(cl.covered) - prev_covered
            for key in newly:
                if key in closed_step and closed_step[key] is None:
                    closed_step[key] = ev.get("step")
            prev_covered = set(cl.covered)
        coverage = cl.coverage()
    except Exception:
        pass

    summary = get_run(run_id) or {}
    if coverage is None:
        coverage = summary.get("checklistCoverage")

    items = [
        {**meta, "closed": closed_step[meta["id"]] is not None, "closedStep": closed_step[meta["id"]]}
        for meta in items_meta
    ]
    return {
        "items": items,
        "coverage": coverage,
        "closed": sum(1 for i in items if i["closed"]),
        "total": len(items),
    }
