"""Write dashboard-readable episode artifacts for the non-notebook runners
(single-shot / repeated). The dashboard reads solution.ipynb / notebook_events
/ feedback_trace from a run's workspace to populate the Notebook, Trajectory,
Errors and Logs tabs, and replays notebook events through the gym checklist to
show coverage. These runners don't use the notebook env, so we synthesize the
same files from the single generated code block.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from gym.feedback import NotebookChecklist
except Exception:  # pragma: no cover
    NotebookChecklist = None  # type: ignore


def checklist_coverage(code: str, stdout: str, target_col: str) -> float | None:
    if NotebookChecklist is None:
        return None
    try:
        cl = NotebookChecklist(target_col=target_col or "target")
        cl.record_execution(source=code, stdout=stdout or "", cell_id="cell_01",
                            step=1, execution_success=True)
        return cl.coverage()
    except Exception:
        return None


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


def write_episode_artifacts(
    workspace_dir: str | Path,
    *,
    code: str,
    stdout: str = "",
    stderr: str = "",
    error_name: str | None = None,
    target_col: str = "",
    coverage: float | None = None,
    steps: int = 1,
) -> float | None:
    """Emit solution.ipynb + notebook_events.json + feedback_trace.json +
    episode_summary.json into workspace_dir. Returns the checklist coverage."""
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    if coverage is None:
        coverage = checklist_coverage(code, stdout, target_col)

    outputs: list[dict] = []
    if stdout:
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout})
    if error_name:
        outputs.append({"output_type": "error", "ename": error_name,
                        "evalue": stderr[:500], "traceback": [stderr]})
    elif stderr:
        outputs.append({"output_type": "stream", "name": "stderr", "text": stderr})

    nb = {
        "cells": [{"cell_type": "code", "execution_count": 1, "metadata": {},
                   "source": code, "outputs": outputs}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    _write(ws / "solution.ipynb", nb)

    event = {
        "step": 1, "type": "add_cell", "stage": "candidate_training",
        "cell_id": "cell_01", "source_after": code,
        "executed": True,
        "execution_result": {
            "execution_count": 1, "outputs": outputs, "stdout": stdout, "stderr": stderr,
            "error_name": error_name, "error_value": stderr[:500] if error_name else None,
            "traceback": [stderr] if error_name else [], "success": error_name is None,
        },
    }
    _write(ws / "notebook_events.json", [event])
    _write(ws / "feedback_trace.json", [{
        "type": "add_cell", "stage": "candidate_training",
        "step": 1, "code": code, "stdout": stdout,
        "stderr": stderr, "feedback_items": [],
    }])
    _write(ws / "episode_summary.json", {
        "steps_used": steps, "error_count": 1 if error_name else 0,
        "checklist_coverage": coverage, "current_stage": "candidate_training",
    })
    return coverage


def _attempt_outputs(stdout: str, stderr: str, error_name: str | None) -> list[dict]:
    outs: list[dict] = []
    if stdout:
        outs.append({"output_type": "stream", "name": "stdout", "text": stdout})
    if error_name:
        outs.append({"output_type": "error", "ename": error_name,
                     "evalue": (stderr or "")[:500], "traceback": [stderr or ""]})
    elif stderr:
        outs.append({"output_type": "stream", "name": "stderr", "text": stderr})
    return outs


def write_attempts_episode(
    workspace_dir: str | Path,
    *,
    attempts: list[dict[str, Any]],
    target_col: str = "",
    coverage: float | None = None,
    metric_name: str = "metric",
) -> float | None:
    """Emit a multi-step episode for the repeated single-shot runner so EVERY
    attempt — including ones that crashed or never produced a model — shows up in
    the dashboard Notebook / Trajectory / Errors / Logs tabs, not just the best
    one. Each ``attempts`` item is a dict with keys: ``attempt`` (1-based),
    ``code``, ``stdout``, ``stderr``, ``error_name`` (or None), ``val_metric``
    (or None), ``is_best`` (bool). Returns the checklist coverage that was used.
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    if not attempts:
        return coverage

    # Coverage: prefer the explicit value; else compute from the best (or last)
    # attempt that actually ran, so a fully-failed run still gets a sane 0-ish.
    if coverage is None:
        ref = next((a for a in attempts if a.get("is_best")), attempts[-1])
        coverage = checklist_coverage(ref.get("code", ""), ref.get("stdout", ""), target_col)

    cells: list[dict] = []
    events: list[dict] = []
    trace: list[dict] = []
    error_count = 0
    for a in attempts:
        step = int(a.get("attempt") or (len(events) + 1))
        code = a.get("code") or ""
        stdout = a.get("stdout") or ""
        stderr = a.get("stderr") or ""
        error_name = a.get("error_name")
        val_metric = a.get("val_metric")
        is_best = bool(a.get("is_best"))
        if error_name:
            error_count += 1
        outs = _attempt_outputs(stdout, stderr, error_name)

        if val_metric is not None:
            head = f"### Попытка {step} — {metric_name} на валидации: {val_metric:.4f}"
        elif error_name:
            head = f"### Попытка {step} — ошибка: {error_name}"
        else:
            head = f"### Попытка {step} — без валидного кандидата"
        if is_best:
            head += "  ·  ✅ лучшая"
        cells.append({"cell_type": "markdown", "metadata": {}, "source": head})
        cells.append({"cell_type": "code", "execution_count": step, "metadata": {},
                      "source": code, "outputs": outs})

        events.append({
            "step": step, "type": "add_cell", "stage": "candidate_training",
            "cell_id": f"attempt_{step:02d}",
            "source_after": code, "executed": True,
            "execution_result": {
                "execution_count": step, "outputs": outs, "stdout": stdout, "stderr": stderr,
                "error_name": error_name, "error_value": (stderr or "")[:500] if error_name else None,
                "traceback": [stderr or ""] if error_name else [], "success": error_name is None,
            },
        })

        feedback_items: list[dict] = []
        if val_metric is not None:
            feedback_items.append({"channel": "validation", "severity": "info",
                                   "message": f"{metric_name} на валидации: {val_metric:.4f}"
                                              + ("  (лучшая попытка)" if is_best else "")})
        if error_name:
            feedback_items.append({"channel": "runtime", "severity": "error",
                                   "message": f"{error_name}: {(stderr or '').strip()[:400]}"})
        trace.append({
            "type": "add_cell", "stage": "candidate_training",
            "step": step, "code": code, "stdout": stdout,
            "stderr": stderr, "feedback_items": feedback_items,
        })

    _write(ws / "solution.ipynb",
           {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    _write(ws / "notebook_events.json", events)
    _write(ws / "feedback_trace.json", trace)
    _write(ws / "episode_summary.json", {
        "steps_used": len(attempts), "error_count": error_count,
        "checklist_coverage": coverage, "current_stage": "candidate_training",
    })
    return coverage
