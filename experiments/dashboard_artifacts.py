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
        "step": 1, "action": "add_cell", "cell_id": "cell_01", "source_after": code,
        "executed": True,
        "execution_result": {
            "execution_count": 1, "outputs": outputs, "stdout": stdout, "stderr": stderr,
            "error_name": error_name, "error_value": stderr[:500] if error_name else None,
            "traceback": [stderr] if error_name else [], "success": error_name is None,
        },
    }
    _write(ws / "notebook_events.json", [event])
    _write(ws / "feedback_trace.json", [{
        "action": "add_cell", "step": 1, "code": code, "stdout": stdout,
        "stderr": stderr, "feedback_items": [],
    }])
    _write(ws / "episode_summary.json", {
        "steps_used": steps, "error_count": 1 if error_name else 0,
        "checklist_coverage": coverage,
    })
    return coverage
