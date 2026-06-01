from __future__ import annotations

import sys
from types import SimpleNamespace

from dashboard.server.app.config import default_python_bin
from dashboard.server.app.services import mlflow_store, run_launcher


def _fake_run(params: dict, metrics: dict, status: str = "FINISHED"):
    return SimpleNamespace(
        data=SimpleNamespace(
            params=params,
            metrics=metrics,
            tags={"mlflow.runName": "dash_unit"},
        ),
        info=SimpleNamespace(
            run_id="1234567890abcdef",
            status=status,
            start_time=1000,
            end_time=2000,
            experiment_id="0",
        ),
    )


def test_default_python_bin_finds_windows_venv_layout(tmp_path):
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert default_python_bin(tmp_path) == str(python)


def test_default_python_bin_finds_unix_venv_layout(tmp_path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert default_python_bin(tmp_path) == str(python)


def test_default_python_bin_falls_back_to_current_interpreter(tmp_path):
    assert default_python_bin(tmp_path) == sys.executable


def test_run_record_maps_baseline_single_shot_and_infers_one_step():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "baseline_single_shot",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={
                "final_test_metric": 0.75,
                "has_test_metric": 1,
                "valid_submit": 1,
            },
        )
    )

    assert record["mode"] == "single"
    assert record["status"] == "success"
    assert record["score"] == 0.75
    assert record["step"] == 1
    assert record["steps"] == 1


def test_run_record_maps_repeated_single_shot_attempt_progress():
    record = mlflow_store._run_record(
        _fake_run(
            params={
                "experiment_type": "repeated_single_shot",
                "max_attempts": "5",
                "model": "fake-model",
                "dataset": "unit-ds",
            },
            metrics={
                "final_test_metric": 0.8,
                "has_test_metric": 1,
                "valid_submit": 1,
                "attempts_used": 3,
            },
        )
    )

    assert record["mode"] == "repeated"
    assert record["status"] == "success"
    assert record["score"] == 0.8
    assert record["step"] == 3
    assert record["steps"] == 5


def test_run_record_hides_placeholder_zero_score_for_failed_submit():
    record = mlflow_store._run_record(
        _fake_run(
            params={"experiment_type": "gym_with_checklist"},
            metrics={
                "test_metric": 0.0,
                "has_test_metric": 0,
                "submit_failed": 1,
            },
        )
    )

    assert record["status"] == "failed"
    assert record["score"] is None


def test_checklist_uses_authoritative_fallback_when_episode_artifacts_missing():
    data = mlflow_store.checklist(None, fallback_coverage=0.88)

    assert data["coverage"] == 0.88
    assert data["closed"] == round(0.88 * data["total"])


def test_run_launcher_planned_steps_match_dashboard_modes():
    assert run_launcher._planned_steps({"mode": "single"}) == 1
    assert run_launcher._planned_steps({"mode": "repeated", "shots": 4}) == 4
    assert run_launcher._planned_steps({"mode": "gym", "maxSteps": 8}) == 8
    assert run_launcher._planned_steps({"mode": "iterative", "maxSteps": 6}) == 6


def test_run_launcher_python_available_accepts_paths_and_rejects_missing(tmp_path):
    assert run_launcher._python_available(sys.executable)
    assert not run_launcher._python_available(str(tmp_path / "missing-python"))
