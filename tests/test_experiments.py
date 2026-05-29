import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from experiments import compare, mlflow_config, run_baseline, run_fixed, run_gym, run_multishot
from experiments import run_matrix


def test_run_gym_load_dataset_returns_splits_and_metadata(tmp_path):
    prepared = tmp_path / "demo" / "prepared"
    prepared.mkdir(parents=True)
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    for split in ("train", "val", "test"):
        df.to_csv(prepared / f"{split}.csv", index=False)
    (prepared / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "target_col": "y",
                "metric": "f1_macro",
                "suite": "unit",
                "split_strategy": "fixed",
                "role": "test",
                "sampled": False,
            }
        ),
        encoding="utf-8",
    )

    train, val, test, meta = run_gym.load_dataset(str(tmp_path / "demo"))

    assert train.equals(df)
    assert val.equals(df)
    assert test.equals(df)
    assert meta["name"] == "demo"
    assert meta["metric"] == "f1_macro"
    assert meta["suite"] == "unit"


def test_run_gym_dataset_name_falls_back_to_file_stem():
    splits = type("Splits", (), {"metadata": type("Meta", (), {"name": ""})()})()

    assert run_gym._dataset_name(splits, "datasets/my_data.csv") == "my_data"
    assert run_gym._dataset_name(splits, None) == "dataset"


def test_run_gym_logs_kernel_backend_from_environment(monkeypatch):
    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "docker")
    assert run_gym._kernel_backend_label() == "jupyter-docker"

    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "local")
    assert run_gym._kernel_backend_label() == "jupyter-local"


def test_run_baseline_extract_code_prefers_python_fence():
    text = "explain\n```python\nprint('ok')\n```"

    assert run_baseline.extract_code(text) == "print('ok')"


def test_run_baseline_extract_code_accepts_plain_fence_and_plain_text():
    assert run_baseline.extract_code("```\nx = 1\n```") == "x = 1"
    assert run_baseline.extract_code("x = 2") == "x = 2"


def test_run_multishot_extract_code_and_feedback():
    assert run_multishot._extract_code("```python\nx = 1\n```") == "x = 1"

    feedback = run_multishot._build_feedback("out", "err", 3)

    assert "[OUTPUT]\nout" in feedback
    assert "[ERROR]\nerr" in feedback
    assert "[BUDGET] 3 shots remaining" in feedback


def test_run_multishot_feedback_omits_empty_output_sections():
    feedback = run_multishot._build_feedback("", "", 0)

    assert "[OUTPUT]" not in feedback
    assert "[ERROR]" not in feedback
    assert "0 shots remaining" in feedback


def test_run_fixed_summary_metrics_do_not_log_missing_test_metric_as_zero():
    metrics = run_fixed._summary_metrics(
        {
            "test_metric": None,
            "checklist_coverage": 0.5,
            "steps_used": 3,
            "errors_count": 1,
            "input_tokens": 10,
            "output_tokens": 4,
            "elapsed_seconds": 2.0,
        }
    )

    assert "test_metric" not in metrics
    assert metrics["has_test_metric"] == 0
    assert metrics["submit_failed"] == 1


def test_configure_mlflow_tracking_ignores_placeholder_uri(monkeypatch):
    calls = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://<server-ip>:5000")
    mlflow = type("MLflow", (), {"set_tracking_uri": lambda self, uri: calls.append(uri)})()

    tracking_uri = mlflow_config.configure_mlflow_tracking(mlflow)

    assert tracking_uri.startswith("sqlite:///")
    assert calls == [tracking_uri]


def test_configure_mlflow_tracking_accepts_real_uri(monkeypatch):
    calls = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow = type("MLflow", (), {"set_tracking_uri": lambda self, uri: calls.append(uri)})()

    assert mlflow_config.configure_mlflow_tracking(mlflow) == "file:./mlruns"
    assert calls == ["file:./mlruns"]


def test_configure_mlflow_tracking_defaults_to_local_sqlite(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    mlflow = type("MLflow", (), {"set_tracking_uri": lambda self, uri: calls.append(uri)})()

    tracking_uri = mlflow_config.configure_mlflow_tracking(mlflow)

    assert tracking_uri == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    assert calls == [tracking_uri]


def test_compare_prints_no_runs_message(monkeypatch, capsys):
    monkeypatch.setattr(compare.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        compare.mlflow,
        "search_runs",
        lambda **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr("sys.argv", ["compare"])

    compare.main()

    assert "No runs found" in capsys.readouterr().out


def test_compare_prints_sorted_table_and_writes_csv(monkeypatch, tmp_path, capsys):
    runs = pd.DataFrame(
        {
            "params.experiment_type": ["gym", "baseline"],
            "params.model": ["m1", "m2"],
            "params.dataset": ["d", "d"],
            "params.mode": ["local", "local"],
            "metrics.test_metric": [0.2, 0.8],
            "metrics.error_count": [1, 0],
        }
    )
    output = tmp_path / "table.csv"
    monkeypatch.setattr(compare.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(compare.mlflow, "search_runs", lambda **kwargs: runs)
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--output", str(output)],
    )

    compare.main()

    printed = capsys.readouterr().out
    assert "Experiment: autovibe-gym" in printed
    assert output.exists()
    saved = pd.read_csv(output)
    assert list(saved["test_metric"]) == [0.8, 0.2]


def test_compare_handles_runs_without_test_metric(monkeypatch, capsys):
    runs = pd.DataFrame(
        {
            "params.experiment_type": ["gym"],
            "params.model": ["m1"],
            "params.dataset": ["d"],
            "params.mode": ["local"],
            "metrics.has_test_metric": [0],
            "metrics.submit_failed": [1],
        }
    )
    monkeypatch.setattr(compare.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(compare.mlflow, "search_runs", lambda **kwargs: runs)
    monkeypatch.setattr("sys.argv", ["compare"])

    compare.main()

    printed = capsys.readouterr().out
    assert "submit_failed" in printed
    assert "m1" in printed


# ---------------------------------------------------------------------------
# run_gym — executor_backend MLflow param reflects AUTOVIBE_KERNEL_BACKEND
# ---------------------------------------------------------------------------

def test_run_gym_executor_backend_param_reflects_env(monkeypatch):
    """_kernel_backend_label() must return the correct label for each backend value."""
    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "docker")
    assert run_gym._kernel_backend_label() == "jupyter-docker"

    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "local")
    assert run_gym._kernel_backend_label() == "jupyter-local"

    monkeypatch.delenv("AUTOVIBE_KERNEL_BACKEND", raising=False)
    assert run_gym._kernel_backend_label() == "jupyter-local"


# ---------------------------------------------------------------------------
# run_matrix — batch orchestrator unit tests
# ---------------------------------------------------------------------------

def test_run_matrix_dry_run_prints_plan(tmp_path, capsys, monkeypatch):
    """--dry-run should print the matrix plan without calling subprocess.run."""
    prepared = tmp_path / "ds_a" / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("sys.argv", [
        "run_matrix",
        "--datasets", str(tmp_path / "ds_a"),
        "--episode-modes", "gym_with_checklist",
        "--dry-run",
    ])

    run_matrix.main()

    out = capsys.readouterr().out
    assert "ds_a" in out
    assert "gym_with_checklist" in out
    assert "dry-run" in out


def test_run_matrix_discover_datasets(tmp_path):
    """_discover_datasets finds directories that contain prepared/meta.json."""
    for name in ("ds_x", "ds_y"):
        (tmp_path / name / "prepared").mkdir(parents=True)
        (tmp_path / name / "prepared" / "meta.json").write_text("{}", encoding="utf-8")
    # A directory without prepared/meta.json should NOT be included
    (tmp_path / "not_a_dataset").mkdir()

    found = run_matrix._discover_datasets(str(tmp_path))
    names = {Path(d).name for d in found}
    assert names == {"ds_x", "ds_y"}


def test_run_matrix_exits_when_no_datasets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "run_matrix",
        "--datasets-root", str(tmp_path),
        "--dry-run",
    ])

    with pytest.raises(SystemExit) as exc:
        run_matrix.main()

    assert exc.value.code == 1


