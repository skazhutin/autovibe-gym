import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments import compare, mlflow_config, run_baseline, run_fixed, run_gym, run_multishot
from experiments import run as run_cli
from experiments import run_all_modes_matrix
from experiments import run_matrix
from experiments.modes import expand_requested_mode
from gym.llm import LLMResponse
from gym.notebook_env import NotebookGymEnv


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


def test_run_fixed_tool_only_stage_stops_at_turn_guard(tmp_path):
    class ToolOnlyClient:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return LLMResponse(text='{"type": "inspect_data"}', input_tokens=1, output_tokens=1)

    def accuracy(y_true, y_pred):
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    data = pd.DataFrame({"x": [0, 1], "target": [0, 1]})
    env = NotebookGymEnv(
        train=data,
        val=data,
        test=data,
        target_col="target",
        metric_fn=accuracy,
        metric_name="accuracy",
        max_steps=10,
        workspace_dir=tmp_path,
        mode="gym_with_checklist",
    )
    client = ToolOnlyClient()
    agent = run_fixed.FixedTransitionsAgent(
        env=env,
        stages=[{"name": "eda", "label": "Stage 1/1 - EDA", "goal": "Inspect data.", "budget": 1}],
        model="fake-model",
        client=client,
    )
    try:
        summary = agent.run()
    finally:
        env.close()

    first_stage = summary["stage_log"][0]
    assert first_stage["turns"] == 4
    assert first_stage["tool_calls"] == 4
    assert first_stage["code_steps"] == 0
    assert first_stage["stop_reason"] == "max_stage_turns"
    assert client.calls == 4


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
    assert "Sorted by: matrix" in printed
    assert output.exists()
    saved = pd.read_csv(output)
    assert list(saved["model"]) == ["m1", "m2"]
    assert list(saved["test_metric"]) == [0.2, 0.8]


def test_compare_metric_sort_orders_by_selected_metric(monkeypatch, capsys):
    runs = pd.DataFrame(
        {
            "params.experiment_type": ["gym", "baseline"],
            "params.model": ["m1", "m2"],
            "params.dataset": ["d", "d"],
            "params.mode": ["local", "local"],
            "metrics.test_metric": [0.2, 0.8],
        }
    )
    monkeypatch.setattr(compare.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(compare.mlflow, "search_runs", lambda **kwargs: runs)
    monkeypatch.setattr("sys.argv", ["compare", "--sort-by", "metric"])

    compare.main()

    printed = capsys.readouterr().out
    assert "Sorted by: test_metric" in printed
    assert printed.index("m2") < printed.index("m1")


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


def test_compare_groups_all_batch_by_mode_order(monkeypatch, capsys):
    runs = pd.DataFrame(
        {
            "params.experiment_type": [
                "gym_with_checklist",
                "baseline_single_shot",
                "fixed_transitions",
                "repeated_single_shot",
                "iterative_no_checklist",
            ],
            "params.requested_mode": ["all", "all", "all", "all", "all"],
            "params.batch_id": ["batch-1", "batch-1", "batch-1", "batch-1", "batch-1"],
            "params.product_mode": [
                "gym_with_checklist",
                "single_shot",
                "fixed_transitions",
                "repeated_single_shot",
                "iterative_no_checklist",
            ],
            "params.mode_label": [
                "gym_with_checklist",
                "single_shot",
                "fixed_transitions",
                "repeated_single_shot",
                "iterative_no_checklist",
            ],
            "params.mode_order": [4, 1, 5, 2, 3],
            "params.model": ["m1", "m1", "m1", "m1", "m1"],
            "params.dataset": ["d", "d", "d", "d", "d"],
            "metrics.test_metric": [0.4, 0.1, 0.5, 0.2, 0.3],
        }
    )
    monkeypatch.setattr(compare.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(compare.mlflow, "search_runs", lambda **kwargs: runs)
    monkeypatch.setattr("sys.argv", ["compare"])

    compare.main()

    printed = capsys.readouterr().out
    assert "requested_mode" in printed
    assert "batch_id" in printed
    assert printed.index("single_shot") < printed.index("repeated_single_shot")
    assert printed.index("repeated_single_shot") < printed.index("iterative_no_checklist")
    assert printed.index("iterative_no_checklist") < printed.index("gym_with_checklist")
    assert printed.index("gym_with_checklist") < printed.index("fixed_transitions")


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


def test_run_all_modes_matrix_dry_run_lists_exact_five_modes(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_all_modes_matrix",
            "--datasets",
            "datasets/demo/prepared",
            "--models",
            "fake-model",
            "--dry-run",
        ],
    )

    run_all_modes_matrix.main()

    out = capsys.readouterr().out
    assert "single-shot" in out
    assert "repeated single-shot" in out
    assert "iterative no-checklist" in out
    assert "flexible gym" in out
    assert "fixed transitions gym" in out
    assert out.count("fake-model") >= 5
    batch_ids = re.findall(r"--batch-id\s+(\S+)", out)
    assert len(batch_ids) >= 5
    assert len(set(batch_ids)) == 1


def test_shared_modes_all_expands_to_five_product_modes():
    assert [m.key for m in expand_requested_mode("all")] == [
        "single_shot",
        "repeated_single_shot",
        "iterative_no_checklist",
        "gym_with_checklist",
        "fixed_transitions",
    ]


def test_common_run_all_dry_run_lists_five_commands_with_shared_batch(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--dataset-dir",
            "datasets/demo/prepared",
            "--mode",
            "all",
            "--model",
            "fake-model",
            "--dry-run",
        ],
    )

    run_cli.main()

    out = capsys.readouterr().out
    assert "[run] Planned 5 run(s)" in out
    assert "single_shot" in out
    assert "repeated_single_shot" in out
    assert "iterative_no_checklist" in out
    assert "gym_with_checklist" in out
    assert "fixed_transitions" in out
    batch_ids = re.findall(r"--batch-id\s+(\S+)", out)
    assert len(batch_ids) >= 5
    assert len(set(batch_ids)) == 1


def test_common_run_selected_modes_dry_run_lists_selected_commands_with_shared_batch(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--dataset-dir",
            "datasets/demo/prepared",
            "--modes",
            "single_shot",
            "gym_with_checklist",
            "fixed_transitions",
            "--model",
            "fake-model",
            "--run-name",
            "unit_batch",
            "--workspace-dir",
            "workspace",
            "--dry-run",
        ],
    )

    run_cli.main()

    out = capsys.readouterr().out
    assert "[run] requested_mode=batch" in out
    assert "[run] Planned 3 run(s)" in out
    assert "single_shot" in out
    assert "gym_with_checklist" in out
    assert "fixed_transitions" in out
    assert "repeated_single_shot" not in out
    assert "unit_batch_single_shot" in out
    assert "workspace\\single_shot" in out or "workspace/single_shot" in out
    batch_ids = re.findall(r"--batch-id\s+(\S+)", out)
    assert len(batch_ids) >= 3
    assert len(set(batch_ids)) == 1


def test_common_run_single_dry_run_stays_single(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--dataset-dir",
            "datasets/demo/prepared",
            "--mode",
            "single_shot",
            "--model",
            "fake-model",
            "--dry-run",
        ],
    )

    run_cli.main()

    out = capsys.readouterr().out
    assert "[run] Planned 1 run(s)" in out
    assert "single_shot" in out


