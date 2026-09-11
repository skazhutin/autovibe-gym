import json
import re
import sys
import types
from contextlib import nullcontext
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
from gym.candidates import CandidateRecord
from gym.terminal_contract import TerminalContractResult
from research.budget import EpisodeBudget, EpisodeBudgetPolicy, FinalizationReservePolicy
from research.stopping import FrozenStoppingPolicy, StoppingController


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


def test_run_baseline_finalizes_manifest_when_budget_stops_before_llm(monkeypatch):
    frame = pd.DataFrame({"x": range(12), "target": [0, 1] * 6})
    metadata = types.SimpleNamespace(
        name="fixture",
        seed=42,
        split_strategy="fixture-split",
        role="test",
        sampled=False,
    )
    splits = types.SimpleNamespace(
        train=frame.copy(),
        val=frame.copy(),
        test=frame.copy(),
        target_col="target",
        metadata=metadata,
    )

    class FailIfCalledClient:
        def complete(self, **_kwargs):
            raise AssertionError("provider call must be stopped by the global budget")

    finalized = []
    fake_mlflow = types.SimpleNamespace(
        set_experiment=lambda *_args, **_kwargs: None,
        start_run=lambda **_kwargs: nullcontext(),
        log_params=lambda *_args, **_kwargs: None,
        set_tags=lambda *_args, **_kwargs: None,
        log_metrics=lambda *_args, **_kwargs: None,
        log_text=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(run_baseline, "mlflow", fake_mlflow)
    monkeypatch.setattr(run_baseline, "configure_mlflow_tracking", lambda *_args: None)
    monkeypatch.setattr(run_baseline, "load_dataset_splits", lambda **_kwargs: splits)
    monkeypatch.setattr(run_baseline, "resolve_metric", lambda *_args: (lambda *_a: 1.0, "accuracy"))
    monkeypatch.setattr(run_baseline, "build_dataset_card", lambda *_args, **_kwargs: "card")
    monkeypatch.setattr(run_baseline, "dataset_source_hash", lambda **_kwargs: "hash")
    monkeypatch.setattr(run_baseline, "apply_model_reference", lambda model: model)
    monkeypatch.setattr(run_baseline, "make_llm_client", FailIfCalledClient)
    monkeypatch.setattr(run_baseline, "start_research_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_baseline,
        "finalize_research_run",
        lambda _recorder, summary, **_kwargs: finalized.append(dict(summary)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline",
            "--dataset",
            "fixture.csv",
            "--target",
            "target",
            "--model",
            "fixture-model",
            "--research-run-dir",
            "runs",
            "--research-experiment-id",
            "paper-v1",
            "--research-total-token-limit",
            "1",
        ],
    )

    run_baseline.main()

    assert finalized[0]["final_status"] == "budget_exhausted"
    assert finalized[0]["finalize_path"] == "budget_stop"
    assert finalized[0]["steps_used"] == 0
    assert finalized[0]["hidden_evaluations"] == 0


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


def test_run_multishot_validation_boundary_prompt_is_explicit_and_opt_in():
    assert (
        run_multishot._system_prompt_for_validation_boundary(
            labels_host_only=False
        )
        == run_multishot.SYSTEM_PROMPT
    )

    controlled = run_multishot._system_prompt_for_validation_boundary(
        labels_host_only=True
    )

    assert "target labels are host-only" in controlled
    assert "model.predict(val_df.head())" in controlled
    assert "val_df.drop(columns=[target_col])" not in controlled


def test_run_multishot_m4_observers_share_frozen_candidate_and_boundary_rules():
    policy = FrozenStoppingPolicy(
        reserve_boundary_resources=("llm_calls",),
        no_improvement_patience=1,
        stop_on_exploration_exhausted=True,
        stop_on_agent_finalize_request=False,
        stop_on_unrecoverable_failure=True,
    )
    incumbent = types.SimpleNamespace(candidate_id="first")

    class Registry:
        def incumbent(self):
            return incumbent

    terminal = types.SimpleNamespace(registry=Registry())
    controller = StoppingController(policy)
    run_multishot._observe_repeated_candidate(
        controller,
        previous_incumbent=incumbent,
        record=types.SimpleNamespace(candidate_id="second"),
        terminal=terminal,
    )

    assert controller.decision.reason == "no_validation_improvement"
    assert controller.decision.incumbent_candidate_id == "first"

    boundary_controller = StoppingController(policy)
    budget = types.SimpleNamespace(
        exploration_boundary_resources=lambda: ("llm_calls",)
    )
    run_multishot._observe_repeated_boundary(
        boundary_controller,
        budget,
        terminal,
    )
    assert boundary_controller.decision.reason == "reserve_boundary_reached"


def test_run_multishot_uses_inactive_m5_path_when_reserve_is_injected(
    monkeypatch,
):
    frame = pd.DataFrame({"x": range(12), "target": [1] * 12})
    metadata = types.SimpleNamespace(
        name="fixture",
        seed=42,
        split_strategy="fixture-split",
        role="test",
        sampled=False,
    )
    splits = types.SimpleNamespace(
        train=frame.copy(),
        val=frame.copy(),
        test=frame.copy(),
        target_col="target",
        metadata=metadata,
    )
    budget = EpisodeBudget(
        EpisodeBudgetPolicy(
            total_token_limit=100_000,
            max_output_tokens_per_call=100,
            max_llm_calls=1,
            max_code_executions=2,
            max_tool_calls=3,
            wall_clock_limit_seconds=120,
        ),
        finalization_reserve=FinalizationReservePolicy(
            max_code_executions=1,
            max_tool_calls=3,
            wall_clock_limit_seconds=60,
        ),
    )
    candidate = CandidateRecord(
        candidate_id="fixture-candidate",
        model_var="model",
        notebook_revision=1,
        clean_run_id="attempt-0",
        metric_name="accuracy",
        validation_metric=1.0,
        validation_success=True,
        raw_inference_ready=True,
    )
    calls = []

    class FakeClient:
        def complete(self, **_kwargs):
            return LLMResponse(
                text="model = object()",
                input_tokens=1,
                output_tokens=1,
            )

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def run(self, code, namespace):
            exec(code, namespace)
            return "", "", namespace

    class FakeRegistry:
        def incumbent(self):
            return candidate

    class FakeAdapter:
        def __init__(self, **kwargs):
            calls.append("adapter_created")
            self.registry = FakeRegistry()
            self.hidden_gate = kwargs["hidden_gate"]

        def validate_and_register(self, **kwargs):
            calls.append("candidate_registered")
            if kwargs.get("before_validation"):
                kwargs["before_validation"]()
            return candidate

        def finalize(self, **kwargs):
            calls.append("terminal_finalize")
            kwargs["execute_code"]("terminal_replay = True", kwargs["namespace_factory"]())
            kwargs["before_preflight"]()
            kwargs["before_artifact_write"]()
            kwargs["before_hidden_evaluation"]()
            self.hidden_gate.attempted = True
            return TerminalContractResult(
                final_status="submitted_protected_replay",
                candidate=candidate,
                validation_metric=1.0,
                hidden_metric=1.0,
            )

    summaries = []
    fake_mlflow = types.SimpleNamespace(
        set_experiment=lambda *_args, **_kwargs: None,
        start_run=lambda **_kwargs: nullcontext(),
        log_params=lambda *_args, **_kwargs: None,
        set_tags=lambda *_args, **_kwargs: None,
        log_metrics=lambda *_args, **_kwargs: None,
        log_text=lambda *_args, **_kwargs: None,
        log_metric=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(run_multishot, "mlflow", fake_mlflow)
    monkeypatch.setattr(run_multishot, "configure_mlflow_tracking", lambda *_a: None)
    monkeypatch.setattr(run_multishot, "load_dataset_splits", lambda **_kwargs: splits)
    monkeypatch.setattr(
        run_multishot,
        "resolve_metric",
        lambda *_args: (lambda *_a: 1.0, "accuracy"),
    )
    monkeypatch.setattr(run_multishot, "build_dataset_card", lambda *_a, **_k: "card")
    monkeypatch.setattr(run_multishot, "dataset_source_hash", lambda **_kwargs: "hash")
    monkeypatch.setattr(run_multishot, "apply_model_reference", lambda model: model)
    monkeypatch.setattr(run_multishot, "make_llm_client", FakeClient)
    monkeypatch.setattr(run_multishot, "CodeExecutor", FakeExecutor)
    monkeypatch.setattr(run_multishot, "create_episode_budget", lambda _args: budget)
    monkeypatch.setattr(run_multishot, "start_research_run", lambda *_a, **_k: None)
    monkeypatch.setattr(run_multishot, "RepeatedSingleShotTerminalAdapter", FakeAdapter)
    monkeypatch.setattr(
        run_multishot,
        "finalize_research_run",
        lambda _recorder, summary, **_kwargs: summaries.append(dict(summary)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multishot",
            "--dataset",
            "fixture.csv",
            "--target",
            "target",
            "--model",
            "fixture-model",
            "--research-v2-metric-direction",
            "higher",
            "--research-v2-score-tolerance",
            "0.000001",
        ],
    )

    run_multishot.main()

    assert calls == ["adapter_created", "candidate_registered", "terminal_finalize"]
    assert summaries[0]["final_status"] == "submitted_protected_replay"
    assert summaries[0]["finalize_path"] == "protected_incumbent_replay"
    assert summaries[0]["test_metric"] == 1.0
    assert summaries[0]["hidden_evaluations"] == 1
    snapshot = budget.snapshot()
    assert snapshot["budget_phase"] == "finalization"
    assert snapshot["finalization_usage"]["code_executions"] == 1
    assert snapshot["finalization_usage"]["tool_calls"] == 3


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
        "--model", "fake-model",
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
        "--model", "fake-model",
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
    assert "fixed gym" in out
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


def test_common_run_exits_nonzero_when_child_fails_without_stop_on_failure(monkeypatch, capsys):
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
        ],
    )

    def fake_run(command):
        return type("Completed", (), {"returncode": 7})()

    monkeypatch.setattr(run_cli.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        run_cli.main()

    assert exc.value.code == 7
    out = capsys.readouterr().out
    assert "[run] Summary" in out
    assert '"returncode": 7' in out


