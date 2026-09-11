import json
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from gym.candidate_store import CandidateBundleError
from gym.finalization import ProtectedFinalizationError
from gym.jupyter_kernel import ContainerJupyterKernelBackend
from gym.notebook_env import NotebookGymEnv
from gym.protocol import Action
from research.budget import (
    BudgetExhausted,
    EpisodeBudget,
    EpisodeBudgetPolicy,
    ExplorationBudgetExhausted,
    FinalizationReservePolicy,
    ValidationQueryPolicy,
)
from research.stopping import FrozenStoppingPolicy


def _accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def _make_env(
    tmp_path,
    *,
    mode="gym_with_checklist",
    hidden_green=False,
    enable_thoughts=False,
    episode_budget=None,
    research_submission=False,
    private_dir=None,
    reset=True,
    stopping_policy=None,
):
    train = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "blue"],
            "x": [0, 1, 2, 3],
            "target": [0, 1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "x": [4, 5],
            "target": [0, 1],
        }
    )
    test_color = ["green"] if hidden_green else ["red", "red"]
    test = pd.DataFrame(
        {
            "color": test_color,
            "x": [6] if hidden_green else [6, 7],
            "target": [1] if hidden_green else [0, 0],
        }
    )
    env = NotebookGymEnv(
        train=train,
        val=val,
        test=test,
        target_col="target",
        metric_fn=_accuracy,
        metric_name="accuracy",
        max_steps=20,
        workspace_dir=tmp_path,
        private_dir=private_dir,
        mode=mode,
        enable_thoughts=enable_thoughts,
        episode_budget=episode_budget,
        research_submission=research_submission,
        stopping_policy=stopping_policy,
    )
    if reset:
        env.reset()
    return env


def _protected_budget(*, code_reserve=2, tool_reserve=3):
    return EpisodeBudget(
        EpisodeBudgetPolicy(
            max_code_executions=12,
            max_tool_calls=12,
            wall_clock_limit_seconds=120,
        ),
        finalization_reserve=FinalizationReservePolicy(
            max_code_executions=code_reserve,
            max_tool_calls=tool_reserve,
            wall_clock_limit_seconds=60,
        ),
    )


def _stopping_policy(**overrides):
    values = {
        "reserve_boundary_resources": ("tool_calls",),
        "no_improvement_patience": None,
        "stop_on_exploration_exhausted": True,
        "stop_on_agent_finalize_request": True,
        "stop_on_unrecoverable_failure": True,
    }
    values.update(overrides)
    return FrozenStoppingPolicy(**values)


def _validation_query_budget(
    *,
    max_queries=8,
    finalization_reserve_queries=0,
    feedback_numeric_decimals=3,
    protected=False,
):
    reserve = None
    if protected:
        reserve = FinalizationReservePolicy(
            max_code_executions=2,
            max_tool_calls=3,
            wall_clock_limit_seconds=60,
        )
    return EpisodeBudget(
        EpisodeBudgetPolicy(
            max_code_executions=20,
            max_tool_calls=20,
            wall_clock_limit_seconds=120,
        ),
        finalization_reserve=reserve,
        validation_query_policy=ValidationQueryPolicy(
            max_queries=max_queries,
            finalization_reserve_queries=finalization_reserve_queries,
            feedback_numeric_decimals=feedback_numeric_decimals,
        ),
    )


def test_workspace_contains_train_and_val_but_no_hidden_test(tmp_path):
    env = _make_env(tmp_path)
    try:
        assert (tmp_path / "data" / "train.csv").exists()
        assert (tmp_path / "data" / "val.csv").exists()
        assert not (tmp_path / "data" / "test.csv").exists()

        result = env.step(Action.code_action("print('test_df' in globals())"))
        assert "False" in result.stdout
    finally:
        env.close()


def test_legacy_code_action_creates_and_executes_code_cell(tmp_path):
    env = _make_env(tmp_path)
    try:
        result = env.step({"type": "code", "stage": "feature_pipeline_building", "code": "value = 41\nprint(value + 1)"})

        assert result.cell_id == "cell_01"
        assert "42" in result.stdout
        assert env.notebook.list_cells()[0]["cell_type"] == "code"
    finally:
        env.close()
def test_unknown_cell_id_is_recoverable_blocker_not_a_crash(tmp_path):
    """Targeting a non-existent cell must not crash the episode; the agent gets a
    blocker naming the real cell ids and can keep going (robustness)."""
    env = _make_env(tmp_path)
    try:
        for action in ("update_cell", "delete_cell", "run_cell", "move_cell"):
            obs = env.step({
                "type": action,
                "stage": "feature_pipeline_building",
                "cell_id": "cell_99",
                "source": "x = 1",
                "new_position": 0,
            })
            assert obs.stderr  # contract blocker, not an exception
            assert "does not exist" in obs.stderr
            assert not obs.done
        ok = env.step({"type": "code", "stage": "feature_pipeline_building", "code": "print(1 + 1)"})
        assert "2" in ok.stdout
    finally:
        env.close()


def test_notebook_env_rejects_missing_unknown_stage_and_unknown_type(tmp_path):
    env = _make_env(tmp_path)
    try:
        missing_stage = env.step({"type": "inspect_data"})
        assert missing_stage.action == "inspect_data"
        assert "stage" in missing_stage.stderr
        assert env.state.step == 0

        unknown_stage = env.step({"type": "inspect_data", "stage": "made_up_stage"})
        assert unknown_stage.action == "inspect_data"
        assert "Unknown stage" in unknown_stage.stderr
        assert env.state.step == 0

        unknown_type = env.step({"type": "dance", "stage": "data_schema_inspection"})
        assert unknown_type.action == "invalid_action"
        assert "Unsupported action type" in unknown_type.stderr
        assert env.state.step == 0
    finally:
        env.close()


def test_notebook_env_rejects_thoughts_planning_and_think_when_disabled(tmp_path):
    env = _make_env(tmp_path)
    try:
        thoughted = env.step(
            {
                "type": "inspect_data",
                "stage": "data_schema_inspection",
                "thoughts": "Inspecting the data.",
            }
        )
        assert "Thoughts mode is disabled" in thoughted.stderr
        assert env.state.step == 0

        planning = env.step({"type": "inspect_data", "stage": "planning"})
        assert "stage 'planning' is not allowed" in planning.stderr
        assert env.state.step == 0

        thinking = env.step(
            {
                "type": "think",
                "stage": "validation_analysis",
                "thoughts": "Reflecting on validation.",
            }
        )
        assert "type 'think' is not allowed" in thinking.stderr
        assert env.state.step == 0
    finally:
        env.close()


def test_notebook_env_thoughts_mode_requires_initial_planning_think(tmp_path):
    env = _make_env(tmp_path, enable_thoughts=True)
    try:
        regular = env.step(
            {
                "type": "inspect_data",
                "stage": "data_schema_inspection",
                "thoughts": "I want to inspect the data first.",
            }
        )
        assert "first action must be type 'think' with stage 'planning'" in regular.stderr
        assert env.scratchpad == []
        assert env.state.step == 0

        missing_thoughts = env.step({"type": "think", "stage": "planning"})
        assert "non-empty 'thoughts'" in missing_thoughts.stderr
        assert env.scratchpad == []
        assert env.state.step == 0

        planned = env.step(
            {
                "type": "think",
                "stage": "planning",
                "thoughts": "I will inspect the data, build a pipeline, validate it, and submit only when ready.",
            }
        )
        assert planned.action == "think"
        assert planned.stage == "planning"
        assert planned.thoughts
        assert env.scratchpad[0]["type"] == "think"
        assert env.scratchpad[0]["stage"] == "planning"
        assert env.state.step == 0
    finally:
        env.close()


def test_think_records_artifacts_without_mutating_notebook_kernel_or_budget(tmp_path):
    env = _make_env(tmp_path, enable_thoughts=True)
    try:
        env.kernel.execute_cell("sentinel_value = 123", store_history=False)
        original_train = env.state.train.copy(deep=True)
        before_cells = env.notebook.list_cells()
        before_revision = env.notebook.revision

        first = env.step(
            {
                "type": "think",
                "stage": "planning",
                "thoughts": "I will inspect schema, train a candidate, validate, check reproducibility, and submit.",
            }
        )
        second = env.step(
            {
                "type": "think",
                "stage": "validation_analysis",
                "thoughts": "The next useful move is to inspect validation readiness without changing data.",
            }
        )
        repeated_planning = env.step(
            {
                "type": "think",
                "stage": "planning",
                "thoughts": "Trying to plan again should be rejected.",
            }
        )

        assert first.action == "think"
        assert second.action == "think"
        assert "only allowed for the first" in repeated_planning.stderr
        assert env.state.step == 0
        assert env.notebook.list_cells() == before_cells
        assert env.notebook.revision == before_revision
        assert env.state.train.equals(original_train)
        probe = env.kernel.execute_cell("print(sentinel_value)", store_history=False)
        assert "123" in probe.stdout
        assert len(env.scratchpad) == 2
        assert env.get_summary()["thoughts_count"] == 2
        assert env.get_summary()["current_stage"] == "validation_analysis"

        public_events = json.loads((tmp_path / "notebook_events.json").read_text(encoding="utf-8"))
        public_trace = json.loads((tmp_path / "feedback_trace.json").read_text(encoding="utf-8"))
        scratchpad = json.loads((tmp_path / "scratchpad.json").read_text(encoding="utf-8"))
        public_summary = json.loads((tmp_path / "episode_summary.json").read_text(encoding="utf-8"))

        think_events = [
            event
            for event in public_events
            if event.get("type") == "think" and event.get("non_mutating") is True
        ]
        assert len(think_events) == 2
        assert all(event.get("non_mutating") is True for event in think_events)
        assert "action" not in json.dumps(public_events)
        assert public_trace[-2]["type"] == "think"
        assert public_trace[-2]["stage"] == "validation_analysis"
        assert public_trace[-2]["thoughts"] == second.thoughts
        assert "action" not in public_trace[-2]
        assert scratchpad[-1]["thoughts"] == second.thoughts
        assert public_summary["current_stage"] == "validation_analysis"
    finally:
        env.close()


def test_restart_and_run_all_clears_interactive_stale_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        stale = env.step(
            {"type": "add_cell", "stage": "feature_pipeline_building", "cell_type": "code", "source": "stale = 123", "execute": True}
        )
        env.step({"type": "delete_cell", "stage": "reproducibility_check", "cell_id": stale.cell_id})
        check = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": "print('stale' in globals())",
                "execute": True,
            }
        )
        assert "True" in check.stdout

        clean = env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        assert "successfully" in clean.stdout
        cell = env.notebook.get_cell(check.cell_id)
        assert "False" in cell.outputs[0]["text"]
    finally:
        env.close()


def test_validate_and_submit_require_clean_run(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )

        rejected = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert "restart_and_run_all" in rejected.stderr

        clean = env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        assert "successfully" in clean.stdout

        validated = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert validated.validation_metric == 0.5
        original = env.candidates.latest()
        assert original is not None
        artifact_path = Path(original.artifact_path)
        notebook_snapshot = artifact_path.parent / "solution.ipynb"
        snapshot_before_mutation = notebook_snapshot.read_bytes()

        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "markdown",
                "source": "changed after validation",
            }
        )
        assert env.candidates.latest() is None
        assert env.candidates.latest_registered() == original
        assert env.candidates.incumbent() == original
        assert env.candidates.all() == [original]
        assert original.candidate_id in env._candidate_objects
        assert artifact_path.exists()
        assert notebook_snapshot.read_bytes() == snapshot_before_mutation
        assert env.candidate_store.verify_record(original).record == original
        assert env.get_summary()["best_validation_metric"] == 0.5
        assert env.get_summary()["validated_candidates_total"] == 1
        assert any(
            event.get("action") == "candidate_current_invalidated"
            and event.get("candidate_id") == original.candidate_id
            for event in env.events
        )
        registered_event = next(
            event
            for event in env.events
            if event.get("action") == "candidate_registered"
        )
        public_registered_event = env._public_event(registered_event)
        assert "artifact_path" not in public_registered_event["candidate"]
        assert str(artifact_path) not in json.dumps(public_registered_event)
        bundle_event = next(
            event
            for event in env.events
            if event.get("action") == "candidate_bundle_written"
        )
        public_bundle_event = env._public_event(bundle_event)
        assert public_bundle_event["schema_version"] == "candidate-bundle-v1"
        assert set(public_bundle_event["hashes"]) == {
            "candidate.json",
            "model.pkl",
            "solution.ipynb",
        }
        assert "private" not in public_bundle_event
        assert str(artifact_path.parent) not in json.dumps(public_bundle_event)
        assert str(artifact_path) not in json.dumps(env.build_context_pack())
        dirty_submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert "restart_and_run_all" in dirty_submit.stderr
    finally:
        env.close()


def test_successful_submit_hides_score_from_agent_context_and_keeps_private_summary(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        feedback = submit.to_feedback_message()

        assert submit.submitted
        assert submit.test_metric is None
        assert "test_metric" not in feedback
        assert "final_test_metric" not in feedback
        assert "1.0" not in feedback
        assert env.get_summary()["final_test_metric"] == 1.0

        trace = json.loads((tmp_path / "feedback_trace.json").read_text(encoding="utf-8"))
        assert "final_test_metric" not in json.dumps(trace)
        assert "test_metric" not in json.dumps(trace)

        nb = nbformat.read(tmp_path / "final_notebook.ipynb", as_version=4)
        nbformat.validate(nb)
        assert "1.0" not in json.dumps(nb)
    finally:
        env.close()


def test_hidden_submit_failure_is_generic(tmp_path):
    """Hidden-test submit failures give retry feedback until retries are exhausted.

    First two failures: not submitted (done=False), generic retry message —
    no hidden data details are leaked.
    Third failure: submitted=True (terminal), generic termination message —
    still no hidden data leaked.
    """
    env = _make_env(tmp_path, hidden_green=True)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _strict_one_hot_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        validated = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert not validated.stderr

        # Attempt 1 — retry path: NOT submitted yet, generic retry message.
        submit1 = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert not submit1.submitted, "First hidden-test failure should allow retry"
        assert not submit1.done
        assert "hidden test set" in submit1.stderr.lower() or "hidden" in submit1.stderr.lower()
        assert "green" not in submit1.stderr   # no hidden data leaked
        assert env.get_summary()["submit_failure_type"]

        # Attempt 2 — another retry, must redo restart_and_run_all + validate.
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        submit2 = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert not submit2.submitted, "Second hidden-test failure should still allow retry"
        assert "green" not in submit2.stderr

        # Attempt 3 — retries exhausted: terminal failure.
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        submit3 = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert submit3.submitted, "Third hidden-test failure should terminate the episode"
        assert "hidden test split" in submit3.stderr
        assert "green" not in submit3.stderr
        assert env.get_summary()["submit_failure_type"]
    finally:
        env.close()


def test_research_hidden_submit_is_terminal_and_one_shot_without_feedback(tmp_path):
    budget = EpisodeBudget(EpisodeBudgetPolicy())
    env = _make_env(
        tmp_path,
        hidden_green=True,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _strict_one_hot_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})

        assert submit.submitted
        assert submit.done
        assert submit.stderr == ""
        assert env.hidden_submit_fail_count == 1
        summary = env.get_summary()
        assert summary["hidden_evaluations"] == 1
        assert summary["final_status"] == "hidden_submit_failed"
    finally:
        env.close()


def test_research_finalize_never_repairs_or_replays_dirty_candidate(tmp_path):
    budget = EpisodeBudget(EpisodeBudgetPolicy())
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        executions_before = budget.code_executions
        restarts_before = env.kernel_restarts_total

        finalized = env.finalize()

        assert finalized.done
        assert finalized.final_status == "invalid_submission"
        assert budget.code_executions == executions_before
        assert env.kernel_restarts_total == restarts_before
        assert env.get_summary()["finalize_path"] == "research_no_repair"
    finally:
        env.close()


def test_m1a_research_finalize_does_not_submit_historical_incumbent(tmp_path):
    budget = EpisodeBudget(EpisodeBudgetPolicy())
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        incumbent = env.candidates.incumbent()
        assert incumbent is not None

        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "markdown",
                "source": "later exploratory revision",
            }
        )
        executions_before = budget.code_executions
        finalized = env.finalize()

        assert finalized.done
        assert finalized.final_status == "invalid_submission"
        assert env.candidates.latest() is None
        assert env.candidates.incumbent() == incumbent
        assert budget.code_executions == executions_before
        assert env.get_summary()["hidden_evaluations"] == 0
        assert env.get_summary()["finalize_path"] == "research_no_repair"
    finally:
        env.close()


def test_validation_query_control_keeps_feedback_labels_out_of_workspace_and_kernel(
    tmp_path,
):
    budget = _validation_query_budget()
    env = _make_env(tmp_path / "controlled", episode_budget=budget)
    legacy = _make_env(tmp_path / "legacy")
    try:
        controlled_val = pd.read_csv(tmp_path / "controlled" / "data" / "val.csv")
        legacy_val = pd.read_csv(tmp_path / "legacy" / "data" / "val.csv")

        assert "target" not in controlled_val.columns
        assert "target" in legacy_val.columns
        assert "target" in env.state.val.columns
        probe = env.step(
            Action.code_action("print(target_col in val_df.columns)")
        )
        assert "False" in probe.stdout
        assert budget.validation_queries == 0
        prompt = env._build_context_prompt()["task"]
        assert "Feedback-validation labels are host-only" in prompt
        assert "target column not present" in prompt
        inspection = env.step(
            {"type": "inspect_data", "stage": "data_schema_inspection"}
        )
        assert "target column not present" in inspection.stdout
        assert budget.validation_queries == 0
    finally:
        env.close()
        legacy.close()


@pytest.mark.parametrize("via_agent_submit", [False, True])
def test_m2b_protected_finalization_replays_host_incumbent_without_mutating_snapshot(
    tmp_path, via_agent_submit
):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    private_dir = tmp_path / "private"
    env = _make_env(
        tmp_path / "workspace",
        private_dir=private_dir,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        incumbent = env.candidates.incumbent()
        assert incumbent is not None
        snapshot_path = env.candidate_store.notebook_snapshot_path(incumbent)
        snapshot_before = snapshot_path.read_bytes()

        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "markdown",
                "source": "later exploratory revision that must not be replayed",
            }
        )
        assert env.candidates.latest() is None
        mutable_revision = env.notebook.revision

        if via_agent_submit:
            finalized = env.step(
                {"type": "submit", "stage": "submission", "model_var": "not_the_incumbent"}
            )
        else:
            finalized = env.finalize("not_the_incumbent")

        assert finalized is not None
        assert finalized.done and finalized.submitted
        assert finalized.final_status == "submitted_protected_replay"
        assert finalized.model_var == "model"
        assert env.notebook.revision == mutable_revision
        assert "later exploratory revision" in env.notebook.notebook.cells[-1].source
        assert snapshot_path.read_bytes() == snapshot_before
        assert env.candidates.all() == [incumbent]
        assert env.candidates.incumbent() == incumbent
        assert env.candidates.latest() is None
        assert env.candidates.is_submitted(incumbent.candidate_id)

        summary = env.get_summary()
        assert summary["valid_submit"] is True
        assert summary["hidden_evaluations"] == 1
        assert summary["final_status"] == "submitted_protected_replay"
        assert summary["finalize_path"] == "protected_incumbent_replay"
        assert summary["reproducibility_level"] == "protected_producing_revision_replay"
        assert summary["finalization_candidate_id"] == incumbent.candidate_id
        assert budget.phase == "finalization"
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 1
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 3

        final_artifact = env.finalization_store.verify_artifact(incumbent.candidate_id)
        assert final_artifact.source_candidate_hashes
        public_text = (env.workspace_dir / "notebook_events.json").read_text(
            encoding="utf-8"
        )
        assert str(private_dir) not in public_text
        assert str(final_artifact.artifact_dir) not in public_text
        assert "final_test_metric" not in public_text
    finally:
        env.close()


def test_m2b_protected_finalization_fails_terminal_without_incumbent(tmp_path):
    budget = _protected_budget()
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        finalized = env.finalize()

        assert finalized is not None
        assert finalized.done and finalized.submitted
        assert finalized.final_status == "no_incumbent_for_finalization"
        assert env.get_summary()["hidden_evaluations"] == 0
        assert budget.phase == "finalization"
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 0
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 0
        assert not (env.private_dir / "finalization").exists()
    finally:
        env.close()


def test_m2b_protected_finalization_rejects_corrupt_incumbent_before_replay(
    tmp_path,
):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        incumbent = env.candidates.incumbent()
        assert incumbent is not None
        with Path(incumbent.artifact_path).open("ab") as handle:
            handle.write(b"corruption")
        restarts_before = env.kernel_restarts_total

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "invalid_candidate_artifact"
        assert env.kernel_restarts_total == restarts_before
        assert env.get_summary()["hidden_evaluations"] == 0
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 0
        assert not (env.private_dir / "finalization").exists()
        public_text = (env.workspace_dir / "notebook_events.json").read_text(
            encoding="utf-8"
        )
        assert "checksum mismatch" not in public_text
        assert str(Path(incumbent.artifact_path).parent) not in public_text
    finally:
        env.close()


def test_m2b_protected_finalization_has_no_live_kernel_fallback_on_replay_failure(
    tmp_path,
):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    source = """
from pathlib import Path
from sklearn.dummy import DummyClassifier

marker = Path('protected-replay-once.marker')
if marker.exists():
    raise RuntimeError('snapshot may run only once')
marker.write_text('first clean run', encoding='utf-8')
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
model = DummyClassifier(strategy='constant', constant=0)
model.fit(X_train, y_train)
""".strip()
    try:
        env.step(Action.add_cell_action(source, cell_type="code", execute=False))
        clean = env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )
        assert "successfully" in clean.stdout
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        incumbent = env.candidates.incumbent()
        assert incumbent is not None

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "protected_replay_failed"
        assert env.get_summary()["hidden_evaluations"] == 0
        assert env.candidates.all() == [incumbent]
        assert not env.candidates.is_submitted(incumbent.candidate_id)
        assert not (env.private_dir / "finalization").exists()
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 1
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 0
    finally:
        env.close()


def test_m2b_protected_finalization_stops_before_overshooting_code_reserve(
    tmp_path,
):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step(
            Action.add_cell_action("helper_value = 1", cell_type="code", execute=False)
        )
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "finalization_reserve_exhausted"
        assert env.get_summary()["hidden_evaluations"] == 0
        snapshot = budget.snapshot()
        assert snapshot["finalization_usage"]["code_executions"] == 1
        assert snapshot["finalization_usage"]["tool_calls"] == 0
        assert budget.exhausted_reason == "finalization_max_code_executions"
        assert not (env.private_dir / "finalization").exists()
    finally:
        env.close()


def test_m2b_protected_finalization_rejects_validation_metric_drift(tmp_path):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        env.metric_fn = lambda y_true, y_pred: 0.25

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "protected_replay_validation_mismatch"
        assert env.get_summary()["hidden_evaluations"] == 0
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 1
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 1
        assert not (env.private_dir / "finalization").exists()
    finally:
        env.close()


def test_m2b_protected_finalization_stores_artifact_before_hidden_gate(
    tmp_path, monkeypatch
):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )

        def fail_write(*args, **kwargs):
            raise ProtectedFinalizationError("private artifact detail")

        monkeypatch.setattr(env.finalization_store, "write_artifact", fail_write)
        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "protected_finalization_artifact_failed"
        assert "private artifact detail" not in finalized.stderr
        assert env.get_summary()["hidden_evaluations"] == 0
        assert budget.snapshot()["finalization_usage"]["code_executions"] == 1
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 2
    finally:
        env.close()


def test_m2b_protected_hidden_failure_is_terminal_and_one_shot(tmp_path):
    budget = _protected_budget(code_reserve=1)
    env = _make_env(
        tmp_path,
        hidden_green=True,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            Action.add_cell_action(
                _strict_one_hot_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        incumbent = env.candidates.incumbent()
        assert incumbent is not None

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.done and finalized.submitted
        assert finalized.final_status == "hidden_submit_failed"
        assert env.get_summary()["hidden_evaluations"] == 1
        assert env.hidden_submit_fail_count == 1
        assert not env.candidates.is_submitted(incumbent.candidate_id)
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 3
        with pytest.raises(RuntimeError, match="already finalized"):
            env.step(
                {"type": "submit", "stage": "submission", "model_var": "model"}
            )
    finally:
        env.close()


def test_m2b_exploration_cutoff_transitions_into_untouched_reserve(tmp_path):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        exploration_tool_limit = (
            budget.policy.max_tool_calls
            - budget.finalization_reserve.max_tool_calls
        )
        while budget.tool_calls < exploration_tool_limit:
            budget.consume_tool_call("test_exploration")
        with pytest.raises(ExplorationBudgetExhausted):
            budget.consume_tool_call("test_exploration_overshoot")
        assert budget.exhausted_reason is None

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.final_status == "submitted_protected_replay"
        snapshot = budget.snapshot()
        assert snapshot["budget_phase"] == "finalization"
        assert snapshot["exploration_exhausted"] is True
        assert snapshot["finalization_usage"]["code_executions"] == 1
        assert snapshot["finalization_usage"]["tool_calls"] == 3
        assert budget.tool_calls == budget.policy.max_tool_calls
    finally:
        env.close()


def test_m4_no_improvement_stops_via_protected_incumbent_replay(tmp_path):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    policy = _stopping_policy(
        reserve_boundary_resources=(),
        no_improvement_patience=1,
    )
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
        stopping_policy=policy,
    )
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        first = env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        assert first.validation_metric == 0.5
        assert env.stopping_controller.decision is None

        second = env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        assert second.validation_metric == 0.5
        assert env.stopping_controller.decision.reason == "no_validation_improvement"

        stopped = env.apply_stopping_policy()

        assert stopped is not None and stopped.submitted
        assert stopped.final_status == "submitted_protected_replay"
        summary = env.get_summary()
        assert summary["stopping_reason"] == "no_validation_improvement"
        assert summary["stopping_transition"] == "finalize"
        assert summary["hidden_evaluations"] == 1
        assert budget.phase == "finalization"
    finally:
        env.close()


def test_m4_reserve_boundary_stops_before_exploration_can_borrow(tmp_path):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
        stopping_policy=_stopping_policy(),
    )
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        exploration_limit = (
            budget.policy.max_tool_calls
            - budget.finalization_reserve.max_tool_calls
        )
        while budget.tool_calls < exploration_limit:
            budget.consume_tool_call("test_exploration")

        stopped = env.apply_stopping_policy()

        assert stopped is not None and stopped.submitted
        assert stopped.final_status == "submitted_protected_replay"
        assert env.get_summary()["stopping_reason"] == "reserve_boundary_reached"
        assert budget.snapshot()["finalization_usage"]["tool_calls"] == 3
        assert budget.tool_calls == budget.policy.max_tool_calls
    finally:
        env.close()


def test_m4_unrecoverable_bundle_failure_terminates_without_hidden_gate(
    tmp_path,
    monkeypatch,
):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
        stopping_policy=_stopping_policy(),
    )
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})

        def fail_write(*_args, **_kwargs):
            raise CandidateBundleError("private storage detail")

        monkeypatch.setattr(env.candidate_store, "write_bundle", fail_write)
        stopped = env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )

        assert stopped.done and stopped.submitted
        assert stopped.final_status == "stopped_unrecoverable_contract_failure"
        summary = env.get_summary()
        assert summary["stopping_reason"] == "unrecoverable_contract_failure"
        assert summary["hidden_evaluations"] == 0
        assert budget.phase == "exploration"
        assert "private storage detail" not in stopped.stderr
    finally:
        env.close()


def test_research_tool_budget_is_enforced_before_overshoot(tmp_path):
    budget = EpisodeBudget(EpisodeBudgetPolicy(max_tool_calls=1))
    env = _make_env(tmp_path, episode_budget=budget, research_submission=True)
    try:
        env.step({"type": "inspect_data", "stage": "data_schema_inspection"})
        with pytest.raises(BudgetExhausted, match="max_tool_calls"):
            env.step({"type": "profile_data", "stage": "data_schema_inspection"})
        assert budget.tool_calls == 1
    finally:
        env.close()


def test_verified_candidate_registry_restores_as_historical_only(tmp_path):
    private_dir = tmp_path / "private"
    first = _make_env(tmp_path / "first", private_dir=private_dir)
    second = None
    try:
        first.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        first.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        first.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        original = first.candidates.incumbent()
        assert original is not None

        first.close()
        second = _make_env(
            tmp_path / "second",
            private_dir=private_dir,
            reset=False,
            research_submission=True,
        )
        restored = second.restore_candidate_registry()

        assert restored == [original]
        assert second.candidates.all() == [original]
        assert second.candidates.incumbent() == original
        assert second.candidates.latest_registered() == original
        assert second.candidates.latest() is None
        assert second._candidate_objects == {}
        assert second.hidden_evaluation_gate.attempted == 0
        blocked = second.submit_by_name("model")
        assert "restart_and_run_all" in blocked.stderr
        assert second.hidden_evaluation_gate.attempted == 0
        assert any(
            event.get("action") == "candidate_registry_restored"
            and event.get("current_candidate_id") is None
            for event in second.events
        )
    finally:
        first.close()
        if second is not None:
            second.close()


def test_research_submit_fails_closed_before_hidden_evaluation_on_bundle_corruption(
    tmp_path,
):
    env = _make_env(tmp_path, research_submission=True)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        candidate = env.candidates.latest()
        assert candidate is not None
        artifact_path = Path(candidate.artifact_path)
        with artifact_path.open("ab") as handle:
            handle.write(b"corruption")

        observation = env.step(
            {"type": "submit", "stage": "submission", "model_var": "model"}
        )

        assert observation.done
        assert observation.submitted
        assert observation.final_status == "invalid_candidate_artifact"
        assert "integrity verification" in observation.stderr
        assert str(artifact_path.parent) not in observation.stderr
        assert env.candidates.latest() is None
        summary = env.get_summary()
        assert summary["hidden_evaluations"] == 0
        assert summary["final_status"] == "invalid_candidate_artifact"
        assert summary["valid_submit"] is False
        public_events = (tmp_path / "notebook_events.json").read_text(encoding="utf-8")
        assert str(artifact_path.parent) not in public_events
        assert "checksum mismatch" not in public_events
    finally:
        env.close()


def test_candidate_bundle_write_failure_is_a_blocker_not_a_partial_registration(
    tmp_path,
    monkeypatch,
):
    env = _make_env(tmp_path, research_submission=True)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})

        def fail_write(*args, **kwargs):
            raise CandidateBundleError("private storage detail")

        monkeypatch.setattr(env.candidate_store, "write_bundle", fail_write)
        observation = env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )

        assert "could not be saved safely" in observation.stderr
        assert "private storage detail" not in observation.stderr
        assert env.candidates.all() == []
        assert env._candidate_objects == {}
        assert env.hidden_evaluation_gate.attempted == 0
        assert any(
            event.get("action") == "candidate_bundle_write_failed"
            for event in env.events
        )
    finally:
        env.close()


def test_kernel_visible_artifacts_do_not_expose_private_evaluation_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )

        before_submit = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _workspace_probe_source(),
                "execute": True,
            }
        )
        assert "PRIVATE_LEAK" not in before_submit.stdout
        assert "test.csv" not in before_submit.stdout

        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        after_validate = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _workspace_probe_source(),
                "execute": True,
            }
        )
        assert "PRIVATE_LEAK" not in after_validate.stdout
        assert "final_test_metric" not in after_validate.stdout
        assert "private_checklist_coverage" not in after_validate.stdout

        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert submit.submitted

        direct_probe = env.kernel.execute_cell(_workspace_probe_source())
        assert direct_probe.success
        assert "PRIVATE_LEAK" not in direct_probe.stdout
        assert "final_test_metric" not in direct_probe.stdout
        assert "private_checklist_coverage" not in direct_probe.stdout

        public_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in tmp_path.rglob("*.json")
        )
        assert "final_test_metric" not in public_text
        assert "private_checklist_coverage" not in public_text
        assert "submit_failure_type" not in public_text
        assert not (tmp_path / "artifacts").exists()

        summary = env.get_summary()
        private_dir = summary["private_episode_dir"]
        private_summary = json.loads(
            (Path(private_dir) / "episode_summary.json").read_text(encoding="utf-8")
        )
        assert private_summary["final_test_metric"] == 1.0
        assert private_summary["private_checklist_coverage"] >= 0.0
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["gym_with_checklist", "iterative_no_checklist"])
def test_notebook_step_budget_blocks_actions_after_exhaustion(tmp_path, mode):
    env = _make_env(tmp_path, mode=mode)
    try:
        first = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": "budget_value = 1\nprint('ran once')",
                "execute": True,
            }
        )
        assert first.done is False

        env.state.max_steps = env.state.step
        before_cells = len(env.notebook.list_cells())
        blocked = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": "budget_value = 2\nprint('should not run')",
                "execute": True,
            }
        )

        assert blocked.done
        assert "Step budget exhausted" in blocked.stderr
        assert len(env.notebook.list_cells()) == before_cells
        probe = env.kernel.execute_cell("print(budget_value)")
        assert "1" in probe.stdout
        assert "2" not in probe.stdout

        validate = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert "Step budget exhausted" in validate.stderr
        assert env.state.step == env.state.max_steps
    finally:
        env.close()


def test_submit_is_allowed_after_budget_exhaustion_for_validated_candidate(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.state.max_steps = 3
        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        validated = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert validated.done
        assert validated.validation_metric == 0.5

        submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert submit.submitted
        assert submit.done
        assert env.get_summary()["final_test_metric"] == 1.0
    finally:
        env.close()


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
def test_notebook_env_docker_backend_end_to_end(tmp_path):
    backend = ContainerJupyterKernelBackend()
    env = _make_env(tmp_path, mode="iterative_no_checklist")
    env.close()
    env = NotebookGymEnv(
        train=env.state.train,
        val=env.state.val,
        test=env.state.test,
        target_col=env.state.target_col,
        metric_fn=_accuracy,
        metric_name="accuracy",
        max_steps=20,
        workspace_dir=tmp_path,
        mode="iterative_no_checklist",
        backend=backend,
    )
    try:
        env.reset()
        bootstrap = env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": (
                    "print(train_df.shape)\n"
                    "print(val_df.shape)\n"
                    "print(target_col)\n"
                    "print('test_df' in globals())"
                ),
                "execute": True,
            }
        )
        assert "(4, 3)" in bootstrap.stdout
        assert "(2, 3)" in bootstrap.stdout
        assert "target" in bootstrap.stdout
        assert "False" in bootstrap.stdout

        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        clean = env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        assert "successfully" in clean.stdout
        validated = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})
        assert validated.validation_metric == 0.5
        submit = env.step({"type": "submit", "stage": "submission", "model_var": "model"})
        assert submit.submitted
        assert submit.test_metric is None
        assert env.get_summary()["final_test_metric"] == 1.0
    finally:
        env.close()


def _constant_model_source():
    return """
from sklearn.dummy import DummyClassifier

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
model = DummyClassifier(strategy='constant', constant=0)
model.fit(X_train, y_train)
""".strip()


def _strict_one_hot_source():
    return """
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='error'), ['color']),
], remainder='passthrough')
model = Pipeline([
    ('prep', preprocessor),
    ('clf', DummyClassifier(strategy='most_frequent')),
])
model.fit(X_train, y_train)
""".strip()


def _workspace_probe_source():
    return """
import json
from pathlib import Path

tokens = [
    "final" + "_test_metric",
    "private" + "_checklist_coverage",
    "submit" + "_failure_type",
]
hits = []
for path in sorted(Path(".").rglob("*")):
    if path.is_file():
        rel = path.as_posix()
        if rel.endswith(".csv") or rel.endswith(".json") or rel.endswith(".pkl"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except UnicodeDecodeError:
                text = ""
            except Exception as exc:
                text = f"ERROR:{type(exc).__name__}"
            if any(token in text for token in tokens):
                hits.append(f"PRIVATE_LEAK:{rel}:{text[:120]}")
            else:
                hits.append(rel)
print("\\n".join(hits))
""".strip()


def test_score_with_coercion_handles_label_encoding():
    from sklearn.metrics import f1_score

    from gym.notebook_env import _score_with_coercion

    def metric(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    y_true = pd.Series(["a", "b", "a", "c"])
    int_preds = [0, 1, 0, 2]  # LabelEncoded predictions against string labels
    assert _score_with_coercion(metric, y_true, int_preds) == 1.0


def test_finalize_submits_live_kernel_model_without_clean_run(tmp_path):
    env = _make_env(tmp_path)
    try:
        # Train a picklable sklearn candidate that predicts on raw rows, leaving
        # the notebook dirty (the agent never ran restart_and_run_all / validate
        # / submit). The 'color' string column is dropped inside the pipeline.
        cell = (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "X = train_df.drop(columns=[target_col])\n"
            "y = train_df[target_col]\n"
            "pre = ColumnTransformer([('num', 'passthrough', ['x'])], remainder='drop')\n"
            "model = Pipeline([('pre', pre), ('clf', DecisionTreeClassifier(random_state=0))])\n"
            "model.fit(X, y)\n"
        )
        env.step(Action.add_cell_action(cell, cell_type="code", execute=True))
        assert env.dirty_since_clean_run is True
        assert env.candidates.latest() is None

        observation = env.finalize()
        assert observation is not None
        assert observation.submitted is True

        summary = env.get_summary()
        assert summary.get("valid_submit") is True
        assert summary.get("final_test_metric") is not None
    finally:
        env.close()


def test_finalize_action_auto_discovers_nonstandard_candidate_name(tmp_path):
    env = _make_env(tmp_path)
    try:
        cell = (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.dummy import DummyClassifier\n"
            "X = train_df.drop(columns=[target_col])\n"
            "y = train_df[target_col]\n"
            "pre = ColumnTransformer([('num', 'passthrough', ['x'])], remainder='drop')\n"
            "my_best_pipe = Pipeline([('pre', pre), ('clf', DummyClassifier(strategy='constant', constant=0))])\n"
            "my_best_pipe.fit(X, y)\n"
        )
        env.step(Action.add_cell_action(cell, cell_type="code", execute=True))

        observation = env.step({"type": "finalize", "stage": "submission", "model_var": "auto"})

        assert observation.submitted is True
        assert env.get_summary()["valid_submit"] is True
        assert "my_best_pipe" in env.get_summary()["finalize_attempted_vars"]
    finally:
        env.close()


def test_model_check_warns_after_broken_code_cell(tmp_path):
    env = _make_env(tmp_path)
    try:
        broken = """
from sklearn.linear_model import LogisticRegression
X = pd.get_dummies(train_df.drop(columns=[target_col]))
y = train_df[target_col]
best_model = LogisticRegression(max_iter=200).fit(X, y)
""".strip()

        observation = env.step(Action.add_cell_action(broken, cell_type="code", execute=True))

        assert "[MODEL CHECK]" in observation.stderr
        assert "best_model" in observation.stderr
        assert "raw validation" in observation.stderr
        assert "ValueError" in observation.stderr or "could not" in observation.stderr

        repeated = env.step({"type": "run_cell", "stage": "validation_analysis", "cell_id": observation.cell_id})
        assert repeated.stderr.count("[MODEL CHECK]") <= 1

        private_dir = Path(env.get_summary()["private_episode_dir"])
        diag_text = (private_dir / "candidate_diagnostics_private.jsonl").read_text(encoding="utf-8")
        assert "best_model" in diag_text
    finally:
        env.close()


def test_validate_raw_error_includes_exception_details(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            Action.add_cell_action(
                """
from sklearn.linear_model import LogisticRegression
X = pd.get_dummies(train_df.drop(columns=[target_col]))
y = train_df[target_col]
model = LogisticRegression(max_iter=200).fit(X, y)
""".strip(),
                cell_type="code",
                execute=False,
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})

        observation = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        assert "[MODEL CHECK]" in observation.stderr
        assert "model" in observation.stderr
        assert "ValueError" in observation.stderr or "could not" in observation.stderr
        assert "hidden-test rows" in observation.stderr
    finally:
        env.close()


def test_validate_scoring_failure_returns_blocker_without_registration(tmp_path):
    env = _make_env(tmp_path)

    def bad_metric(y_true, y_pred):
        raise ValueError("labels are incompatible")

    env.metric_fn = bad_metric
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(),
                cell_type="code",
                execute=False,
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})

        before_failures = env.model_check_failure_count
        observation = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        assert observation.validation_metric is None
        assert "[MODEL CHECK]" in observation.stderr
        assert "validation metric could not be computed" in observation.stderr
        assert "ValueError" in observation.stderr
        assert env.candidates.latest() is None
        assert env.state.submitted is False
        assert env.get_summary()["final_test_metric"] is None
        assert env.model_check_failure_count == before_failures + 1

        diagnostic = env.candidate_diagnostics[-1]
        assert diagnostic["source"] == "validate"
        assert diagnostic["candidate_var"] == "model"
        assert diagnostic["raw_val_predict_ok"] is True
        assert diagnostic["prediction_length_ok"] is True
        assert diagnostic["prediction_nan_free"] is True
        assert diagnostic["validation_metric"] is None
        assert diagnostic["error_type"] == "ValueError"
    finally:
        env.close()


def test_validate_raw_failure_increments_model_check_count_once(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            Action.add_cell_action(
                """
from sklearn.linear_model import LogisticRegression
X = pd.get_dummies(train_df.drop(columns=[target_col]))
y = train_df[target_col]
model = LogisticRegression(max_iter=200).fit(X, y)
""".strip(),
                cell_type="code",
                execute=False,
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        before_failures = env.model_check_failure_count

        observation = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        assert "[MODEL CHECK]" in observation.stderr
        assert env.model_check_failure_count == before_failures + 1
        private_dir = Path(env.get_summary()["private_episode_dir"])
        rows = [
            json.loads(line)
            for line in (private_dir / "candidate_diagnostics_private.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        validate_rows = [
            row
            for row in rows
            if row["source"] == "validate" and row["candidate_var"] == "model"
        ]
        assert len(validate_rows) == 1
        assert validate_rows[0]["raw_val_predict_ok"] is False
    finally:
        env.close()


def test_cloudpickle_serializes_function_transformer_candidate(tmp_path):
    env = _make_env(tmp_path)
    try:
        source = """
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

X = train_df.drop(columns=[target_col])
y = train_df[target_col]
model = Pipeline([
    ('select_x', FunctionTransformer(lambda df: df[['x']], validate=False)),
    ('clf', DummyClassifier(strategy='constant', constant=0)),
])
model.fit(X, y)
""".strip()
        env.step(Action.add_cell_action(source, cell_type="code", execute=False))
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})

        observation = env.step({"type": "validate", "stage": "validation_analysis", "model_var": "model"})

        assert observation.validation_metric == 0.5
        assert observation.stderr == ""
    finally:
        env.close()


def test_inspect_and_profile_are_compact_and_do_not_include_hidden_test(tmp_path):
    env = _make_env(tmp_path, hidden_green=True)
    try:
        inspect_obs = env.step({"type": "inspect_data", "stage": "data_schema_inspection"})
        profile_obs = env.step({"type": "profile_data", "stage": "data_quality_inspection", "profile": "compact"})

        assert "[DATA INSPECTION]" in inspect_obs.stdout
        assert "[DATA PROFILE]" in profile_obs.stdout
        assert "green" not in inspect_obs.stdout
        assert "green" not in profile_obs.stdout
        assert "<html" not in profile_obs.stdout.lower()
        assert len(profile_obs.stdout) <= 6000

        private_dir = Path(env.get_summary()["private_episode_dir"])
        assert (private_dir / "data_inspection_private.json").exists()
        assert (private_dir / "data_profile_private.json").exists()
        assert not (tmp_path / "data_profile_private.json").exists()
    finally:
        env.close()


def test_ydata_profile_artifacts_remain_private(tmp_path, monkeypatch):
    def fake_ydata(train, target_col, private_dir, **kwargs):
        html_path = Path(private_dir) / "data_profile_ydata.html"
        json_path = Path(private_dir) / "data_profile_ydata.json"
        html_path.write_text("<html>private profile</html>", encoding="utf-8")
        json_path.write_text('{"private": true}', encoding="utf-8")
        return {
            "available": True,
            "success": True,
            "timed_out": False,
            "rows_profiled": len(train),
            "cols_profiled": train.shape[1],
            "html_path": str(html_path),
            "json_path": str(json_path),
            "summary": {
                "n_rows": len(train),
                "n_cols": train.shape[1],
                "missing_cells": 0,
                "duplicate_rows": 0,
                "variable_types": {"Numeric": 1},
            },
        }

    monkeypatch.setattr("gym.notebook_env.run_ydata_profile", fake_ydata)
    env = _make_env(tmp_path, hidden_green=True)
    try:
        observation = env.step({"type": "profile_data", "stage": "data_quality_inspection", "profile": "ydata"})

        private_dir = Path(env.get_summary()["private_episode_dir"])
        assert (private_dir / "data_profile_ydata.html").exists()
        assert (private_dir / "data_profile_ydata.json").exists()
        assert "<html" not in observation.stdout.lower()
        assert str(private_dir) not in observation.stdout
        assert "data_profile_ydata" not in observation.stdout

        public_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in tmp_path.rglob("*.json")
        )
        assert "data_profile_ydata" not in public_text
        assert str(private_dir) not in public_text
    finally:
        env.close()


def test_check_list_and_quick_validate_candidate_tools(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            Action.add_cell_action(
                """
from sklearn.dummy import DummyClassifier
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
alt_model = DummyClassifier(strategy='constant', constant=0)
alt_model.fit(X_train, y_train)
""".strip(),
                cell_type="code",
                execute=True,
            )
        )
        listed = env.step({"type": "list_candidates", "stage": "candidate_training"})
        checked = env.step({"type": "check_candidate", "stage": "validation_analysis", "model_var": "auto"})
        quick = env.step({"type": "quick_validate", "stage": "validation_analysis", "model_var": "auto"})

        assert "alt_model" in listed.stdout
        assert "alt_model" in checked.stdout
        assert quick.validation_metric == 0.5
        assert env.candidates.latest() is None
        assert "artifacts" not in listed.stdout
    finally:
        env.close()


def test_cleanlab_diagnose_disabled_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOVIBE_ENABLE_CLEANLAB", raising=False)
    env = _make_env(tmp_path)
    try:
        observation = env.step({"type": "cleanlab_diagnose", "stage": "validation_analysis", "model_var": "auto"})

        assert "[CLEANLAB DIAGNOSTIC]" in observation.stdout
        assert "disabled" in observation.stdout
    finally:
        env.close()


def test_cleanlab_diagnostics_artifacts_remain_private(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOVIBE_ENABLE_CLEANLAB", "1")
    cleanlab_pkg = types.ModuleType("cleanlab")
    cleanlab_filter = types.ModuleType("cleanlab.filter")
    cleanlab_filter.find_label_issues = lambda **kwargs: [0]
    cleanlab_pkg.filter = cleanlab_filter
    monkeypatch.setitem(sys.modules, "cleanlab", cleanlab_pkg)
    monkeypatch.setitem(sys.modules, "cleanlab.filter", cleanlab_filter)

    env = _make_env(tmp_path)
    try:
        source = """
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X = train_df.drop(columns=[target_col])
y = train_df[target_col]
model = Pipeline([
    ('prep', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['color']),
        ('num', 'passthrough', ['x']),
    ])),
    ('clf', LogisticRegression(max_iter=200)),
])
model.fit(X, y)
""".strip()
        env.step(Action.add_cell_action(source, cell_type="code", execute=True))

        observation = env.step({"type": "cleanlab_diagnose", "stage": "validation_analysis", "model_var": "model"})

        private_dir = Path(env.get_summary()["private_episode_dir"])
        assert (private_dir / "cleanlab_diagnostics_private.json").exists()
        assert (private_dir / "cleanlab_issues_private.csv").exists()
        assert "cleanlab_diagnostics_private" not in observation.stdout
        assert "cleanlab_issues_private" not in observation.stdout
        assert str(private_dir) not in observation.stdout

        public_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in tmp_path.rglob("*.json")
        )
        assert "cleanlab_diagnostics_private" not in public_text
        assert "cleanlab_issues_private" not in public_text
        assert str(private_dir) not in public_text
    finally:
        env.close()


def test_tune_hyperparameters_caps_and_injects_tuned_model(tmp_path):
    env = _make_env(tmp_path)
    try:
        source = """
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
model = Pipeline([
    ('pre', ColumnTransformer([('num', 'passthrough', ['x'])], remainder='drop')),
    ('clf', DecisionTreeClassifier(random_state=0)),
])
model.fit(X, y)
""".strip()
        env.step(Action.add_cell_action(source, cell_type="code", execute=True))

        observation = env.step(
            {
                "type": "tune_hyperparameters",
                "stage": "model_improvement",
                "model_var": "model",
                "search_space": {"clf__max_depth": {"type": "int", "low": 1, "high": 2}},
                "n_trials": 3,
                "timeout_sec": 10,
            }
        )

        assert "[TUNING]" in observation.stdout
        assert "new_model_var=tuned_model" in observation.stdout
        listed = env.step({"type": "list_candidates", "stage": "candidate_training"})
        assert "tuned_model" in listed.stdout
    finally:
        env.close()


def test_context_pack_preserves_model_check_and_finalization_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.state.max_steps = 5
        broken = """
from sklearn.linear_model import LogisticRegression
X = pd.get_dummies(train_df.drop(columns=[target_col]))
y = train_df[target_col]
best_model = LogisticRegression(max_iter=200).fit(X, y)
""".strip()
        env.step(Action.add_cell_action(broken, cell_type="code", execute=True))

        pack = env.build_context_pack()

        assert pack["budget_remaining"] == 4
        assert "best_model" in pack["candidate_vars_seen"]
        assert pack["model_check_failures"]
        assert "finalize" in pack["finalization_requirements"]
    finally:
        env.close()


def test_m3_context_pack_v1_is_public_deterministic_and_incumbent_complete(
    tmp_path,
):
    budget = _protected_budget(code_reserve=1, tool_reserve=3)
    env = _make_env(
        tmp_path,
        hidden_green=True,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step({"type": "restart_and_run_all", "stage": "reproducibility_check"})
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        incumbent = env.candidates.incumbent()
        assert incumbent is not None
        env.step(
            {
                "type": "run_cell",
                "stage": "validation_analysis",
                "cell_id": "missing-cell",
            }
        )
        env.candidate_diagnostics.append(
            {"error_message": "PRIVATE_DIAGNOSTIC_SENTINEL"}
        )
        env.private_summary["final_test_metric"] = "PRIVATE_SCORE_SENTINEL"

        first = env.build_context_pack_v1()
        second = env.build_context_pack_v1()
        text = json.dumps(first, ensure_ascii=False, sort_keys=True)

        assert first == second
        assert first["schema_version"] == "context-pack-v1"
        assert first["task_contract"]["target_column"] == "target"
        assert first["task_contract"]["metric_direction"] == "higher"
        assert first["incumbent_state"]["candidate_id"] == incumbent.candidate_id
        assert len(first["tested_hypotheses"]) == 1
        assert first["tested_hypotheses"][0]["outcome"] == "validated"
        assert first["tested_hypotheses"][0]["candidate_id"] == incumbent.candidate_id
        assert first["remaining_resources"]["episode_budget"]["phase"] == "exploration"
        assert first["remaining_resources"]["episode_budget"][
            "finalization_remaining"
        ]["tool_calls"] == 3
        assert "unknown_cell_id" in first["unresolved_errors"][-1]["feedback_keys"]
        assert "think" not in first["allowed_actions"]
        assert "artifact_path" not in text
        assert str(env.private_dir) not in text
        assert "PRIVATE_DIAGNOSTIC_SENTINEL" not in text
        assert "PRIVATE_SCORE_SENTINEL" not in text
        assert "green" not in text

        resolved = env.step(
            {
                "type": "run_cell",
                "stage": "validation_analysis",
                "cell_id": "cell_01",
            }
        )
        assert "unknown_cell_id" not in resolved.stderr
        after_resolution = env.build_context_pack_v1()
        assert all(
            error["action"] != "run_cell"
            for error in after_resolution["unresolved_errors"]
        )
    finally:
        env.close()


def test_validation_query_cap_is_pre_query_and_public_score_precision_is_frozen(
    tmp_path,
):
    budget = _validation_query_budget(
        max_queries=2,
        feedback_numeric_decimals=2,
    )
    env = _make_env(tmp_path, episode_budget=budget)
    env.metric_fn = lambda _y_true, _y_pred: 0.12345
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )

        first = env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        second = env.step(
            {
                "type": "quick_validate",
                "stage": "validation_analysis",
                "model_var": "model",
            }
        )

        assert first.validation_metric == 0.12
        assert second.validation_metric == 0.12
        assert "validation_accuracy=0.12" in first.stdout
        assert "validation_accuracy=0.12" in second.stdout
        assert env.candidates.incumbent().validation_metric == 0.12
        assert second.notebook_status["validation_queries_used"] == 2
        assert second.notebook_status["validation_queries_remaining"] == 0

        with pytest.raises(BudgetExhausted, match="max_validation_queries"):
            env.step(
                {
                    "type": "check_candidate",
                    "stage": "validation_analysis",
                    "model_var": "model",
                }
            )
        assert budget.validation_queries == 2
        assert budget.validation_queries_by_source == {
            "quick_validate": 1,
            "validate": 1,
        }
    finally:
        env.close()


def test_repeated_and_revision_reuse_validation_queries_are_all_charged(tmp_path):
    budget = _validation_query_budget(max_queries=3)
    env = _make_env(tmp_path, episode_budget=budget)
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        env.step(
            Action.add_cell_action("revision_marker = 1", cell_type="code", execute=False)
        )
        env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )

        assert budget.validation_queries == 3
        assert budget.validation_queries_by_source == {"validate": 3}
        assert len(env.candidates.all()) == 3
    finally:
        env.close()


def test_validation_like_tools_share_one_query_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOVIBE_ENABLE_CLEANLAB", "1")
    cleanlab_pkg = types.ModuleType("cleanlab")
    cleanlab_filter = types.ModuleType("cleanlab.filter")
    cleanlab_filter.find_label_issues = lambda **kwargs: [0]
    cleanlab_pkg.filter = cleanlab_filter
    monkeypatch.setitem(sys.modules, "cleanlab", cleanlab_pkg)
    monkeypatch.setitem(sys.modules, "cleanlab.filter", cleanlab_filter)

    budget = _validation_query_budget(max_queries=12)
    env = _make_env(tmp_path, episode_budget=budget)
    source = """
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
model = Pipeline([
    ('prep', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['color']),
        ('num', 'passthrough', ['x']),
    ])),
    ('clf', LogisticRegression(max_iter=200)),
])
model.fit(X, y)
""".strip()
    try:
        env.step(Action.add_cell_action(source, cell_type="code", execute=False))
        env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )
        env.step(
            {
                "type": "check_candidate",
                "stage": "validation_analysis",
                "model_var": "model",
            }
        )
        env.step(
            {
                "type": "quick_validate",
                "stage": "validation_analysis",
                "model_var": "model",
            }
        )
        env.step(
            {
                "type": "tune_hyperparameters",
                "stage": "model_improvement",
                "model_var": "model",
                "search_space": {
                    "clf__C": {"type": "float", "low": 0.5, "high": 1.5}
                },
                "n_trials": 2,
                "timeout_sec": 10,
            }
        )
        cleanlab = env.step(
            {
                "type": "cleanlab_diagnose",
                "stage": "validation_analysis",
                "model_var": "model",
            }
        )

        confidence_line = next(
            line for line in cleanlab.stdout.splitlines() if "confidence=" in line
        )
        confidence_text = confidence_line.split("confidence=", 1)[1]
        assert len(confidence_text.split(".", 1)[1]) == 3
        assert budget.validation_queries == 7
        assert budget.validation_queries_by_source == {
            "check_candidate": 1,
            "cleanlab_diagnose": 1,
            "quick_validate": 1,
            "tune_hyperparameters": 1,
            "tune_hyperparameters_preflight": 1,
            "tune_hyperparameters_trial": 2,
        }
    finally:
        env.close()


def test_protected_finalization_uses_only_reserved_validation_query(tmp_path):
    budget = _validation_query_budget(
        max_queries=2,
        finalization_reserve_queries=1,
        protected=True,
    )
    env = _make_env(
        tmp_path,
        episode_budget=budget,
        research_submission=True,
    )
    try:
        env.step(
            Action.add_cell_action(
                _constant_model_source(), cell_type="code", execute=False
            )
        )
        env.step(
            {"type": "restart_and_run_all", "stage": "reproducibility_check"}
        )
        env.step(
            {"type": "validate", "stage": "validation_analysis", "model_var": "model"}
        )
        with pytest.raises(
            ExplorationBudgetExhausted,
            match="exploration_max_validation_queries",
        ):
            env.step(
                {
                    "type": "quick_validate",
                    "stage": "validation_analysis",
                    "model_var": "model",
                }
            )

        finalized = env.finalize()

        assert finalized is not None
        assert finalized.submitted
        assert budget.validation_queries == 2
        assert budget.validation_queries_by_source == {
            "protected_finalization": 1,
            "validate": 1,
        }
        summary = env.get_summary()
        assert summary["validation_queries_total"] == 2
        assert summary["feedback_validation_metric"] == 0.5
        assert summary["hidden_minus_feedback_validation_metric"] == 0.5
        query_context = env.build_context_pack_v1()["remaining_resources"][
            "episode_budget"
        ]["validation_query_budget"]
        assert query_context["queries_used"] == 2
        assert query_context["global_remaining_queries"] == 0
    finally:
        env.close()
