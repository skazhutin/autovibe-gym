import json
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from gym.jupyter_kernel import ContainerJupyterKernelBackend
from gym.notebook_env import NotebookGymEnv
from gym.protocol import Action


def _accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def _make_env(tmp_path, *, mode="directive_gym", hidden_green=False, enable_thoughts=False):
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
        mode=mode,
        enable_thoughts=enable_thoughts,
    )
    env.reset()
    return env


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

        env.step(
            {
                "type": "add_cell",
                "stage": "feature_pipeline_building",
                "cell_type": "markdown",
                "source": "changed after validation",
            }
        )
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


@pytest.mark.parametrize("mode", ["directive_gym", "free_gym"])
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
    env = _make_env(tmp_path, mode="free_gym")
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
        mode="free_gym",
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
