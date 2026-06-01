import json
import subprocess
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from gym.jupyter_kernel import ContainerJupyterKernelBackend
from gym.notebook_env import NotebookGymEnv
from gym.protocol import Action


def _accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def _make_env(tmp_path, *, mode="gym_with_checklist", hidden_green=False):
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
        result = env.step({"type": "code", "code": "value = 41\nprint(value + 1)"})

        assert result.cell_id == "cell_01"
        assert "42" in result.stdout
        assert env.notebook.list_cells()[0]["cell_type"] == "code"
    finally:
        env.close()


def test_restart_and_run_all_clears_interactive_stale_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        stale = env.step(
            {"type": "add_cell", "cell_type": "code", "source": "stale = 123", "execute": True}
        )
        env.step({"type": "delete_cell", "cell_id": stale.cell_id})
        check = env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": "print('stale' in globals())",
                "execute": True,
            }
        )
        assert "True" in check.stdout

        clean = env.step({"type": "restart_and_run_all"})
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
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )

        rejected = env.step({"type": "validate", "model_var": "model"})
        assert "restart_and_run_all" in rejected.stderr

        clean = env.step({"type": "restart_and_run_all"})
        assert "successfully" in clean.stdout

        validated = env.step({"type": "validate", "model_var": "model"})
        assert validated.validation_metric == 0.5

        env.step(
            {
                "type": "add_cell",
                "cell_type": "markdown",
                "source": "changed after validation",
            }
        )
        dirty_submit = env.step({"type": "submit", "model_var": "model"})
        assert "restart_and_run_all" in dirty_submit.stderr
    finally:
        env.close()


def test_successful_submit_hides_score_from_agent_context_and_keeps_private_summary(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all"})
        env.step({"type": "validate", "model_var": "model"})

        submit = env.step({"type": "submit", "model_var": "model"})
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
    env = _make_env(tmp_path, hidden_green=True)
    try:
        env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": _strict_one_hot_source(),
                "execute": True,
            }
        )
        env.step({"type": "restart_and_run_all"})
        validated = env.step({"type": "validate", "model_var": "model"})
        assert not validated.stderr

        submit = env.step({"type": "submit", "model_var": "model"})
        assert submit.submitted
        assert "hidden test split" in submit.stderr
        assert "green" not in submit.stderr
        assert env.get_summary()["submit_failure_type"]
    finally:
        env.close()


def test_kernel_visible_artifacts_do_not_expose_private_evaluation_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": True,
            }
        )

        before_submit = env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": _workspace_probe_source(),
                "execute": True,
            }
        )
        assert "PRIVATE_LEAK" not in before_submit.stdout
        assert "test.csv" not in before_submit.stdout

        env.step({"type": "restart_and_run_all"})
        env.step({"type": "validate", "model_var": "model"})

        after_validate = env.step(
            {
                "type": "add_cell",
                "cell_type": "code",
                "source": _workspace_probe_source(),
                "execute": True,
            }
        )
        assert "PRIVATE_LEAK" not in after_validate.stdout
        assert "final_test_metric" not in after_validate.stdout
        assert "private_checklist_coverage" not in after_validate.stdout

        env.step({"type": "restart_and_run_all"})
        env.step({"type": "validate", "model_var": "model"})
        submit = env.step({"type": "submit", "model_var": "model"})
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

        validate = env.step({"type": "validate", "model_var": "model"})
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
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        env.step({"type": "restart_and_run_all"})
        validated = env.step({"type": "validate", "model_var": "model"})
        assert validated.done
        assert validated.validation_metric == 0.5

        submit = env.step({"type": "submit", "model_var": "model"})
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
                "cell_type": "code",
                "source": _constant_model_source(),
                "execute": False,
            }
        )
        clean = env.step({"type": "restart_and_run_all"})
        assert "successfully" in clean.stdout
        validated = env.step({"type": "validate", "model_var": "model"})
        assert validated.validation_metric == 0.5
        submit = env.step({"type": "submit", "model_var": "model"})
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
