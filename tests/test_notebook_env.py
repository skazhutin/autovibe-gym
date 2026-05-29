import json

import nbformat
import pandas as pd

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
