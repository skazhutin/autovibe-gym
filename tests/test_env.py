"""Tests for GymEnv — no LLM required."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from gym.env import GymEnv


def _make_env(max_steps=5):
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "quality": np.random.randint(0, 3, n),
    })
    train = df.iloc[:60].reset_index(drop=True)
    val   = df.iloc[60:80].reset_index(drop=True)
    test  = df.iloc[80:].reset_index(drop=True)

    metric_fn = lambda y, p: f1_score(y, p, average="weighted", zero_division=0)

    return GymEnv(
        train=train, val=val, test=test,
        target_col="quality",
        metric_fn=metric_fn,
        metric_name="f1_weighted",
        max_steps=max_steps,
    )


def _make_categorical_env():
    train = pd.DataFrame({
        "color": ["red", "blue", "red", "blue", "red", "blue"],
        "value": [1, 2, 3, 4, 5, 6],
        "target": [0, 1, 0, 1, 0, 1],
    })
    val = pd.DataFrame({
        "color": ["red", "blue"],
        "value": [7, 8],
        "target": [0, 1],
    })
    test = pd.DataFrame({
        "color": ["red", "blue"],
        "value": [9, 10],
        "target": [0, 1],
    })
    metric_fn = lambda y, p: f1_score(y, p, average="weighted", zero_division=0)
    return GymEnv(
        train=train,
        val=val,
        test=test,
        target_col="target",
        metric_fn=metric_fn,
        metric_name="f1_weighted",
        max_steps=5,
    )


def test_reset_returns_context(tmp_path):
    env = _make_env()
    ctx = env.reset()
    assert "task" in ctx
    assert "quality" in ctx["task"]


def test_step_returns_result(tmp_path):
    env = _make_env()
    env.reset()
    env.state.namespace = {
        "train_df": env.state.train.copy(),
        "val_df": env.state.val.copy(),
        "target_col": "quality",
    }
    result = env.step("x = 1 + 1\nprint(x)")
    assert result.step == 1
    assert "2" in result.stdout
    assert isinstance(result.hints, list)
    assert 0.0 <= result.checklist_coverage <= 1.0
    assert len(env.state.cell_history) == 1
    assert env.state.cell_history.last().code == "x = 1 + 1\nprint(x)"


def test_budget_decrements(tmp_path):
    env = _make_env(max_steps=3)
    env.reset()
    env.state.namespace = {"train_df": env.state.train.copy(), "val_df": env.state.val.copy(), "target_col": "quality"}
    assert env.budget_remaining() == 3
    env.step("pass")
    assert env.budget_remaining() == 2


def test_submit_scores_test_set():
    env = _make_env()
    env.reset()

    X_train = env.state.train.drop(columns=["quality"])
    y_train = env.state.train["quality"]
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    score = env.submit(model)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert len(env.state.cell_history) == 1
    assert env.state.cell_history.last().submitted is True


def test_double_submit_raises():
    env = _make_env()
    env.reset()
    model = DummyClassifier()
    model.fit(
        env.state.train.drop(columns=["quality"]),
        env.state.train["quality"],
    )
    env.submit(model)
    with pytest.raises(RuntimeError, match="Already submitted"):
        env.submit(model)


def test_step_after_submit_raises():
    env = _make_env()
    env.reset()
    model = DummyClassifier()
    model.fit(
        env.state.train.drop(columns=["quality"]),
        env.state.train["quality"],
    )
    env.submit(model)
    with pytest.raises(RuntimeError, match="finalized"):
        env.step("pass")


def test_done_flag_at_max_steps():
    env = _make_env(max_steps=2)
    env.reset()
    env.state.namespace = {"train_df": env.state.train.copy(), "val_df": env.state.val.copy(), "target_col": "quality"}
    env.step("pass")
    result = env.step("pass")
    assert result.done is True


def test_get_summary_shape():
    env = _make_env()
    env.reset()
    env.state.namespace = {"train_df": env.state.train.copy(), "val_df": env.state.val.copy(), "target_col": "quality"}
    env.step("x = 1")
    summary = env.get_summary()
    assert "steps_used" in summary
    assert "checklist_coverage" in summary
    assert "errors_count" in summary
    assert "cells_used" in summary
    assert "elapsed_seconds" in summary


def test_reset_clears_cell_history():
    env = _make_env()
    env.reset()
    env.step("print('cell')")
    assert len(env.state.cell_history) == 1

    env.reset()

    assert len(env.state.cell_history) == 0


def test_manual_preprocessing_model_gets_raw_validation_hint():
    env = _make_categorical_env()
    env.reset()

    result = env.step({
        "type": "code",
        "code": """
from sklearn.linear_model import LogisticRegression

X_train = pd.get_dummies(train_df.drop(columns=[target_col]))
y_train = train_df[target_col]
best_model = LogisticRegression(max_iter=200)
best_model.fit(X_train, y_train)
""".strip(),
    })

    joined_hints = "\n".join(result.hints)
    assert "[MODEL CHECK]" in joined_hints
    assert "raw validation" in joined_hints


def test_pipeline_model_passes_raw_validation_check():
    env = _make_categorical_env()
    env.reset()

    result = env.step({
        "type": "code",
        "code": """
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(exclude=["number"]).columns
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])
best_model = Pipeline([
    ("prep", preprocessor),
    ("clf", DummyClassifier(strategy="most_frequent")),
])
best_model.fit(X_train, y_train)
""".strip(),
    })

    joined_hints = "\n".join(result.hints)
    assert "[MODEL CHECK]" not in joined_hints


def test_submit_with_raw_validation_failure_does_not_finalize_env():
    env = _make_categorical_env()
    env.reset()
    env.step({
        "type": "code",
        "code": """
from sklearn.linear_model import LogisticRegression

X_train = pd.get_dummies(train_df.drop(columns=[target_col]))
y_train = train_df[target_col]
best_model = LogisticRegression(max_iter=200)
best_model.fit(X_train, y_train)
""".strip(),
    })

    result = env.step({"type": "submit", "model_var": "best_model"})

    assert result.submitted is False
    assert env.state.submitted is False
    assert "raw validation" in result.stderr


def test_hidden_test_failure_closes_without_leaking_hidden_values():
    train = pd.DataFrame({
        "color": ["red", "blue", "red", "blue"],
        "target": [0, 1, 0, 1],
    })
    val = pd.DataFrame({
        "color": ["red", "blue"],
        "target": [0, 1],
    })
    test = pd.DataFrame({
        "color": ["green"],
        "target": [1],
    })
    metric_fn = lambda y, p: f1_score(y, p, average="weighted", zero_division=0)
    env = GymEnv(
        train=train,
        val=val,
        test=test,
        target_col="target",
        metric_fn=metric_fn,
        metric_name="f1_weighted",
        max_steps=5,
    )
    env.reset()
    train_result = env.step({
        "type": "code",
        "code": """
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="error"), ["color"]),
])
best_model = Pipeline([
    ("prep", preprocessor),
    ("clf", DummyClassifier(strategy="most_frequent")),
])
best_model.fit(X_train, y_train)
""".strip(),
    })
    assert "[MODEL CHECK]" not in "\n".join(train_result.hints)

    result = env.step({"type": "submit", "model_var": "best_model"})

    assert result.submitted is True
    assert result.done is True
    assert result.test_metric is None
    assert env.state.submitted is True
    assert "hidden test split" in result.stderr
    assert "green" not in result.stderr
    assert env.get_summary()["submitted"] is True
