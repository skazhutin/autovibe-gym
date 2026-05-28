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
    assert "elapsed_seconds" in summary
