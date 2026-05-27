"""Tests for Checklist — no LLM or server required."""
import pytest
from gym.checklist import Checklist


@pytest.fixture
def cl():
    return Checklist(target_col="quality")


def test_initial_coverage_zero(cl):
    assert cl.coverage() == 0.0


def test_eda_check(cl):
    cl.evaluate(code="print(df.describe())", stdout="", namespace={}, history=[])
    assert cl.items["eda"].passed


def test_missing_values_check(cl):
    cl.evaluate(code="df.isnull().sum()", stdout="", namespace={}, history=[])
    assert cl.items["missing_values"].passed


def test_duplicates_check(cl):
    cl.evaluate(code="df.drop_duplicates(inplace=True)", stdout="", namespace={}, history=[])
    assert cl.items["duplicates"].passed


def test_target_leak_check_drop(cl):
    cl.evaluate(code="X = df.drop('quality', axis=1)", stdout="", namespace={}, history=[])
    assert cl.items["target_leak"].passed


def test_target_leak_check_X_train(cl):
    cl.evaluate(code="X_train = train_df.drop('quality', axis=1)", stdout="", namespace={}, history=[])
    assert cl.items["target_leak"].passed


def test_train_val_split_check(cl):
    cl.evaluate(code="score = model.score(val_df[features], y_val)", stdout="", namespace={}, history=[])
    assert cl.items["train_val_split"].passed


def test_feature_engineering_check(cl):
    cl.evaluate(code="X['log_feat'] = np.log(X['feat'] + 1)", stdout="", namespace={}, history=[])
    assert cl.items["feature_engineering"].passed


def test_model_selection_requires_two_models(cl):
    from gym.env import StepResult

    step1 = StepResult(action="code", step=1, budget_remaining=9,
                       code="m = RandomForestClassifier()")
    # Only one model seen — should not pass
    cl.evaluate(code="", stdout="", namespace={}, history=[step1])
    assert not cl.items["model_selection"].passed

    step2 = StepResult(action="code", step=2, budget_remaining=8,
                       code="m2 = LogisticRegression()")
    cl.evaluate(code="", stdout="", namespace={}, history=[step1, step2])
    assert cl.items["model_selection"].passed


def test_hyperparameter_tuning_check(cl):
    cl.evaluate(code="gs = GridSearchCV(model, param_grid)", stdout="", namespace={}, history=[])
    assert cl.items["hyperparameter_tuning"].passed


def test_full_coverage(cl):
    from gym.env import StepResult

    code = """
df.describe()
df.isnull().sum()
df.drop_duplicates()
X = df.drop('quality', axis=1)
X['feat2'] = np.log(X['feat'] + 1)
val_df
gs = GridSearchCV(model, {})
"""
    h1 = StepResult(action="code", step=1, budget_remaining=9,
                    code="m = RandomForestClassifier()")
    h2 = StepResult(action="code", step=2, budget_remaining=8,
                    code="m2 = LogisticRegression()")

    cl.evaluate(code=code, stdout="", namespace={}, history=[h1, h2])
    assert cl.coverage() == 1.0


def test_hints_only_for_unpassed(cl):
    hints = cl.evaluate(code="print(df.describe())", stdout="", namespace={}, history=[])
    keys = [h for h in cl.items if not cl.items[h].passed]
    # All pending items should have hints
    assert len(hints) == len(keys)
    assert "eda" not in [cl.items[k].key for k in keys]
