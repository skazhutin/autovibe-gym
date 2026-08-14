import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from research.submission import (
    HiddenEvaluationAlreadyAttempted,
    HiddenEvaluationGate,
    validate_submission_candidate,
)


class _UnfittedModel:
    def predict(self, features):
        raise RuntimeError("not fitted")

    def fit(self, features, target):
        raise AssertionError("validator must never fit")


class _NullModel:
    def predict(self, features):
        return np.full(len(features), np.nan)


def _data():
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0, 0, 1, 1])
    return features, target


def test_common_validator_accepts_fitted_raw_row_candidate():
    features, target = _data()
    model = LogisticRegression().fit(features, target)

    result = validate_submission_candidate(model, features)

    assert result.valid
    assert result.serializable
    assert result.prediction_length_ok
    assert result.prediction_nan_free


def test_common_validator_never_repairs_or_fits_candidate():
    features, _ = _data()

    result = validate_submission_candidate(_UnfittedModel(), features)

    assert not result.valid
    assert result.raw_prediction_ok is False
    assert result.error_type == "RuntimeError"


def test_common_validator_rejects_null_predictions():
    features, _ = _data()

    result = validate_submission_candidate(_NullModel(), features)

    assert not result.valid
    assert result.prediction_nan_free is False


def test_hidden_evaluation_gate_allows_exactly_one_attempt_even_after_failure():
    features, target = _data()
    gate = HiddenEvaluationGate()

    with pytest.raises(RuntimeError, match="not fitted"):
        gate.evaluate(_UnfittedModel(), features, target, lambda y, p: 0.0)
    with pytest.raises(HiddenEvaluationAlreadyAttempted):
        gate.evaluate(_NullModel(), features, target, lambda y, p: 0.0)


def test_hidden_evaluation_gate_returns_metric_without_exposing_feedback():
    features, target = _data()
    model = LogisticRegression().fit(features, target)
    gate = HiddenEvaluationGate()

    score = gate.evaluate(model, features, target, lambda y, p: float((y == p).mean()))

    assert score == 1.0
    assert gate.attempted
