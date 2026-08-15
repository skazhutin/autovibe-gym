import os
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import research.submission as submission_module
from research.submission import (
    HiddenEvaluationAlreadyAttempted,
    HiddenEvaluationGate,
    prediction_backend_for_execution,
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


class _CrashModel:
    def predict(self, features):
        os._exit(23)


class _SlowModel:
    def predict(self, features):
        time.sleep(5)
        return np.zeros(len(features))


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


def test_common_validator_contains_prediction_process_crash():
    features, _ = _data()

    result = validate_submission_candidate(
        _CrashModel(), features, timeout_seconds=5
    )

    assert not result.valid
    assert result.raw_prediction_ok is False
    assert result.error_type == "PredictionProcessCrash"


def test_common_validator_times_out_prediction_process():
    features, _ = _data()

    result = validate_submission_candidate(
        _SlowModel(), features, timeout_seconds=0.1
    )

    assert not result.valid
    assert result.raw_prediction_ok is False
    assert result.error_type == "PredictionTimeout"


def test_docker_kernel_selects_secret_isolated_prediction_backend(monkeypatch):
    features, _ = _data()
    captured = {}

    def fake_docker(model_payload, raw_features, *, timeout_seconds):
        captured["model_payload"] = model_payload
        captured["rows"] = len(raw_features)
        captured["timeout_seconds"] = timeout_seconds
        return {"ok": True, "predictions": np.zeros(len(raw_features))}

    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "docker")
    monkeypatch.delenv("AUTOVIBE_SUBMISSION_PREDICT_BACKEND", raising=False)
    monkeypatch.setattr(submission_module, "_predict_docker", fake_docker)

    result = validate_submission_candidate(_NullModel(), features)

    assert result.valid
    assert captured["model_payload"]
    assert captured["rows"] == len(features)
    assert captured["timeout_seconds"] == 120.0


def test_execution_backend_selects_matching_prediction_isolation():
    assert prediction_backend_for_execution("docker") == "docker"
    assert prediction_backend_for_execution("DOCKER") == "docker"
    assert prediction_backend_for_execution("subprocess") == "process"
    assert prediction_backend_for_execution(None) == "process"


def test_explicit_prediction_backend_overrides_kernel_environment(monkeypatch):
    features, _ = _data()
    captured = {}

    def fake_docker(model_payload, raw_features, *, timeout_seconds):
        captured["rows"] = len(raw_features)
        return {"ok": True, "predictions": [0.0] * len(raw_features)}

    monkeypatch.setenv("AUTOVIBE_KERNEL_BACKEND", "local")
    monkeypatch.setattr(submission_module, "_predict_docker", fake_docker)

    result = validate_submission_candidate(
        _NullModel(), features, prediction_backend="docker"
    )

    assert result.valid
    assert captured["rows"] == len(features)


@pytest.mark.parametrize(
    "payload",
    [
        '{"ok": true, "predictions": [{"value": 1}]}',
        '{"ok": true, "predictions": [NaN]}',
        '{"ok": false, "error_type": "RuntimeError"}',
    ],
)
def test_docker_result_decoder_rejects_non_scalar_or_incomplete_payloads(payload):
    result = submission_module._decode_docker_result(payload)

    assert not result["ok"]
    assert result["error_type"] == "PredictionContractError"


def test_docker_result_decoder_accepts_typed_json_scalar_vector():
    result = submission_module._decode_docker_result(
        '{"ok": true, "predictions": [1, 2.5, "yes", true, null]}'
    )

    assert result == {"ok": True, "predictions": [1, 2.5, "yes", True, None]}
    assert "result.json" in submission_module.DOCKER_PREDICTION_WORKER
    assert "cloudpickle.dumps(result)" not in submission_module.DOCKER_PREDICTION_WORKER


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
