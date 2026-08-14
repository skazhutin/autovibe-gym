"""Common no-repair submission validation and one-shot hidden evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cloudpickle
import numpy as np
import pandas as pd

@dataclass
class SubmissionValidation:
    valid: bool
    exists: bool
    has_predict: bool
    serializable: bool | None = None
    raw_prediction_ok: bool | None = None
    prediction_length_ok: bool | None = None
    prediction_nan_free: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    predictions: Any = field(default=None, repr=False, compare=False)


def validate_submission_candidate(model: Any, raw_features: pd.DataFrame) -> SubmissionValidation:
    if model is None:
        return SubmissionValidation(
            valid=False,
            exists=False,
            has_predict=False,
            error_type="CandidateMissing",
            error_message="No candidate object was provided.",
        )
    has_predict = callable(getattr(model, "predict", None))
    if not has_predict:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=False,
            error_type="ModelInterfaceError",
            error_message="Candidate does not expose predict(X).",
        )
    try:
        cloudpickle.dumps(model)
    except Exception as error:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=True,
            serializable=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    try:
        predictions = model.predict(raw_features)
    except Exception as error:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=True,
            serializable=True,
            raw_prediction_ok=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    length_ok = _prediction_length(predictions) == len(raw_features)
    nan_free = _predictions_nan_free(predictions)
    return SubmissionValidation(
        valid=bool(length_ok and nan_free),
        exists=True,
        has_predict=True,
        serializable=True,
        raw_prediction_ok=True,
        prediction_length_ok=length_ok,
        prediction_nan_free=nan_free,
        error_type=None if length_ok and nan_free else "PredictionContractError",
        error_message=None if length_ok and nan_free else "Predictions must be one non-null value per row.",
        predictions=predictions,
    )


class HiddenEvaluationAlreadyAttempted(RuntimeError):
    pass


class HiddenEvaluationGate:
    def __init__(self):
        self.attempted = False

    def evaluate(
        self,
        model: Any,
        raw_features: pd.DataFrame,
        target: pd.Series,
        metric_fn: Callable,
    ) -> float:
        from gym.scoring import score_with_coercion

        if self.attempted:
            raise HiddenEvaluationAlreadyAttempted(
                "Hidden evaluation is limited to one attempt per agent outcome."
            )
        self.attempted = True
        predictions = model.predict(raw_features)
        if _prediction_length(predictions) != len(raw_features):
            raise ValueError("Hidden predictions have the wrong length.")
        if not _predictions_nan_free(predictions):
            raise ValueError("Hidden predictions contain null values.")
        return score_with_coercion(metric_fn, target, predictions)


def _prediction_length(predictions: Any) -> int:
    try:
        return len(predictions)
    except TypeError:
        return -1


def _predictions_nan_free(predictions: Any) -> bool:
    try:
        values = np.asarray(predictions)
        if values.ndim == 0:
            return False
        return not bool(pd.isna(values).any())
    except Exception:
        return False
