"""Shared metric scoring helpers used by the runners and the notebook env."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_with_coercion(metric_fn: Any, y_true: Any, preds: Any) -> float:
    """Score predictions, tolerating label-encoding dtype mismatches.

    Agents routinely LabelEncode the target and return integer predictions while
    the held-out split keeps the original (e.g. string) labels. Map that case
    before calling metrics that may accept mixed label dtypes but score them
    incorrectly, then retain the older cast/mapping fallbacks. ``y_true`` is
    expected to be a pandas Series.
    """
    target = pd.Series(y_true)
    predictions = pd.Series(preds)
    target_is_non_numeric = not (
        pd.api.types.is_numeric_dtype(target.dtype)
        or pd.api.types.is_bool_dtype(target.dtype)
    )
    if target_is_non_numeric and _integer_encoded(predictions):
        try:
            classes = sorted(target.dropna().unique())
            encoded = [int(value) for value in predictions]
            if all(0 <= value < len(classes) for value in encoded):
                mapped = np.array([classes[value] for value in encoded])
                return float(metric_fn(y_true, mapped))
        except (TypeError, ValueError, IndexError):
            pass

    try:
        return float(metric_fn(y_true, preds))
    except (ValueError, TypeError):
        try:
            preds_cast = pd.Series(preds).astype(y_true.dtype).values
            return float(metric_fn(y_true, preds_cast))
        except Exception:
            classes = sorted(pd.Series(y_true).unique())
            preds_mapped = np.array([classes[int(p)] for p in preds])
            return float(metric_fn(y_true, preds_mapped))


def _integer_encoded(values: Any) -> bool:
    for value in values:
        if isinstance(value, (bool, np.bool_)) or pd.isna(value):
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric) or not numeric.is_integer():
            return False
    return True
