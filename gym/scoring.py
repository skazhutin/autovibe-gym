"""Shared metric scoring helpers used by the runners and the notebook env."""
from __future__ import annotations

from typing import Any


def score_with_coercion(metric_fn: Any, y_true: Any, preds: Any) -> float:
    """Score predictions, tolerating label-encoding dtype mismatches.

    Agents routinely LabelEncode the target and return integer predictions while
    the held-out split keeps the original (e.g. string) labels. Try directly,
    then cast predictions to the target dtype, then map integer predictions
    through the sorted class labels. ``y_true`` is expected to be a pandas
    Series (it exposes ``.dtype`` and ``.unique()``).
    """
    try:
        return float(metric_fn(y_true, preds))
    except (ValueError, TypeError):
        import numpy as np
        import pandas as pd

        try:
            preds_cast = pd.Series(preds).astype(y_true.dtype).values
            return float(metric_fn(y_true, preds_cast))
        except Exception:
            classes = sorted(pd.Series(y_true).unique())
            preds_mapped = np.array([classes[int(p)] for p in preds])
            return float(metric_fn(y_true, preds_mapped))
