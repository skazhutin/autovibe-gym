from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_compact_profile(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
    metric_name: str | None = None,
    max_rows_sample: int = 3,
    max_columns: int = 60,
) -> dict[str, Any]:
    """Build a compact train/validation-only profile for prompts and tools."""
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    train_features = _drop_target(train, target_col)
    val_features = _drop_target(val, target_col)
    visible_columns = list(train.columns[:max_columns])
    truncated_columns = max(len(train.columns) - len(visible_columns), 0)

    profile: dict[str, Any] = {
        "target_col": target_col,
        "metric": metric_name,
        "train_shape": tuple(train.shape),
        "val_shape": tuple(val.shape),
        "columns": visible_columns,
        "columns_truncated": truncated_columns,
        "dtype_counts": _dtype_counts(train),
        "target_distribution_train": _target_distribution(train, target_col),
        "target_distribution_val": _target_distribution(val, target_col),
        "missing_values_top": _missing_top(train_features),
        "missing_values_val_top": _missing_top(val_features),
        "constant_columns": _constant_columns(train_features),
        "quasi_constant_columns": _quasi_constant_columns(train_features),
        "duplicate_rows_train": int(train.duplicated().sum()),
        "object_or_categorical_columns": _categorical_columns(train_features),
        "high_cardinality_columns": _high_cardinality_columns(train_features),
        "id_like_or_suspicious_columns": _id_like_columns(train_features),
        "possible_datetime_columns": _possible_datetime_columns(train_features),
        "possible_text_columns": _possible_text_columns(train_features),
        "numeric_summary_top": _numeric_summary(train_features),
        "categorical_summary_top": _categorical_summary(train_features),
        "train_val_schema_diff": _schema_diff(train_features, val_features),
        "unseen_categories_in_val": _unseen_categories(train_features, val_features),
        "train_val_missingness_shift": _missingness_shift(train_features, val_features),
        "train_val_numeric_shift": _numeric_shift(train_features, val_features),
        "target_association_warnings": _target_associations(train, target_col),
        "sample_rows_train": _sample_rows(train, max_rows_sample),
        "suggested_next_steps": [],
    }
    profile["class_imbalance_warning"] = _class_imbalance_warning(profile)
    profile["modelling_implications"] = _modelling_implications(profile)
    profile["suggested_next_steps"] = _suggest_next_steps(profile)
    return _json_safe(profile)


def format_profile_for_agent(profile: dict[str, Any], max_chars: int = 5000) -> str:
    """Format a compact profile without exposing private artifact paths."""
    lines: list[str] = [
        "[DATA PROFILE]",
        f"target_col={profile.get('target_col')}",
        f"metric={profile.get('metric')}",
        f"train_shape={profile.get('train_shape')}",
        f"val_shape={profile.get('val_shape')}",
        "",
        "dtype_counts:",
        _format_value(profile.get("dtype_counts")),
        "",
        "target_distribution_train:",
        _format_value(profile.get("target_distribution_train")),
        "",
        "target_distribution_val:",
        _format_value(profile.get("target_distribution_val")),
        "",
        "missing_values_top:",
        _format_value(profile.get("missing_values_top")),
        "",
        "constant_columns:",
        _format_value(profile.get("constant_columns")),
        "",
        "quasi_constant_columns:",
        _format_value(profile.get("quasi_constant_columns")),
        "",
        f"duplicate_rows_train={profile.get('duplicate_rows_train')}",
        "",
        "object_or_categorical_columns:",
        _format_value(profile.get("object_or_categorical_columns")),
        "",
        "high_cardinality_columns:",
        _format_value(profile.get("high_cardinality_columns")),
        "",
        "id_like_or_suspicious_columns:",
        _format_value(profile.get("id_like_or_suspicious_columns")),
        "",
        "possible_datetime_columns:",
        _format_value(profile.get("possible_datetime_columns")),
        "",
        "possible_text_columns:",
        _format_value(profile.get("possible_text_columns")),
        "",
        "numeric_summary_top:",
        _format_value(profile.get("numeric_summary_top")),
        "",
        "train_val_schema_diff:",
        _format_value(profile.get("train_val_schema_diff")),
        "",
        "unseen_categories_in_val:",
        _format_value(profile.get("unseen_categories_in_val")),
        "",
        "train_val_numeric_shift:",
        _format_value(profile.get("train_val_numeric_shift")),
        "",
        "target_association_warnings:",
        _format_value(profile.get("target_association_warnings")),
        "",
        "sample_rows_train:",
        _format_value(profile.get("sample_rows_train")),
        "",
        "Suggested next steps:",
    ]
    lines.extend(f"- {item}" for item in profile.get("suggested_next_steps", []))
    return _clip("\n".join(lines), max_chars)


def build_dataset_card(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
    metric_name: str,
    max_chars: int = 5000,
) -> str:
    profile = build_compact_profile(
        train=train,
        val=val,
        target_col=target_col,
        metric_name=metric_name,
    )
    text = format_profile_for_agent(profile, max_chars=max_chars)
    return text.replace("[DATA PROFILE]", "[DATA INSPECTION]", 1)


def run_ydata_profile(
    train: pd.DataFrame,
    target_col: str,
    private_dir: Path,
    *,
    max_rows: int,
    max_cols: int,
    timeout_sec: int,
    minimal: bool = True,
) -> dict[str, Any]:
    """Run ydata-profiling if available and save only private artifacts."""
    started = time.time()
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        return {
            "available": False,
            "success": False,
            "error_type": "ImportError",
            "error_message": "ydata-profiling is not installed",
        }

    try:
        df = train.copy()
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=42)
        if df.shape[1] > max_cols:
            keep = [target_col] if target_col in df.columns else []
            keep.extend([c for c in df.columns if c != target_col][: max_cols - len(keep)])
            df = df[keep]

        if time.time() - started > timeout_sec:
            raise TimeoutError(f"ydata setup exceeded {timeout_sec}s")

        profile = ProfileReport(
            df,
            title="AutoVibe Train Data Profile",
            minimal=minimal,
            explorative=not minimal,
            progress_bar=False,
        )
        private_dir.mkdir(parents=True, exist_ok=True)
        html_path = private_dir / "data_profile_ydata.html"
        json_path = private_dir / "data_profile_ydata.json"
        profile.to_file(str(html_path))
        if time.time() - started > timeout_sec:
            raise TimeoutError(f"ydata profiling exceeded {timeout_sec}s")
        json_text = profile.to_json()
        json_path.write_text(json_text, encoding="utf-8")
        parsed = json.loads(json_text)
        return {
            "available": True,
            "success": True,
            "rows_profiled": int(df.shape[0]),
            "cols_profiled": int(df.shape[1]),
            "elapsed_seconds": round(time.time() - started, 2),
            "html_path": str(html_path),
            "json_path": str(json_path),
            "summary": extract_ydata_summary(parsed),
        }
    except Exception as exc:
        return {
            "available": True,
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": _clip(str(exc), 500),
            "elapsed_seconds": round(time.time() - started, 2),
        }


def extract_ydata_summary(profile_json: dict[str, Any]) -> dict[str, Any]:
    """Extract a defensive, compact summary from ydata JSON."""
    table = profile_json.get("table") or {}
    variables = profile_json.get("variables") or {}
    alerts = profile_json.get("alerts") or []

    type_counts: dict[str, int] = {}
    high_missing: list[str] = []
    constant: list[str] = []
    high_cardinality: list[str] = []
    skewed_numeric: list[str] = []

    for name, info in variables.items():
        if not isinstance(info, dict):
            continue
        var_type = str(info.get("type") or info.get("var_type") or "unknown")
        type_counts[var_type] = type_counts.get(var_type, 0) + 1
        missing_pct = _to_float(info.get("p_missing"))
        if missing_pct is not None and missing_pct >= 0.2:
            high_missing.append(name)
        n_distinct = _to_float(info.get("n_distinct"))
        count = _to_float(info.get("n"))
        if n_distinct == 1:
            constant.append(name)
        if count and n_distinct and count > 0 and n_distinct / count > 0.9:
            high_cardinality.append(name)
        skewness = _to_float(info.get("skewness"))
        if skewness is not None and abs(skewness) >= 2:
            skewed_numeric.append(name)

    alert_text = []
    if isinstance(alerts, list):
        for alert in alerts[:20]:
            alert_text.append(_clip(str(alert), 160))
    elif isinstance(alerts, dict):
        for key, value in list(alerts.items())[:20]:
            alert_text.append(_clip(f"{key}: {value}", 160))

    return {
        "n_rows": table.get("n"),
        "n_cols": table.get("n_var"),
        "missing_cells": table.get("n_cells_missing"),
        "duplicate_rows": table.get("n_duplicates"),
        "variable_types": type_counts,
        "high_missing_columns": high_missing[:15],
        "constant_columns": constant[:15],
        "high_cardinality_columns": high_cardinality[:15],
        "skewed_numeric_columns": skewed_numeric[:15],
        "alerts": alert_text,
    }


def format_ydata_profile_for_agent(
    compact_profile: dict[str, Any],
    ydata_result: dict[str, Any],
    *,
    max_chars: int = 5000,
) -> str:
    backend = "compact+ydata" if ydata_result.get("success") else "compact"
    lines = [
        "[DATA PROFILE]",
        f"backend={backend}",
        f"rows_profiled={ydata_result.get('rows_profiled', compact_profile.get('train_shape', ['?'])[0])}",
        f"cols_profiled={ydata_result.get('cols_profiled', compact_profile.get('train_shape', ['?', '?'])[1])}",
        f"target_col={compact_profile.get('target_col')}",
        "",
        "Overview:",
    ]
    ysummary = ydata_result.get("summary") if ydata_result.get("success") else {}
    if ysummary:
        lines.extend(
            [
                f"- n_rows={ysummary.get('n_rows')}",
                f"- n_cols={ysummary.get('n_cols')}",
                f"- missing_cells={ysummary.get('missing_cells')}",
                f"- duplicate_rows={ysummary.get('duplicate_rows')}",
                f"- variable_types={_format_value(ysummary.get('variable_types'))}",
            ]
        )
    else:
        lines.extend(
            [
                f"- train_shape={compact_profile.get('train_shape')}",
                f"- val_shape={compact_profile.get('val_shape')}",
                f"- duplicate_rows={compact_profile.get('duplicate_rows_train')}",
                f"- dtype_counts={_format_value(compact_profile.get('dtype_counts'))}",
            ]
        )

    if ydata_result and not ydata_result.get("success"):
        lines.extend(
            [
                "",
                "[PROFILE] ydata-profiling was unavailable or failed; compact pandas profile was returned instead.",
            ]
        )
    elif ydata_result.get("success"):
        lines.append("Full ydata profile saved as private MLflow artifact for developers.")

    lines.extend(
        [
            "",
            "Warnings:",
            f"- high_missing_columns: {_format_value(compact_profile.get('missing_values_top'))}",
            f"- constant_columns: {_format_value(compact_profile.get('constant_columns'))}",
            f"- high_cardinality_columns: {_format_value(compact_profile.get('high_cardinality_columns'))}",
            f"- skewed_numeric_columns: {_format_value((ysummary or {}).get('skewed_numeric_columns'))}",
            f"- possible_id_like_columns: {_format_value(compact_profile.get('id_like_or_suspicious_columns'))}",
            f"- possible_datetime_columns: {_format_value(compact_profile.get('possible_datetime_columns'))}",
            f"- possible_text_columns: {_format_value(compact_profile.get('possible_text_columns'))}",
            f"- train_val_schema_issues: {_format_value(compact_profile.get('train_val_schema_diff'))}",
            f"- unseen_categories_in_val: {_format_value(compact_profile.get('unseen_categories_in_val'))}",
            "",
            "Target:",
            f"- target_distribution_train: {_format_value(compact_profile.get('target_distribution_train'))}",
            f"- target_distribution_val: {_format_value(compact_profile.get('target_distribution_val'))}",
            f"- class_imbalance_warning: {_format_value(compact_profile.get('class_imbalance_warning'))}",
            "",
            "Candidate modelling implications:",
            f"- recommended_preprocessing: {_format_value(compact_profile.get('suggested_next_steps'))}",
        ]
    )
    implications = compact_profile.get("modelling_implications") or {}
    lines.extend(
        [
            f"- likely_need_categorical_handling: {implications.get('likely_need_categorical_handling')}",
            f"- likely_need_imputation: {implications.get('likely_need_imputation')}",
            "- likely_need_raw_row_pipeline: yes",
        ]
    )
    return _clip("\n".join(lines), max_chars)


def profile_config_from_env() -> dict[str, Any]:
    return {
        "enable_ydata": os.getenv("AUTOVIBE_ENABLE_YDATA_PROFILE", "0") == "1",
        "max_rows": _int_env("AUTOVIBE_PROFILE_MAX_ROWS", 5000),
        "max_cols": _int_env("AUTOVIBE_PROFILE_MAX_COLS", 80),
        "timeout_sec": _int_env("AUTOVIBE_PROFILE_TIMEOUT_SEC", 60),
        "minimal": os.getenv("AUTOVIBE_PROFILE_MINIMAL", "1") != "0",
    }


def _drop_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col in df.columns:
        return df.drop(columns=[target_col])
    return df.copy()


def _dtype_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()}


def _target_distribution(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    if target_col not in df.columns:
        return {"error": "target column not present"}
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=False) > 20:
        desc = target.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        return {str(k): _round(v) for k, v in desc.items()}
    counts = target.value_counts(dropna=False).head(20)
    total = max(len(target), 1)
    return {
        _short_value(k): {"count": int(v), "pct": round(float(v) / total, 4)}
        for k, v in counts.items()
    }


def _missing_top(df: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    if df.empty:
        return []
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False).head(limit)
    total = max(len(df), 1)
    return [
        {"column": str(col), "missing": int(count), "pct": round(float(count) / total, 4)}
        for col, count in counts.items()
    ]


def _constant_columns(df: pd.DataFrame, limit: int = 30) -> list[str]:
    return [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1][:limit]


def _quasi_constant_columns(df: pd.DataFrame, limit: int = 30) -> list[str]:
    out: list[str] = []
    n = max(len(df), 1)
    for col in df.columns:
        top = df[col].value_counts(dropna=False).head(1)
        if not top.empty and float(top.iloc[0]) / n >= 0.98 and df[col].nunique(dropna=False) > 1:
            out.append(str(col))
    return out[:limit]


def _categorical_columns(df: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    return [
        {"column": str(col), "nunique": int(df[col].nunique(dropna=True))}
        for col in cols[:limit]
    ]


def _high_cardinality_columns(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if nunique >= min(50, max(10, int(0.5 * n))) or (nunique / n >= 0.8 and n >= 10):
            out.append({"column": str(col), "nunique": nunique, "unique_ratio": round(nunique / n, 4)})
    return out[:limit]


def _id_like_columns(df: pd.DataFrame, limit: int = 20) -> list[str]:
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        name = str(col).lower()
        nunique = int(df[col].nunique(dropna=True))
        if name in {"id", "uuid", "guid"} or name.endswith("_id") or nunique / n >= 0.95:
            out.append(str(col))
    return out[:limit]


def _possible_datetime_columns(df: pd.DataFrame, limit: int = 20) -> list[str]:
    out = []
    for col in df.columns:
        series = df[col]
        name = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(series) or any(k in name for k in ("date", "time", "timestamp")):
            out.append(str(col))
            continue
        if series.dtype == object:
            sample = series.dropna().astype(str).head(25)
            if len(sample) >= 3:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.8:
                    out.append(str(col))
    return out[:limit]


def _possible_text_columns(df: pd.DataFrame, limit: int = 20) -> list[str]:
    out = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            continue
        avg_len = sample.str.len().mean()
        has_space = sample.str.contains(r"\s", regex=True).mean()
        if avg_len >= 25 or has_space >= 0.5:
            out.append(str(col))
    return out[:limit]


def _numeric_summary(df: pd.DataFrame, limit: int = 12) -> dict[str, dict[str, Any]]:
    numeric = df.select_dtypes(include=[np.number]).columns[:limit]
    out: dict[str, dict[str, Any]] = {}
    for col in numeric:
        desc = df[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        out[str(col)] = {str(k): _round(v) for k, v in desc.items()}
    return out


def _categorical_summary(df: pd.DataFrame, limit: int = 12) -> dict[str, dict[str, Any]]:
    cols = df.select_dtypes(include=["object", "category", "bool"]).columns[:limit]
    out: dict[str, dict[str, Any]] = {}
    for col in cols:
        counts = df[col].value_counts(dropna=False).head(5)
        out[str(col)] = {_short_value(k): int(v) for k, v in counts.items()}
    return out


def _schema_diff(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
    train_cols = set(map(str, train.columns))
    val_cols = set(map(str, val.columns))
    dtype_mismatch = []
    for col in sorted(train_cols & val_cols):
        if str(train[col].dtype) != str(val[col].dtype):
            dtype_mismatch.append(
                {"column": col, "train_dtype": str(train[col].dtype), "val_dtype": str(val[col].dtype)}
            )
    return {
        "missing_in_val": sorted(train_cols - val_cols)[:30],
        "extra_in_val": sorted(val_cols - train_cols)[:30],
        "dtype_mismatch": dtype_mismatch[:30],
    }


def _unseen_categories(train: pd.DataFrame, val: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    out = []
    common = [c for c in train.columns if c in val.columns]
    for col in common:
        if not (
            pd.api.types.is_object_dtype(train[col])
            or isinstance(train[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train[col])
        ):
            continue
        train_values = set(train[col].dropna().astype(str).unique())
        val_values = set(val[col].dropna().astype(str).unique())
        unseen = sorted(val_values - train_values)
        if unseen:
            out.append({"column": str(col), "unseen_count": len(unseen), "examples": unseen[:8]})
    return out[:limit]


def _missingness_shift(train: pd.DataFrame, val: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    out = []
    for col in train.columns:
        if col not in val.columns:
            continue
        train_pct = float(train[col].isna().mean())
        val_pct = float(val[col].isna().mean())
        delta = abs(train_pct - val_pct)
        if delta >= 0.1:
            out.append({"column": str(col), "train_pct": round(train_pct, 4), "val_pct": round(val_pct, 4), "delta": round(delta, 4)})
    out.sort(key=lambda item: item["delta"], reverse=True)
    return out[:limit]


def _numeric_shift(train: pd.DataFrame, val: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    out = []
    for col in train.select_dtypes(include=[np.number]).columns:
        if col not in val.columns or not pd.api.types.is_numeric_dtype(val[col]):
            continue
        t = train[col].dropna()
        v = val[col].dropna()
        if len(t) < 2 or len(v) < 2:
            continue
        t_std = float(t.std()) or 1.0
        mean_diff = abs(float(v.mean()) - float(t.mean())) / t_std
        q_train = t.quantile([0.25, 0.5, 0.75]).to_numpy()
        q_val = v.quantile([0.25, 0.5, 0.75]).to_numpy()
        q_diff = float(np.nanmean(np.abs(q_val - q_train))) / t_std
        if mean_diff >= 0.5 or q_diff >= 0.5:
            out.append({"column": str(col), "mean_diff_std": round(mean_diff, 4), "quantile_diff_std": round(q_diff, 4)})
    out.sort(key=lambda item: max(item["mean_diff_std"], item["quantile_diff_std"]), reverse=True)
    return out[:limit]


def _target_associations(df: pd.DataFrame, target_col: str, limit: int = 12) -> list[dict[str, Any]]:
    if target_col not in df.columns:
        return []
    target = df[target_col]
    out = []
    if pd.api.types.is_numeric_dtype(target):
        for col in _drop_target(df, target_col).select_dtypes(include=[np.number]).columns:
            corr = df[col].corr(target)
            if pd.notna(corr) and abs(float(corr)) >= 0.4:
                out.append({"column": str(col), "association": "corr", "value": round(float(corr), 4)})
    else:
        for col in _drop_target(df, target_col).select_dtypes(include=[np.number]).columns:
            grouped = df.groupby(target_col, dropna=False)[col].mean(numeric_only=True)
            if len(grouped) > 1:
                spread = float(grouped.max() - grouped.min())
                std = float(df[col].std()) or 1.0
                strength = spread / std
                if strength >= 0.7:
                    out.append({"column": str(col), "association": "class_mean_spread", "value": round(strength, 4)})
    out.sort(key=lambda item: abs(item["value"]), reverse=True)
    return out[:limit]


def _class_imbalance_warning(profile: dict[str, Any]) -> str | None:
    dist = profile.get("target_distribution_train")
    if not isinstance(dist, dict) or not dist:
        return None
    pcts = [v.get("pct") for v in dist.values() if isinstance(v, dict) and "pct" in v]
    if not pcts:
        return None
    if min(pcts) <= 0.1 or max(pcts) >= 0.8:
        return "target distribution is imbalanced; consider stratification, class weights, or robust metrics"
    return None


def _modelling_implications(profile: dict[str, Any]) -> dict[str, str]:
    return {
        "likely_need_categorical_handling": "yes" if profile.get("object_or_categorical_columns") else "no",
        "likely_need_imputation": "yes" if profile.get("missing_values_top") else "no",
        "likely_need_raw_row_pipeline": "yes",
    }


def _suggest_next_steps(profile: dict[str, Any]) -> list[str]:
    steps = [
        "Build a simple raw-row-ready sklearn Pipeline before tuning.",
        "Validate with the environment before final submission.",
    ]
    if profile.get("missing_values_top"):
        steps.insert(0, "Add imputation inside the Pipeline.")
    if profile.get("object_or_categorical_columns"):
        steps.insert(0, "Encode categorical columns inside a ColumnTransformer with unknown-category handling.")
    if profile.get("possible_datetime_columns"):
        steps.append("If deriving datetime features, put the derivation inside the final estimator or transformer.")
    if profile.get("high_cardinality_columns") or profile.get("id_like_or_suspicious_columns"):
        steps.append("Treat high-cardinality or id-like columns cautiously; avoid leakage-prone identifiers.")
    if profile.get("unseen_categories_in_val"):
        steps.append("Use encoders that tolerate validation/test categories unseen during fit.")
    return steps[:8]


def _sample_rows(df: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        return []
    rows = df.head(max_rows).to_dict(orient="records")
    return [
        {str(k): _short_value(v) for k, v in row.items()}
        for row in rows
    ]


def _format_value(value: Any) -> str:
    if value in (None, [], {}):
        return "none"
    try:
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"


def _short_value(value: Any, limit: int = 80) -> str:
    if pd.isna(value):
        return "<NA>"
    text = str(value)
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _round(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 5)
    except Exception:
        return _short_value(value)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return _short_value(value)
