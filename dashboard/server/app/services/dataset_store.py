"""Dataset discovery and management for ``<repo>/datasets/<name>/``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from gym.dataset_ingestion import (
    create_default_dataset_config,
    list_dataset_files,
    load_dataset_config,
    prepare_dataset as prepare_dataset_dir,
    preview_dataset_entry,
    save_dataset_config,
    upload_dataset_file,
    validate_dataset_config,
    download_dataset_url,
)

from ..config import get_settings

_MIN_METRICS = {"rmse", "rmsle", "mae", "mse", "logloss"}


def _meta_dir(dataset_root: Path) -> Path:
    if (dataset_root / "prepared" / "meta.json").exists():
        return dataset_root / "prepared"
    return dataset_root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _prepared_meta(dataset_root: Path) -> dict[str, Any]:
    return _read_json(_meta_dir(dataset_root) / "meta.json")


def _metric_goal(metric_name: str | None) -> str:
    if not metric_name:
        return "max"
    name = metric_name.strip().lower()
    if name.startswith("neg_"):
        return "max"
    return "min" if name in _MIN_METRICS else "max"


def _task_label(meta: dict[str, Any], target_unique: int | None) -> str:
    task = (meta.get("task_type") or "").lower()
    if "regress" in task:
        return "Регрессия"
    if "class" in task:
        return "Классификация"
    metric = (meta.get("metric_name") or meta.get("metric") or "").lower()
    if metric.startswith("neg_") or metric in _MIN_METRICS:
        return "Регрессия"
    if metric.startswith("f1") or "auc" in metric or "acc" in metric:
        return "Классификация"
    if target_unique is not None:
        return "Классификация" if target_unique <= 20 else "Регрессия"
    return "—"


def _summary_description(config_data: dict[str, Any], meta: dict[str, Any]) -> str:
    dataset_notes = config_data.get("dataset_notes") or meta.get("dataset_notes") or meta.get("notes") or {}
    return (
        str(dataset_notes.get("short_description") or "")
        or str((meta.get("notes") or {}).get("description") or "")
        or str(meta.get("description") or "")
    )


def _source_label(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("title") or source.get("name") or source.get("url") or "—")
    return str(source) if source else "—"


def describe_dataset(dataset_root: Path, *, deep: bool = False) -> dict[str, Any]:
    prepared = (_meta_dir(dataset_root) / "meta.json").exists() and (_meta_dir(dataset_root) / "train.csv").exists()
    meta = _prepared_meta(dataset_root)

    config_data: dict[str, Any] = {}
    try:
        config_data = load_dataset_config(dataset_root).to_dict()
    except Exception:
        config_data = {}

    task_cfg = dict(config_data.get("task") or {})
    display_meta = {
        "task_type": meta.get("task_type") or task_cfg.get("type"),
        "metric": meta.get("metric_name") or meta.get("metric") or task_cfg.get("metric"),
    }
    target = meta.get("target_col") or task_cfg.get("target_col") or "—"
    metric = display_meta["metric"] or "—"
    rows = int(meta.get("n_rows_prepared") or 0)
    cols = int(meta.get("n_features") or 0)
    target_unique: int | None = None

    if prepared and deep and not rows:
        train_path = _meta_dir(dataset_root) / "train.csv"
        val_path = _meta_dir(dataset_root) / "val.csv"
        test_path = _meta_dir(dataset_root) / "test.csv"
        try:
            rows = len(pd.read_csv(train_path)) + len(pd.read_csv(val_path)) + len(pd.read_csv(test_path))
            if target and target != "—":
                sample = pd.read_csv(train_path)
                cols = max(sample.shape[1] - 1, 0)
                target_unique = int(sample[target].nunique()) if target in sample.columns else None
        except Exception:
            rows = cols = 0

    return {
        "id": dataset_root.name,
        "name": meta.get("name") or config_data.get("name") or dataset_root.name,
        "task": _task_label(display_meta, target_unique),
        "metric": metric,
        "metricGoal": _metric_goal(metric),
        "rows": rows,
        "cols": cols,
        "target": target,
        "source": _source_label(meta.get("source") or config_data.get("source")),
        "desc": _summary_description(config_data, meta),
        "prepared": prepared,
        "dir": str(dataset_root),
        "datasetDir": str(dataset_root),
        "seed": meta.get("seed") or (config_data.get("split") or {}).get("seed") or 42,
        "suite": meta.get("suite") or config_data.get("suite"),
        "ingestionMode": (config_data.get("ingestion") or {}).get("mode"),
    }


def list_datasets(deep: bool = True) -> list[dict[str, Any]]:
    root = get_settings().datasets_dir
    if not root.exists():
        return []
    return [describe_dataset(child, deep=deep) for child in sorted(root.iterdir()) if child.is_dir()]


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    root = get_settings().datasets_dir / dataset_id
    if not root.is_dir():
        return None
    summary = describe_dataset(root, deep=True)
    config_data: dict[str, Any] | None = None
    try:
        config_data = load_dataset_config(root).to_dict()
    except Exception:
        config_data = None
    return {
        **summary,
        "config": config_data,
        "preparedMeta": _prepared_meta(root),
        "files": list_dataset_files(root),
    }


def create_dataset(name: str) -> dict[str, Any]:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_").strip("-_") or "dataset"
    root = get_settings().datasets_dir / safe
    if root.exists():
        raise ValueError(f"Dataset '{safe}' already exists.")
    save_dataset_config(root, create_default_dataset_config(safe))
    return get_dataset(safe) or describe_dataset(root, deep=False)


def save_config(dataset_id: str, config_data: dict[str, Any]) -> dict[str, Any]:
    root = get_settings().datasets_dir / dataset_id
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    save_dataset_config(root, config_data)
    return get_dataset(dataset_id) or describe_dataset(root, deep=False)


def upload_files(dataset_id: str, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    root = get_settings().datasets_dir / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    for file_name, content in files:
        saved.append(upload_dataset_file(root, file_name, content))
    return saved


def download_files(dataset_id: str, urls: list[dict[str, str]]) -> list[dict[str, Any]]:
    root = get_settings().datasets_dir / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for item in urls:
        results.append(
            download_dataset_url(
                root,
                item["url"],
                suggested_name=item.get("suggested_name") or None,
            )
        )
    return results


def preview(dataset_id: str, *, logical_name: str | None = None, split_role: str | None = None, joined: bool = False, limit: int = 20) -> dict[str, Any]:
    root = get_settings().datasets_dir / dataset_id
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return preview_dataset_entry(root, logical_name=logical_name, split_role=split_role, joined=joined, limit=limit)


def validate(dataset_id: str) -> dict[str, Any]:
    root = get_settings().datasets_dir / dataset_id
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return validate_dataset_config(root)


def prepare(dataset_id: str) -> dict[str, Any]:
    root = get_settings().datasets_dir / dataset_id
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return prepare_dataset_dir(root)


def preview_rows(dataset_id: str, split: str = "train", limit: int = 50) -> dict[str, Any]:
    meta_dir = _meta_dir(get_settings().datasets_dir / dataset_id)
    path = meta_dir / f"{split}.csv"
    if not path.exists():
        return {"columns": [], "rows": [], "total": 0}
    df = pd.read_csv(path)
    head = df.head(limit)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": head.where(pd.notnull(head), None).values.tolist(),
        "total": int(df.shape[0]),
        "shown": int(head.shape[0]),
    }


def column_stats(dataset_id: str, split: str = "train") -> list[dict[str, Any]]:
    meta_dir = _meta_dir(get_settings().datasets_dir / dataset_id)
    path = meta_dir / f"{split}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    n = max(len(df), 1)
    stats: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isnull().sum())
        is_num = pd.api.types.is_numeric_dtype(series)
        hist: list[float] = []
        if is_num and series.notnull().any():
            counts, _ = pd.cut(series.dropna(), bins=min(12, max(series.nunique(), 1)), retbins=True)
            hist = counts.value_counts(sort=False).tolist()
        else:
            hist = series.value_counts().head(12).tolist()
        stats.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "kind": "numeric" if is_num else "categorical",
                "missingPct": round(missing / n * 100, 1),
                "unique": int(series.nunique(dropna=True)),
                "hist": [int(x) for x in hist],
            }
        )
    return stats


def delete_dataset(dataset_id: str) -> bool:
    root = get_settings().datasets_dir / dataset_id
    if not root.is_dir():
        return False
    shutil.rmtree(root)
    return True
