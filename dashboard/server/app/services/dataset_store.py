"""Dataset discovery and management over ``<repo>/datasets/<name>/``.

A dataset is "runnable" when ``prepared/`` holds train/val/test.csv + meta.json
(the form ``experiments.run_gym --dataset-dir`` consumes). The management screen
also surfaces not-yet-prepared dataset folders so they can be inspected/removed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_settings

# Heuristic: how the UI labels the task and which direction is "better".
_MIN_METRICS = {"rmse", "rmsle", "mae", "mse", "logloss"}


def _meta_dir(dataset_root: Path) -> Path:
    """Return the dir holding meta.json (``prepared/`` if present)."""
    if (dataset_root / "prepared" / "meta.json").exists():
        return dataset_root / "prepared"
    return dataset_root


def _read_meta(meta_path: Path) -> dict[str, Any]:
    try:
        return json.loads(meta_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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


def _csv_shape(path: Path) -> tuple[int, int]:
    """(rows, cols) without loading the whole frame into memory twice."""
    if not path.exists():
        return (0, 0)
    df = pd.read_csv(path)
    return (int(df.shape[0]), int(df.shape[1]))


def _source_label(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("name") or source.get("url") or source.get("id") or "—")
    return str(source) if source else "—"


def describe_dataset(dataset_root: Path, *, deep: bool = False) -> dict[str, Any]:
    """Build the UI Dataset record. ``deep`` reads CSVs for row/col counts."""
    name = dataset_root.name
    meta_dir = _meta_dir(dataset_root)
    meta = _read_meta(meta_dir / "meta.json")
    prepared = (meta_dir / "meta.json").exists() and (meta_dir / "train.csv").exists()

    target = meta.get("target_col") or meta.get("target")
    metric = meta.get("metric_name") or meta.get("metric")

    rows = cols = 0
    target_unique: int | None = None
    if prepared and deep:
        tr, tc = _csv_shape(meta_dir / "train.csv")
        vr, _ = _csv_shape(meta_dir / "val.csv")
        ter, _ = _csv_shape(meta_dir / "test.csv")
        rows = tr + vr + ter
        cols = max(tc - 1, 0)  # features (exclude target column)
        if target:
            try:
                col = pd.read_csv(meta_dir / "train.csv", usecols=[target])[target]
                target_unique = int(col.nunique())
            except (ValueError, KeyError, OSError):
                target_unique = None

    return {
        "id": name,
        "name": meta.get("name") or name,
        "task": _task_label(meta, target_unique),
        "metric": metric or "—",
        "metricGoal": _metric_goal(metric),
        "rows": rows,
        "cols": cols,
        "target": target or "—",
        "source": _source_label(meta.get("source")),
        "desc": (meta.get("notes") or {}).get("description")
        or (meta.get("notes") or {}).get("desc")
        or meta.get("description")
        or "",
        "prepared": prepared,
        "dir": str(dataset_root),
        "datasetDir": str(dataset_root),
        "seed": meta.get("seed", 42),
        "suite": meta.get("suite"),
    }


def list_datasets(deep: bool = True) -> list[dict[str, Any]]:
    s = get_settings()
    root = s.datasets_dir
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        out.append(describe_dataset(child, deep=deep))
    return out


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    s = get_settings()
    root = s.datasets_dir / dataset_id
    if not root.is_dir():
        return None
    return describe_dataset(root, deep=True)


def preview_rows(dataset_id: str, split: str = "train", limit: int = 50) -> dict[str, Any]:
    s = get_settings()
    meta_dir = _meta_dir(s.datasets_dir / dataset_id)
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
    s = get_settings()
    meta_dir = _meta_dir(s.datasets_dir / dataset_id)
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


def update_meta(dataset_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    s = get_settings()
    root = s.datasets_dir / dataset_id
    if not root.is_dir():
        return None
    meta_dir = _meta_dir(root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "meta.json"
    meta = _read_meta(meta_path)
    allowed = {"name", "target_col", "metric_name", "task_type", "source", "seed", "description"}
    for key, value in updates.items():
        if value is None:
            continue
        if key == "target":
            meta["target_col"] = value
        elif key == "metric":
            meta["metric_name"] = value
        elif key == "desc":
            meta.setdefault("notes", {})["description"] = value
        elif key in allowed:
            meta[key] = value
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return describe_dataset(root, deep=True)


def create_dataset(name: str, files: dict[str, bytes], meta: dict[str, Any]) -> dict[str, Any]:
    """Create datasets/<name>/prepared/ from uploaded CSVs + meta fields."""
    s = get_settings()
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_").strip("-_") or "dataset"
    root = s.datasets_dir / safe
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if split in files and files[split]:
            (prepared / f"{split}.csv").write_bytes(files[split])
    meta_out = {"name": meta.get("name") or safe}
    if meta.get("target"):
        meta_out["target_col"] = meta["target"]
    if meta.get("metric"):
        meta_out["metric_name"] = meta["metric"]
    if meta.get("task_type"):
        meta_out["task_type"] = meta["task_type"]
    if meta.get("seed") is not None:
        meta_out["seed"] = meta["seed"]
    if meta.get("desc"):
        meta_out["notes"] = {"description": meta["desc"]}
    (prepared / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), "utf-8")
    return describe_dataset(root, deep=True)


def delete_dataset(dataset_id: str) -> bool:
    s = get_settings()
    root = s.datasets_dir / dataset_id
    if not root.is_dir():
        return False
    shutil.rmtree(root)
    return True
