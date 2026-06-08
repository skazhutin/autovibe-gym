"""Task discovery, upload staging, preparation, and metadata management.

The runner contract stays intentionally small: a runnable task has
``prepared/train.csv``, ``prepared/val.csv``, ``prepared/test.csv``, and
``prepared/meta.json``. The dashboard can store richer raw files and
``dataset_config.json`` beside that prepared form.
"""
from __future__ import annotations

import gzip
import ipaddress
import json
import re
import shutil
import socket
import tarfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ..config import get_settings

_MIN_METRICS = {"rmse", "rmsle", "mae", "mse", "logloss"}
_MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 500 * 1024 * 1024
_MAX_EXTRACTED_FILES = 500
_TABLE_EXTS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
}
_ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".gz"}
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_DATASET_CACHE: dict[tuple[str, bool], tuple[tuple[tuple[str, int, int], ...], dict[str, Any]]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_task_id(value: str) -> str:
    """Return a stable, path-safe task slug."""
    original = (value or "").strip()
    original_parts = PurePosixPath(original.replace("\\", "/")).parts
    if any(part in {"..", "."} for part in original_parts) or "/" in original or "\\" in original:
        raise ValueError("Task id must not contain path separators or relative path segments.")
    raw = original.lower()
    raw = re.sub(r"\s+", "-", raw)
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
    safe = re.sub(r"-{2,}", "-", safe).strip("-_")
    if not safe or not _SLUG_RE.match(safe):
        raise ValueError("Task id must start with a letter/number and use only a-z, 0-9, '-' or '_'.")
    return safe


def _safe_child(base: Path, rel: str | Path) -> Path:
    """Resolve a user-controlled relative path under ``base``."""
    if isinstance(rel, Path):
        rel_text = rel.as_posix()
    else:
        rel_text = str(rel)
    rel_text = rel_text.replace("\\", "/").strip("/")
    parts = PurePosixPath(rel_text).parts
    if not rel_text or any(part in {"", ".", ".."} for part in parts) or PurePosixPath(rel_text).is_absolute():
        raise ValueError(f"Unsafe path: {rel}")
    root = base.resolve()
    out = (root / Path(*parts)).resolve()
    out.relative_to(root)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_yaml_config(dataset_root: Path) -> dict[str, Any]:
    for filename in ("config.yaml", "config.yml"):
        path = dataset_root / filename
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text("utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _timestamp_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


def _meta_dir(dataset_root: Path) -> Path:
    if (dataset_root / "prepared" / "meta.json").exists():
        return dataset_root / "prepared"
    return dataset_root


def _config_path(dataset_root: Path) -> Path:
    return dataset_root / "dataset_config.json"


def _metric_goal(metric_name: str | None) -> str:
    if not metric_name:
        return "max"
    name = metric_name.strip().lower()
    if name.startswith("neg_"):
        return "max"
    return "min" if name in _MIN_METRICS else "max"


def _task_type(meta: dict[str, Any], config: dict[str, Any], target_unique: int | None) -> str:
    task = ((config.get("task") or {}).get("task_type") or meta.get("task_type") or "").lower()
    if task in {"classification", "regression", "auto"}:
        return task
    metric = (meta.get("metric_name") or meta.get("metric") or "").lower()
    if metric.startswith("neg_") or metric in _MIN_METRICS:
        return "regression"
    if metric.startswith("f1") or "auc" in metric or "acc" in metric or metric == "logloss":
        return "classification"
    if target_unique is not None:
        return "classification" if target_unique <= 20 else "regression"
    return "unknown"


def _task_label(task_type: str) -> str:
    if task_type == "classification":
        return "classification"
    if task_type == "regression":
        return "regression"
    return "unknown"


def _source_label(config: dict[str, Any], meta: dict[str, Any]) -> str:
    sources = config.get("sources")
    if isinstance(sources, list) and sources:
        first = sources[0] or {}
        if isinstance(first, dict):
            value = first.get("name") or first.get("url") or first.get("id") or first.get("provider")
            return str(value) if value else "-"
    source = meta.get("source")
    if isinstance(source, dict):
        value = source.get("name") or source.get("url") or source.get("id") or source.get("provider")
        return str(value) if value else "-"
    return str(source) if source else "-"


def _csv_shape(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    cols = len(pd.read_csv(path, nrows=0).columns)
    try:
        with path.open("rb") as fh:
            rows = max(sum(1 for _ in fh) - 1, 0)
    except OSError:
        rows = 0
    return (int(rows), int(cols))


def _split_paths(root: Path) -> dict[str, Path]:
    meta_dir = _meta_dir(root)
    return {split: meta_dir / f"{split}.csv" for split in ("train", "val", "test")}


def _task_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Fingerprint files that affect the task card/detail summary."""
    paths = [
        _config_path(root),
        _meta_dir(root) / "meta.json",
        *_split_paths(root).values(),
        root / "config.yaml",
        root / "config.yml",
    ]
    out: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append((path.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(out)


def _invalidate_task_cache(task_id: str | None = None) -> None:
    if task_id is None:
        _DATASET_CACHE.clear()
        return
    for key in [key for key in _DATASET_CACHE if key[0] == task_id]:
        _DATASET_CACHE.pop(key, None)


def _status_from_files(root: Path, meta: dict[str, Any] | None = None) -> str:
    paths = _split_paths(root)
    has_train = paths["train"].exists()
    has_val = paths["val"].exists()
    has_test = paths["test"].exists()
    has_meta = bool(meta) or (_meta_dir(root) / "meta.json").exists()
    if has_train and has_val and has_test and has_meta:
        return "prepared"
    if has_train:
        return "partial"
    return "unprepared"


def _describe_split(root: Path, split: str, deep: bool) -> dict[str, Any] | None:
    path = _split_paths(root)[split]
    if not path.exists():
        return None
    rows = cols = 0
    if deep:
        rows, cols = _csv_shape(path)
    return {
        "path": f"prepared/{split}.csv" if (root / "prepared").exists() else f"{split}.csv",
        "source_path": None,
        "rows": rows,
        "cols": cols,
    }


def _config_from_legacy(dataset_root: Path, meta: dict[str, Any], deep: bool) -> dict[str, Any]:
    yaml_config = _read_yaml_config(dataset_root)
    yaml_task = dict(yaml_config.get("task") or {})
    yaml_split = dict(yaml_config.get("split") or {})
    yaml_raw = dict(yaml_config.get("raw_data") or {})
    status = _status_from_files(dataset_root, meta)
    target = meta.get("target_col") or meta.get("target") or yaml_task.get("target_col") or ""
    metric = meta.get("metric_name") or meta.get("metric") or yaml_task.get("metric") or ""
    task_type = meta.get("task_type") or yaml_task.get("type") or _task_type(meta, {}, None)
    source = meta.get("source") or yaml_config.get("source")
    sources = [source] if source else []
    return {
        "id": dataset_root.name,
        "name": meta.get("name") or yaml_config.get("name") or dataset_root.name,
        "created_at": _timestamp_value(meta.get("created_at") or yaml_config.get("created_at")),
        "updated_at": _timestamp_value(meta.get("updated_at")),
        "version": 1,
        "status": status,
        "task": {
            "task_type": task_type,
            "target_col": target,
            "metric_name": metric,
            "metric_goal": _metric_goal(metric),
            "positive_label": None,
            "class_labels": [],
            "id_columns": [],
            "ignore_columns": [],
            "sample_weight_col": None,
            "group_col": None,
            "time_col": None,
        },
        "splits": {
            "mode": "prepared_files",
            "train": _describe_split(dataset_root, "train", deep),
            "val": _describe_split(dataset_root, "val", deep),
            "test": _describe_split(dataset_root, "test", deep),
            "ratios": None,
            "seed": int(meta.get("seed") or yaml_split.get("seed") or 42),
            "shuffle": True,
            "stratify": "auto",
        },
        "raw_files": [{"path": f"raw_data/{path}", "name": Path(path).name} for path in yaml_raw.get("files", [])],
        "agent_notes": {
            "task_description": (meta.get("notes") or {}).get("description", ""),
            "data_structure": "",
            "column_descriptions": {},
            "additional_comments": "",
            "leakage_warning": "",
            "visible_to_agent": True,
        },
        "sources": sources,
        "tags": list(meta.get("tags") or []),
        "warnings": [],
    }


def describe_task(dataset_root: Path, *, deep: bool = False) -> dict[str, Any]:
    """Build the UI Task record. ``deep`` reads prepared CSV row counts."""
    meta_dir = _meta_dir(dataset_root)
    meta = _read_json(meta_dir / "meta.json")
    config = _read_json(_config_path(dataset_root))
    if not config:
        config = _config_from_legacy(dataset_root, meta, deep)

    task = dict(config.get("task") or {})
    splits = dict(config.get("splits") or {})
    target = task.get("target_col") or meta.get("target_col") or meta.get("target")
    metric = task.get("metric_name") or meta.get("metric_name") or meta.get("metric")
    metric_goal = task.get("metric_goal") or _metric_goal(metric)

    paths = _split_paths(dataset_root)
    has_train = paths["train"].exists()
    has_val = paths["val"].exists()
    has_test = paths["test"].exists()
    rows = cols = 0
    target_unique: int | None = None
    split_summary: dict[str, Any] = {}
    if deep:
        for split, path in paths.items():
            if not path.exists():
                split_summary[split] = None
                continue
            sr, sc = _csv_shape(path)
            rows += sr
            cols = max(cols, max(sc - (1 if target else 0), 0))
            split_summary[split] = {
                "path": f"prepared/{split}.csv" if (dataset_root / "prepared").exists() else f"{split}.csv",
                "source_path": ((splits.get(split) or {}) if isinstance(splits.get(split), dict) else {}).get("source_path"),
                "rows": sr,
                "cols": sc,
            }
        target_unique_needed = _task_type(meta, config, None) == "unknown"
        if target and has_train and target_unique_needed:
            try:
                col = pd.read_csv(paths["train"], usecols=[target])[target]
                target_unique = int(col.nunique())
            except (ValueError, KeyError, OSError):
                target_unique = None
    else:
        for split in ("train", "val", "test"):
            split_summary[split] = splits.get(split)

    task_type = _task_type(meta, config, target_unique)
    status = str(config.get("status") or _status_from_files(dataset_root, meta))
    prepared = status == "prepared" and has_train and has_val and has_test
    notes = dict(config.get("agent_notes") or {})
    desc = (
        notes.get("task_description")
        or (meta.get("notes") or {}).get("description")
        or (meta.get("notes") or {}).get("desc")
        or meta.get("description")
        or ""
    )
    warnings = list(config.get("warnings") or [])
    return {
        "id": dataset_root.name,
        "name": config.get("name") or meta.get("name") or dataset_root.name,
        "task": _task_label(task_type),
        "taskType": task_type,
        "metric": metric or "-",
        "metricGoal": metric_goal,
        "rows": rows,
        "cols": cols,
        "target": target or "-",
        "source": _source_label(config, meta),
        "desc": desc,
        "prepared": prepared,
        "status": status,
        "dir": str(dataset_root),
        "datasetDir": str(dataset_root),
        "seed": int(splits.get("seed") or meta.get("seed", 42)),
        "tags": list(config.get("tags") or meta.get("tags") or []),
        "createdAt": config.get("created_at") or meta.get("created_at"),
        "updatedAt": config.get("updated_at") or meta.get("updated_at"),
        "hasTrain": has_train,
        "hasVal": has_val,
        "hasTest": has_test,
        "splits": split_summary,
        "rawFiles": list(config.get("raw_files") or []),
        "warnings": warnings,
        "warningsCount": len(warnings),
        "sources": list(config.get("sources") or []),
    }


def list_tasks(deep: bool = True) -> list[dict[str, Any]]:
    root = get_settings().datasets_dir
    if not root.exists():
        return []
    tasks: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        fingerprint = _task_fingerprint(child)
        key = (child.name, deep)
        cached = _DATASET_CACHE.get(key)
        if cached and cached[0] == fingerprint:
            tasks.append(dict(cached[1]))
            continue
        described = describe_task(child, deep=deep)
        _DATASET_CACHE[key] = (fingerprint, described)
        tasks.append(dict(described))
    return tasks


def _task_root(task_id: str) -> Path:
    safe = sanitize_task_id(task_id)
    return _safe_child(get_settings().datasets_dir, safe)


def get_task(task_id: str) -> dict[str, Any] | None:
    try:
        root = _task_root(task_id)
    except ValueError:
        return None
    if not root.is_dir():
        return None
    return describe_task(root, deep=True)


def _format_from_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return "tar.gz"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or path.suffix.lower() in _ARCHIVE_EXTS


def _is_table(path: Path) -> bool:
    return path.suffix.lower() in _TABLE_EXTS


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=nrows)
        if suffix == ".txt":
            return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True, nrows=nrows)
        if suffix == ".json":
            try:
                return pd.read_json(path, lines=True, nrows=nrows)
            except ValueError:
                df = pd.read_json(path)
                return df.head(nrows) if nrows is not None else df
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
            return df.head(nrows) if nrows is not None else df
        if suffix == ".parquet":
            df = pd.read_parquet(path)
            return df.head(nrows) if nrows is not None else df
        if suffix == ".feather":
            df = pd.read_feather(path)
            return df.head(nrows) if nrows is not None else df
    except ImportError as exc:
        raise ValueError(f"Reading {suffix} requires an optional dependency: {exc}") from exc
    except ModuleNotFoundError as exc:
        raise ValueError(f"Reading {suffix} requires an optional dependency: {exc}") from exc
    raise ValueError(f"Unsupported table format: {suffix or path.name}")


def _preview_table(path: Path, *, limit: int = 50) -> dict[str, Any]:
    if not _is_table(path):
        raise ValueError(f"Not a supported table file: {path.name}")
    head = _read_table(path, nrows=limit)
    total: int | None = None
    if path.suffix.lower() in {".csv", ".tsv", ".txt", ".jsonl"}:
        try:
            with path.open("rb") as fh:
                total = max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            total = None
    dtypes = {str(c): str(t) for c, t in head.dtypes.items()}
    missing = {str(c): int(head[c].isnull().sum()) for c in head.columns}
    return {
        "columns": [str(c) for c in head.columns],
        "rows": head.where(pd.notnull(head), None).values.tolist(),
        "total": total,
        "shown": int(head.shape[0]),
        "dtypes": dtypes,
        "missing": missing,
        "warnings": [] if total is not None else ["Total rows are unknown for this format."],
    }


def preview_rows(task_id: str, split: str = "train", limit: int = 50) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    root = _task_root(task_id)
    path = _split_paths(root)[split]
    if not path.exists():
        return {"columns": [], "rows": [], "total": 0, "shown": 0, "dtypes": {}, "missing": {}, "warnings": []}
    out = _preview_table(path, limit=limit)
    if out["total"] is None:
        out["total"] = int(pd.read_csv(path).shape[0])
    return out


def column_stats(task_id: str, split: str = "train") -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    root = _task_root(task_id)
    path = _split_paths(root)[split]
    if not path.exists():
        return []
    df = pd.read_csv(path)
    cfg = get_task_config(task_id) or {}
    task = dict(cfg.get("task") or {})
    target = task.get("target_col")
    id_cols = set(task.get("id_columns") or [])
    ignore_cols = set(task.get("ignore_columns") or [])
    n = max(len(df), 1)
    stats: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isnull().sum())
        is_num = pd.api.types.is_numeric_dtype(series)
        if is_num and series.notnull().any():
            try:
                bins = min(12, max(int(series.nunique(dropna=True)), 1))
                counts = pd.cut(series.dropna(), bins=bins).value_counts(sort=False).tolist()
            except ValueError:
                counts = [int(series.notnull().sum())]
        else:
            counts = series.value_counts(dropna=True).head(12).tolist()
        stats.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "kind": "numeric" if is_num else "categorical",
                "missingPct": round(missing / n * 100, 1),
                "unique": int(series.nunique(dropna=True)),
                "hist": [int(x) for x in counts],
                "target": str(col) == str(target),
                "ignored": str(col) in ignore_cols,
                "idColumn": str(col) in id_cols,
            }
        )
    return stats


def _upload_root(upload_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_-]{8,80}$", upload_id or ""):
        raise ValueError("Invalid upload id")
    root = _safe_child(get_settings().uploads_dir, upload_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_upload_root() -> tuple[str, Path]:
    upload_id = uuid.uuid4().hex
    return upload_id, _upload_root(upload_id)


def _safe_filename(name: str) -> str:
    parsed = Path(unquote(urlparse(name).path)).name if "://" in name else Path(name).name
    clean = "".join(ch for ch in parsed if ch.isalnum() or ch in "._- ").strip().replace(" ", "_")
    return clean or f"file-{uuid.uuid4().hex[:8]}"


def _validate_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs are supported")
    host = parsed.hostname
    if not host:
        raise ValueError("Download URL must include a hostname")
    if host.lower() in {"localhost"} or host.lower().endswith(".localhost"):
        raise ValueError("Downloads from localhost are not allowed")
    try:
        addresses = socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve download host: {host}") from exc
    for family, _, _, _, sockaddr in addresses:
        raw_ip = sockaddr[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ValueError(f"Could not parse resolved address for {host}: {raw_ip}") from exc
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Downloads from private/internal addresses are not allowed: {host}")
    return url


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _file_info(path: Path, base: Path) -> dict[str, Any]:
    rel = path.resolve().relative_to(base.resolve()).as_posix()
    info: dict[str, Any] = {
        "id": rel,
        "path": rel,
        "name": path.name,
        "size": int(path.stat().st_size) if path.is_file() else 0,
        "format": _format_from_path(path) if path.is_file() else "dir",
        "kind": "dir" if path.is_dir() else "file",
        "readable": False,
        "rows": None,
        "cols": None,
        "status": "ready",
        "warnings": [],
    }
    if path.is_file() and _is_table(path):
        info["readable"] = True
        try:
            preview = _preview_table(path, limit=5)
            info["rows"] = preview["total"]
            info["cols"] = len(preview["columns"])
            info["status"] = "readable"
        except ValueError as exc:
            info["status"] = "error"
            info["warnings"] = [str(exc)]
    elif path.is_file() and _is_archive(path):
        info["status"] = "archive"
    return info


def _file_tree(base: Path) -> list[dict[str, Any]]:
    if not base.exists():
        return []

    def node(path: Path) -> dict[str, Any]:
        info = _file_info(path, base)
        if path.is_dir():
            info["children"] = [node(child) for child in sorted(path.iterdir())]
        return info

    return [node(child) for child in sorted(base.iterdir())]


def _flat_files(base: Path) -> list[dict[str, Any]]:
    if not base.exists():
        return []
    return [_file_info(path, base) for path in sorted(base.rglob("*")) if path.is_file()]


def upload_file(filename: str, data: bytes, upload_id: str | None = None) -> dict[str, Any]:
    uid, root = (upload_id, _upload_root(upload_id)) if upload_id else _new_upload_root()
    assert uid is not None
    uploaded = root / "uploaded"
    uploaded.mkdir(parents=True, exist_ok=True)
    dest = _safe_child(uploaded, _safe_filename(filename))
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}-{uuid.uuid4().hex[:6]}{dest.suffix}")
    dest.write_bytes(data)
    return {"upload_id": uid, "file": _file_info(dest, root), "files": _file_tree(root), "flat": _flat_files(root)}


def upload_from_url(url: str, upload_id: str | None = None) -> dict[str, Any]:
    safe_url = _validate_download_url(url)
    parsed = urlparse(safe_url)
    req = Request(safe_url, headers={"User-Agent": "autovibe-gym-dashboard/0.1"})
    opener = build_opener(_SafeRedirectHandler)
    with opener.open(req, timeout=20) as resp:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                raise ValueError("Download exceeds the maximum allowed size")
            chunks.append(chunk)
    return upload_file(_safe_filename(parsed.path or "download"), b"".join(chunks), upload_id)


def list_uploaded_files(upload_id: str) -> dict[str, Any]:
    root = _upload_root(upload_id)
    return {"upload_id": upload_id, "files": _file_tree(root), "flat": _flat_files(root)}


def _stage_file(upload_id: str, rel_path: str) -> Path:
    root = _upload_root(upload_id)
    return _safe_child(root, rel_path)


def preview_upload(upload_id: str, rel_path: str, limit: int = 50) -> dict[str, Any]:
    path = _stage_file(upload_id, rel_path)
    if not path.exists() or not path.is_file():
        raise ValueError("Uploaded file not found")
    return _preview_table(path, limit=limit)


def _archive_target(base: Path, member_name: str) -> Path:
    clean = member_name.replace("\\", "/").strip("/")
    parts = PurePosixPath(clean).parts
    if not clean or any(part in {"", ".", ".."} for part in parts) or PurePosixPath(clean).is_absolute():
        raise ValueError(f"Unsafe archive member: {member_name}")
    return _safe_child(base, clean)


def extract_upload_archive(upload_id: str, rel_path: str | None = None) -> dict[str, Any]:
    root = _upload_root(upload_id)
    candidates = [_safe_child(root, rel_path)] if rel_path else [p for p in root.rglob("*") if p.is_file() and _is_archive(p)]
    extracted_root = root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0

    def check_file(size: int) -> None:
        nonlocal count, total
        count += 1
        check_bytes(size)

    def check_bytes(size: int) -> None:
        nonlocal total
        total += max(size, 0)
        if count > _MAX_EXTRACTED_FILES:
            raise ValueError("Archive contains too many files")
        if total > _MAX_EXTRACTED_BYTES:
            raise ValueError("Archive extraction exceeds the maximum allowed size")

    for archive in candidates:
        if not archive.exists() or not archive.is_file():
            raise ValueError("Archive file not found")
        lower = archive.name.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        _archive_target(extracted_root, info.filename).mkdir(parents=True, exist_ok=True)
                        continue
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        raise ValueError(f"Refusing to extract symlink: {info.filename}")
                    check_file(info.file_size)
                    target = _archive_target(extracted_root, info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        elif lower.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Refusing to extract link: {member.name}")
                    target = _archive_target(extracted_root, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    check_file(member.size)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        elif lower.endswith(".gz"):
            out_name = archive.name[:-3] or f"decompressed-{uuid.uuid4().hex[:6]}"
            target = _archive_target(extracted_root, out_name)
            check_file(0)
            with gzip.open(archive, "rb") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    check_bytes(len(chunk))
                    dst.write(chunk)
        else:
            raise ValueError(f"Unsupported archive format: {archive.name}")
    return {"upload_id": upload_id, "files": _file_tree(root), "flat": _flat_files(root)}


def _copy_upload_to_dataset(upload_id: str | None, dataset_root: Path) -> list[dict[str, Any]]:
    if not upload_id:
        return []
    stage = _upload_root(upload_id)
    raw_root = dataset_root / "raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        dest = _safe_child(raw_root, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        info = _file_info(dest, dataset_root)
        info["path"] = f"raw/{rel}"
        info["original_name"] = path.name
        copied.append(info)
    return copied


def _source_path(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            nested = val.get("source_path") or val.get("path")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _read_source_table(upload_id: str | None, dataset_root: Path, source_path: str) -> tuple[pd.DataFrame, str]:
    rel = source_path.replace("\\", "/").strip("/")
    if rel.startswith("raw/"):
        path = _safe_child(dataset_root, rel)
        config_path = rel
    elif rel.startswith("prepared/"):
        path = _safe_child(dataset_root, rel)
        config_path = rel
    elif upload_id:
        path = _stage_file(upload_id, rel)
        config_path = f"raw/{rel}"
    else:
        path = _safe_child(dataset_root / "raw", rel)
        config_path = f"raw/{rel}"
    if not path.exists():
        raise ValueError(f"Selected file does not exist: {source_path}")
    return _read_table(path), config_path


def _validate_target(df: pd.DataFrame, target: str, split: str) -> None:
    if target and target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing in {split}.")


def _stratify_series(df: pd.DataFrame, target: str, task_type: str, mode: str, warnings: list[str]) -> Any:
    if mode == "off" or not target or target not in df.columns:
        return None
    if mode == "on" or task_type == "classification" or (task_type == "auto" and df[target].nunique() <= 50):
        counts = df[target].value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            return df[target]
        warnings.append("Stratification was requested but class counts are too small; regular split was used.")
    return None


def _split_raw_table(
    df: pd.DataFrame,
    *,
    target: str,
    task_type: str,
    ratios: dict[str, Any],
    seed: int,
    shuffle: bool,
    stratify_mode: str,
    warnings: list[str],
) -> dict[str, pd.DataFrame]:
    train_ratio = float(ratios.get("train", 0.7))
    val_ratio = float(ratios.get("val", 0.15))
    test_ratio = float(ratios.get("test", 0.15))
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0 or abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0, with train > 0 and val/test >= 0.")
    _validate_target(df, target, "raw table")
    if val_ratio == 0 and test_ratio == 0:
        return {"train": df.reset_index(drop=True)}

    temp_ratio = val_ratio + test_ratio
    strat = _stratify_series(df, target, task_type, stratify_mode, warnings) if shuffle else None
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=seed,
            shuffle=shuffle,
            stratify=strat,
        )
    except ValueError as exc:
        warnings.append(f"Stratified split failed ({exc}); regular split was used.")
        train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=seed, shuffle=shuffle)

    out = {"train": train_df.reset_index(drop=True)}
    if val_ratio > 0 and test_ratio > 0:
        relative_test = test_ratio / temp_ratio
        second_strat = _stratify_series(temp_df, target, task_type, stratify_mode, warnings) if shuffle else None
        try:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test,
                random_state=seed,
                shuffle=shuffle,
                stratify=second_strat,
            )
        except ValueError as exc:
            warnings.append(f"Secondary stratified split failed ({exc}); regular split was used.")
            val_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=seed, shuffle=shuffle)
        out["val"] = val_df.reset_index(drop=True)
        out["test"] = test_df.reset_index(drop=True)
    elif val_ratio > 0:
        out["val"] = temp_df.reset_index(drop=True)
    elif test_ratio > 0:
        out["test"] = temp_df.reset_index(drop=True)
    return out


def _split_train_for_val(
    train_df: pd.DataFrame,
    *,
    target: str,
    task_type: str,
    val_ratio: float,
    seed: int,
    stratify_mode: str,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    strat = _stratify_series(train_df, target, task_type, stratify_mode, warnings)
    try:
        new_train, val = train_test_split(train_df, test_size=val_ratio, random_state=seed, shuffle=True, stratify=strat)
    except ValueError as exc:
        warnings.append(f"Validation stratified split failed ({exc}); regular split was used.")
        new_train, val = train_test_split(train_df, test_size=val_ratio, random_state=seed, shuffle=True)
    return new_train.reset_index(drop=True), val.reset_index(drop=True)


def _write_prepared(
    dataset_root: Path,
    frames: dict[str, pd.DataFrame],
    source_paths: dict[str, str | None],
) -> dict[str, Any]:
    prepared = dataset_root / "prepared"
    if prepared.exists():
        for old in prepared.glob("*.csv"):
            old.unlink()
    prepared.mkdir(parents=True, exist_ok=True)
    split_cfg: dict[str, Any] = {}
    for split, df in frames.items():
        df.to_csv(prepared / f"{split}.csv", index=False)
        split_cfg[split] = {
            "path": f"prepared/{split}.csv",
            "source_path": source_paths.get(split),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
    for split in ("train", "val", "test"):
        split_cfg.setdefault(split, None)
    return split_cfg


def _compatible_meta(config: dict[str, Any]) -> dict[str, Any]:
    task = dict(config.get("task") or {})
    notes = dict(config.get("agent_notes") or {})
    meta = {
        "name": config.get("name") or config.get("id"),
        "target_col": task.get("target_col"),
        "metric_name": task.get("metric_name"),
        "task_type": task.get("task_type") or "auto",
        "seed": (config.get("splits") or {}).get("seed", 42),
        "notes": {"description": notes.get("task_description") or ""},
    }
    sources = config.get("sources") or []
    if sources:
        meta["source"] = sources[0]
    return {k: v for k, v in meta.items() if v is not None}


def _status_from_prepared(frames: dict[str, pd.DataFrame]) -> str:
    if all(split in frames for split in ("train", "val", "test")):
        return "prepared"
    if "train" in frames:
        return "partial"
    return "unprepared"


def create_from_config(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = sanitize_task_id(str(payload.get("id") or payload.get("name") or "dataset"))
    final_root = _task_root(task_id)
    if final_root.exists():
        raise ValueError(f"Task '{task_id}' already exists.")
    upload_id = payload.get("upload_id") or payload.get("uploadId")
    task = dict(payload.get("task") or {})
    splits = dict(payload.get("splits") or {})
    warnings = list(payload.get("warnings") or [])
    target = str(task.get("target_col") or task.get("target") or "").strip()
    metric = str(task.get("metric_name") or task.get("metric") or "").strip()
    task_type = str(task.get("task_type") or "auto")
    seed = int(splits.get("seed") or task.get("seed") or payload.get("seed") or 42)
    if not payload.get("name"):
        raise ValueError("Task name is required.")
    if not target:
        raise ValueError("Target column is required.")
    if not metric:
        raise ValueError("Metric is required.")

    dataset_root = _safe_child(get_settings().datasets_dir, f".{task_id}-{uuid.uuid4().hex[:8]}")
    dataset_root.mkdir(parents=True)
    try:
        return _create_from_config_in_root(
            payload=payload,
            task_id=task_id,
            dataset_root=dataset_root,
            final_root=final_root,
            upload_id=str(upload_id) if upload_id else None,
            task=task,
            splits=splits,
            warnings=warnings,
            target=target,
            metric=metric,
            task_type=task_type,
            seed=seed,
        )
    except Exception:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        raise


def _create_from_config_in_root(
    *,
    payload: dict[str, Any],
    task_id: str,
    dataset_root: Path,
    final_root: Path,
    upload_id: str | None,
    task: dict[str, Any],
    splits: dict[str, Any],
    warnings: list[str],
    target: str,
    metric: str,
    task_type: str,
    seed: int,
) -> dict[str, Any]:
    raw_files = _copy_upload_to_dataset(str(upload_id) if upload_id else None, dataset_root)
    mode = str(splits.get("mode") or "raw_split")
    source_paths: dict[str, str | None] = {}
    frames: dict[str, pd.DataFrame] = {}

    if mode == "raw_split":
        raw_path = _source_path(splits, "raw_path", "rawPath", "source_path", "source")
        if not raw_path:
            raise ValueError("Raw split mode requires a selected raw table.")
        df, config_source = _read_source_table(str(upload_id) if upload_id else None, dataset_root, raw_path)
        ratios = dict(splits.get("ratios") or {"train": 0.7, "val": 0.15, "test": 0.15})
        frames = _split_raw_table(
            df,
            target=target,
            task_type=task_type,
            ratios=ratios,
            seed=seed,
            shuffle=bool(splits.get("shuffle", True)),
            stratify_mode=str(splits.get("stratify") or "auto"),
            warnings=warnings,
        )
        source_paths = {split: config_source for split in frames}
    elif mode == "prepared_files":
        mapping = splits.get("mapping") or splits
        train_path = _source_path(mapping, "train_source_path", "trainSourcePath", "train")
        if not train_path:
            raise ValueError("Prepared-files mode requires a train file.")
        train_df, train_source = _read_source_table(str(upload_id) if upload_id else None, dataset_root, train_path)
        _validate_target(train_df, target, "train")
        frames["train"] = train_df.reset_index(drop=True)
        source_paths["train"] = train_source
        val_path = _source_path(mapping, "val_source_path", "valSourcePath", "val")
        test_path = _source_path(mapping, "test_source_path", "testSourcePath", "test")
        if val_path:
            val_df, val_source = _read_source_table(str(upload_id) if upload_id else None, dataset_root, val_path)
            _validate_target(val_df, target, "val")
            frames["val"] = val_df.reset_index(drop=True)
            source_paths["val"] = val_source
        elif bool(splits.get("create_val_from_train") or splits.get("createValFromTrain")):
            frames["train"], frames["val"] = _split_train_for_val(
                frames["train"],
                target=target,
                task_type=task_type,
                val_ratio=float(splits.get("val_ratio") or splits.get("valRatio") or 0.15),
                seed=seed,
                stratify_mode=str(splits.get("stratify") or "auto"),
                warnings=warnings,
            )
            source_paths["val"] = train_source
        else:
            warnings.append("Validation split is missing; multi-step validation will be unavailable.")
        if test_path:
            test_df, test_source = _read_source_table(str(upload_id) if upload_id else None, dataset_root, test_path)
            _validate_target(test_df, target, "test")
            frames["test"] = test_df.reset_index(drop=True)
            source_paths["test"] = test_source
        else:
            warnings.append("Without a test split, final benchmark scoring will be unavailable.")
    else:
        raise ValueError("splits.mode must be raw_split or prepared_files.")

    split_cfg = _write_prepared(dataset_root, frames, source_paths)
    status = _status_from_prepared(frames)
    created = _now_iso()
    final_config = {
        "id": task_id,
        "name": str(payload.get("name")),
        "created_at": created,
        "updated_at": created,
        "version": 1,
        "status": status,
        "task": {
            "task_type": task_type,
            "target_col": target,
            "metric_name": metric,
            "metric_goal": task.get("metric_goal") or _metric_goal(metric),
            "positive_label": task.get("positive_label"),
            "class_labels": list(task.get("class_labels") or []),
            "id_columns": list(task.get("id_columns") or []),
            "ignore_columns": list(task.get("ignore_columns") or []),
            "sample_weight_col": task.get("sample_weight_col"),
            "group_col": task.get("group_col"),
            "time_col": task.get("time_col"),
            "max_runtime": task.get("max_runtime"),
            "max_steps": task.get("max_steps"),
            "allowed_libraries": list(task.get("allowed_libraries") or []),
            "constraints": task.get("constraints") or "",
        },
        "splits": {
            "mode": mode,
            **split_cfg,
            "ratios": splits.get("ratios"),
            "seed": seed,
            "shuffle": bool(splits.get("shuffle", True)),
            "stratify": str(splits.get("stratify") or "auto"),
            "create_val_from_train": bool(splits.get("create_val_from_train") or splits.get("createValFromTrain")),
            "val_ratio": splits.get("val_ratio") or splits.get("valRatio"),
        },
        "raw_files": raw_files,
        "agent_notes": {
            "task_description": ((payload.get("agent_notes") or {}).get("task_description") or payload.get("desc") or ""),
            "data_structure": (payload.get("agent_notes") or {}).get("data_structure") or "",
            "column_descriptions": (payload.get("agent_notes") or {}).get("column_descriptions") or {},
            "additional_comments": (payload.get("agent_notes") or {}).get("additional_comments") or "",
            "leakage_warning": (payload.get("agent_notes") or {}).get("leakage_warning") or "",
            "visible_to_agent": bool((payload.get("agent_notes") or {}).get("visible_to_agent", True)),
        },
        "sources": list(payload.get("sources") or []),
        "tags": list(payload.get("tags") or []),
        "warnings": warnings,
    }
    _write_json(_config_path(dataset_root), final_config)
    _write_json(dataset_root / "prepared" / "meta.json", _compatible_meta(final_config))
    if final_root.exists():
        raise ValueError(f"Task '{task_id}' already exists.")
    dataset_root.rename(final_root)
    _invalidate_task_cache(task_id)
    return describe_task(final_root, deep=True)


def get_task_config(task_id: str) -> dict[str, Any] | None:
    try:
        root = _task_root(task_id)
    except ValueError:
        return None
    if not root.is_dir():
        return None
    meta = _read_json(_meta_dir(root) / "meta.json")
    config = _read_json(_config_path(root))
    return config or _config_from_legacy(root, meta, deep=True)


def update_task_config(task_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    root = _task_root(task_id)
    if not root.is_dir():
        return None
    current = get_task_config(task_id) or {}
    merged = {**current, **updates}
    if "task" in updates:
        merged["task"] = {**dict(current.get("task") or {}), **dict(updates.get("task") or {})}
    if "splits" in updates:
        merged["splits"] = {**dict(current.get("splits") or {}), **dict(updates.get("splits") or {})}
    if "agent_notes" in updates:
        merged["agent_notes"] = {**dict(current.get("agent_notes") or {}), **dict(updates.get("agent_notes") or {})}
    merged["id"] = root.name
    merged["updated_at"] = _now_iso()
    merged["status"] = _status_from_files(root, _read_json(_meta_dir(root) / "meta.json"))
    _write_json(_config_path(root), merged)
    _write_json(_meta_dir(root) / "meta.json", _compatible_meta(merged))
    _invalidate_task_cache(task_id)
    return describe_task(root, deep=True)


def update_meta(task_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Backward-compatible shallow metadata edit endpoint."""
    config = get_task_config(task_id)
    if config is None:
        return None
    task = dict(config.get("task") or {})
    notes = dict(config.get("agent_notes") or {})
    if updates.get("name"):
        config["name"] = updates["name"]
    if updates.get("target"):
        task["target_col"] = updates["target"]
    if updates.get("metric"):
        task["metric_name"] = updates["metric"]
        task["metric_goal"] = _metric_goal(str(updates["metric"]))
    if updates.get("task_type"):
        task["task_type"] = updates["task_type"]
    if updates.get("seed") is not None:
        config.setdefault("splits", {})["seed"] = updates["seed"]
    if updates.get("desc") is not None:
        notes["task_description"] = updates["desc"]
    config["task"] = task
    config["agent_notes"] = notes
    return update_task_config(task_id, config)


def create_task(name: str, files: dict[str, bytes], meta: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible multipart CSV upload wrapper."""
    upload = None
    upload_id = None
    for split, data in files.items():
        if data:
            upload = upload_file(f"{split}.csv", data, upload_id)
            upload_id = upload["upload_id"]
    mapping: dict[str, Any] = {}
    if "train" in files:
        mapping["train"] = "uploaded/train.csv"
    if "val" in files:
        mapping["val"] = "uploaded/val.csv"
    if "test" in files:
        mapping["test"] = "uploaded/test.csv"
    return create_from_config(
        {
            "id": sanitize_task_id(name),
            "name": meta.get("name") or name,
            "upload_id": upload_id,
            "task": {
                "task_type": meta.get("task_type") or "auto",
                "target_col": meta.get("target") or meta.get("target_col"),
                "metric_name": meta.get("metric") or meta.get("metric_name"),
            },
            "splits": {"mode": "prepared_files", "mapping": mapping, "seed": meta.get("seed", 42)},
            "agent_notes": {"task_description": meta.get("desc") or "", "visible_to_agent": True},
        }
    )


def prepare_task(task_id: str) -> dict[str, Any] | None:
    """Rebuild prepared files from the saved config when raw files are present."""
    config = get_task_config(task_id)
    if config is None:
        return None
    root = _task_root(task_id)
    task = dict(config.get("task") or {})
    splits = dict(config.get("splits") or {})
    warnings = list(config.get("warnings") or [])
    target = str(task.get("target_col") or "")
    metric = str(task.get("metric_name") or "")
    task_type = str(task.get("task_type") or "auto")
    seed = int(splits.get("seed") or 42)
    if not target or not metric:
        raise ValueError("Task config must define target_col and metric_name before prepare.")

    frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str | None] = {}
    mode = str(splits.get("mode") or "prepared_files")
    if mode == "raw_split":
        source = ((splits.get("train") or {}) if isinstance(splits.get("train"), dict) else {}).get("source_path")
        if not source:
            raise ValueError("Raw split config is missing source_path.")
        df, config_source = _read_source_table(None, root, str(source))
        frames = _split_raw_table(
            df,
            target=target,
            task_type=task_type,
            ratios=dict(splits.get("ratios") or {"train": 0.7, "val": 0.15, "test": 0.15}),
            seed=seed,
            shuffle=bool(splits.get("shuffle", True)),
            stratify_mode=str(splits.get("stratify") or "auto"),
            warnings=warnings,
        )
        source_paths = {split: config_source for split in frames}
    elif mode == "prepared_files":
        for split in ("train", "val", "test"):
            split_cfg = splits.get(split)
            if not isinstance(split_cfg, dict):
                continue
            split_source = _source_path(split_cfg, "source_path", "path")
            if not split_source:
                continue
            df, config_source = _read_source_table(None, root, str(split_source))
            _validate_target(df, target, split)
            frames[split] = df.reset_index(drop=True)
            source_paths[split] = config_source
        if "train" not in frames:
            raise ValueError("Prepared-files config is missing a train source.")
        if "val" not in frames and splits.get("create_val_from_train"):
            frames["train"], frames["val"] = _split_train_for_val(
                frames["train"],
                target=target,
                task_type=task_type,
                val_ratio=float(splits.get("val_ratio") or 0.15),
                seed=seed,
                stratify_mode=str(splits.get("stratify") or "auto"),
                warnings=warnings,
            )
            source_paths["val"] = source_paths.get("train")
    else:
        raise ValueError("splits.mode must be raw_split or prepared_files.")

    split_cfg = _write_prepared(root, frames, source_paths)
    config["splits"] = {**splits, **split_cfg}
    config["updated_at"] = _now_iso()
    config["warnings"] = warnings
    config["status"] = _status_from_prepared(frames)
    _write_json(_config_path(root), config)
    _write_json(root / "prepared" / "meta.json", _compatible_meta(config))
    _invalidate_task_cache(task_id)
    return describe_task(root, deep=True)


def delete_task(task_id: str) -> bool:
    try:
        root = _task_root(task_id)
    except ValueError:
        return False
    if not root.is_dir():
        return False
    shutil.rmtree(root)
    _invalidate_task_cache(task_id)
    return True


# Backward-compatible dataset_* aliases retained for older tests and callers.
sanitize_dataset_id = sanitize_task_id
list_datasets = list_tasks
get_dataset = get_task
get_dataset_config = get_task_config
describe_dataset = describe_task
_invalidate_dataset_cache = _invalidate_task_cache
