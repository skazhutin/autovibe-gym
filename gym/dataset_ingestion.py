from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import shutil
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
import zipfile

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .tabular_io import (
    PREVIEW_FULL_READ_LIMIT_BYTES,
    TabularIOError,
    infer_format,
    is_safe_archive_member,
    is_supported_tabular_path,
    list_archive_members,
    load_tabular_dataframe,
)


CONFIG_FILENAMES = ("config.yaml", "config.yml")
FRACTION_TOLERANCE = 1e-6
DOWNLOAD_TIMEOUT_SEC = 30
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
JOIN_ROW_GROWTH_WARN_RATIO = 1.2
JOIN_ROW_GROWTH_ERROR_RATIO = 3.0


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    suite: str
    source: dict[str, Any]
    ingestion: dict[str, Any]
    relations: dict[str, Any]
    task: dict[str, Any]
    split: dict[str, Any]
    preparation: dict[str, Any]
    role: str | None
    notes: dict[str, Any]
    dataset_notes: dict[str, Any]
    raw_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "suite": self.suite,
            "source": self.source,
            "dataset_notes": self.dataset_notes,
            "ingestion": self.ingestion,
            "task": self.task,
            "split": self.split,
            "preparation": self.preparation,
        }
        if self.relations:
            payload["relations"] = self.relations
        if self.role is not None:
            payload["role"] = self.role
        if self.notes:
            payload["notes"] = self.notes
        return payload


def discover_dataset_dirs(datasets_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    if not datasets_root.exists():
        return discovered
    for child in datasets_root.iterdir():
        if not child.is_dir():
            continue
        if config_path(child) is not None:
            discovered[child.name] = child
        elif (child / "meta.json").exists() and (child / "train.csv").exists():
            discovered[child.name] = child
    return dict(sorted(discovered.items()))


def config_path(dataset_dir: Path) -> Path | None:
    for filename in CONFIG_FILENAMES:
        path = dataset_dir / filename
        if path.exists():
            return path
    return None


def create_default_dataset_config(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "suite": "custom",
        "source": {
            "title": "",
            "url": "",
            "license": "",
            "citation": "",
            "description": "",
        },
        "dataset_notes": {
            "short_description": "",
            "llm_context": "",
            "warnings": [],
            "known_pitfalls": [],
        },
        "ingestion": {
            "mode": "raw",
            "files": [
                {
                    "logical_name": "table_1",
                    "role": "base",
                    "source_type": "upload",
                    "url": "",
                    "path": "",
                    "format": "auto",
                    "read_options": {},
                    "optional": False,
                    "archive_member": "",
                }
            ],
        },
        "relations": {
            "base_table": "table_1",
            "joins": [],
        },
        "task": {
            "type": "classification",
            "target_col": "",
            "metric": "f1_weighted",
            "forbidden_columns": [],
        },
        "split": {
            "strategy": "stratified_random",
            "seed": 42,
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "create_val_from_train_if_missing": True,
            "val_fraction_from_train": 0.15,
        },
        "preparation": {
            "drop_columns": [],
            "rename_columns": {},
            "target_mapping": {},
        },
    }


def load_dataset_config(dataset_dir: Path) -> DatasetConfig:
    cfg_path = config_path(dataset_dir)
    if cfg_path is None:
        meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
        dataset_notes = dict(meta.get("dataset_notes") or meta.get("notes") or {})
        notes = dict(meta.get("notes") or {})
        return DatasetConfig(
            name=str(meta.get("name") or dataset_dir.name),
            suite="legacy",
            source=dict(meta.get("source") or {"type": "legacy"}),
            ingestion={"mode": "legacy_prepared", "files": []},
            relations={},
            task={
                "type": meta.get("task_type", "classification"),
                "target_col": meta["target_col"],
                "metric": meta.get("metric") or meta.get("metric_name") or "f1_weighted",
                "forbidden_columns": [],
            },
            split={
                "strategy": meta.get("split_strategy", "fixed"),
                "seed": meta.get("seed", 42),
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
            },
            preparation={},
            role=meta.get("role"),
            notes=notes,
            dataset_notes=dataset_notes,
            raw_data={"files": []},
        )

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset config must be a mapping: {cfg_path}")
    return normalize_dataset_config(raw, fallback_name=dataset_dir.name)


def normalize_dataset_config(raw: dict[str, Any], *, fallback_name: str = "dataset") -> DatasetConfig:
    name = str(raw.get("name") or fallback_name)
    suite = str(raw.get("suite") or "custom")
    source = _normalize_source(raw.get("source"))
    notes = dict(raw.get("notes") or {})
    dataset_notes = _normalize_dataset_notes(raw.get("dataset_notes"), notes)
    ingestion = _normalize_ingestion(raw.get("ingestion"), raw.get("raw_data"))
    relations = _normalize_relations(raw.get("relations"), ingestion)
    task = _normalize_task(raw.get("task"))
    split = _normalize_split(raw.get("split"))
    preparation = _normalize_preparation(raw.get("preparation"))
    raw_data = _legacy_raw_data_view(raw.get("raw_data"), ingestion)
    return DatasetConfig(
        name=name,
        suite=suite,
        source=source,
        ingestion=ingestion,
        relations=relations,
        task=task,
        split=split,
        preparation=preparation,
        role=raw.get("role"),
        notes=notes,
        dataset_notes=dataset_notes,
        raw_data=raw_data,
    )


def save_dataset_config(dataset_dir: Path, config_data: dict[str, Any]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw_data").mkdir(parents=True, exist_ok=True)
    normalized = normalize_dataset_config(config_data, fallback_name=dataset_dir.name)
    path = dataset_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(normalized.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def list_dataset_files(dataset_dir: Path) -> list[dict[str, Any]]:
    raw_dir = dataset_dir / "raw_data"
    if not raw_dir.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dataset_dir).as_posix()
        if rel.startswith("raw_data/_zip_cache/"):
            continue
        try:
            fmt = infer_format(path.name)
        except TabularIOError:
            fmt = "unsupported"
        entry = {
            "path": rel,
            "name": path.name,
            "size": int(path.stat().st_size),
            "format": fmt,
        }
        if fmt == "zip":
            entry["archive_members"] = list_archive_members(path)
        files.append(entry)
    return files


def upload_dataset_file(dataset_dir: Path, file_name: str, content: bytes) -> dict[str, Any]:
    raw_dir = dataset_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _unique_safe_filename(raw_dir, file_name or "upload.bin")
    target = raw_dir / safe_name
    target.write_bytes(content)
    response = {
        "path": target.relative_to(dataset_dir).as_posix(),
        "name": target.name,
        "size": int(target.stat().st_size),
        "format": infer_format(target.name) if is_supported_tabular_path(target.name) else "unsupported",
    }
    if response["format"] == "zip":
        response["archive_members"] = list_archive_members(target)
    return response


def download_dataset_url(
    dataset_dir: Path,
    url: str,
    *,
    suggested_name: str | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> dict[str, Any]:
    _validate_download_url(url)
    raw_dir = dataset_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(url)
    source_name = suggested_name or Path(parsed.path).name or _download_name_from_url(url)
    safe_name = _unique_safe_filename(raw_dir, source_name)
    target = raw_dir / safe_name

    request = urllib.request.Request(url, headers={"User-Agent": "AutoVibeGym/1.0"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            with target.open("wb") as fh:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"Download exceeds {max_bytes // (1024 * 1024)} MB limit for '{url}'."
                        )
                    fh.write(chunk)
    except urllib.error.URLError as exc:
        if target.exists():
            target.unlink(missing_ok=True)
        raise ValueError(f"Failed to download '{url}': {exc}") from exc
    except Exception:
        if target.exists():
            target.unlink(missing_ok=True)
        raise

    response = {
        "path": target.relative_to(dataset_dir).as_posix(),
        "name": target.name,
        "size": int(target.stat().st_size),
        "url": url,
        "format": infer_format(target.name) if is_supported_tabular_path(target.name) else "unsupported",
    }
    if response["format"] == "zip":
        response["archive_members"] = list_archive_members(target)
    return response


def load_raw_dataframe(dataset_dir: Path, config: DatasetConfig) -> pd.DataFrame:
    if config.ingestion.get("mode") not in {"raw", "", None}:
        raise ValueError("load_raw_dataframe only supports raw ingestion datasets.")
    tables, _, _ = _load_raw_tables(dataset_dir, config)
    if _uses_relations(config):
        joined, _, errors = _build_joined_dataframe(tables, config)
        if errors:
            raise ValueError("; ".join(errors))
        return joined
    frames = list(tables.values())
    if not frames:
        raise FileNotFoundError(f"No raw tables configured for {dataset_dir}")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False)


def preview_dataset_entry(
    dataset_dir: Path,
    *,
    logical_name: str | None = None,
    split_role: str | None = None,
    joined: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    config = load_dataset_config(dataset_dir)
    if joined:
        if config.ingestion.get("mode") != "raw":
            raise ValueError("Joined preview is only available for raw ingestion datasets.")
        tables, warnings, _ = _load_raw_tables(dataset_dir, config, preview_rows=limit)
        joined_df, join_diagnostics, errors = _build_joined_dataframe(tables, config)
        if errors:
            raise ValueError("; ".join(errors))
        joined_df, prep_info = apply_declared_preparation(joined_df, config, max_rows=None)
        preview = _frame_preview(joined_df.head(limit), target_col=config.task.get("target_col"))
        preview["warnings"].extend(warnings)
        preview["warnings"].extend(prep_info.get("warnings", []))
        preview["join_diagnostics"] = join_diagnostics
        preview["logical_name"] = "joined"
        preview["format"] = "joined"
        return preview

    entry = _find_entry(config, logical_name=logical_name, split_role=split_role)
    if entry is None:
        raise ValueError("Requested dataset file was not found in the config.")
    resolved, resolution_warnings = _resolve_file_entry(dataset_dir, entry)
    if resolved["format"] == "zip" and not entry.get("archive_member"):
        return {
            "logical_name": entry.get("logical_name") or entry.get("role") or Path(entry.get("path") or "").stem,
            "format": "zip",
            "path": entry.get("path"),
            "columns": [],
            "rows": [],
            "shape": {"rows": None, "cols": None},
            "dtypes": {},
            "missing_counts": {},
            "target_distribution": None,
            "warnings": resolution_warnings,
            "archive_members": list_archive_members(Path(resolved["path"])),
        }

    load_result = load_tabular_dataframe(
        Path(resolved["path"]),
        format_name=str(resolved["effective_format"]),
        read_options=dict(entry.get("read_options") or {}),
        nrows=limit,
        preview_limit_bytes=PREVIEW_FULL_READ_LIMIT_BYTES,
    )
    preview = _frame_preview(
        load_result.dataframe,
        target_col=config.task.get("target_col"),
        exact_rows_known=load_result.exact_rows_known,
    )
    preview.update(
        {
            "logical_name": entry.get("logical_name") or entry.get("role") or Path(entry.get("path") or "").stem,
            "format": load_result.format,
            "path": entry.get("path"),
            "archive_members": list_archive_members(Path(resolved["path"])) if load_result.format == "zip" else [],
        }
    )
    preview["warnings"].extend(resolution_warnings)
    preview["warnings"].extend(load_result.warnings)
    return preview


def validate_dataset_config(dataset_dir: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"files": {}, "joins": []}

    try:
        config = load_dataset_config(dataset_dir)
    except Exception as exc:
        errors.append(_issue(str(exc), field="config"))
        return {"ok": False, "errors": errors, "warnings": warnings, "diagnostics": diagnostics}

    mode = config.ingestion.get("mode")
    files = list(config.ingestion.get("files") or [])
    if mode not in {"raw", "pre_split"}:
        errors.append(_issue(f"Unsupported ingestion mode '{mode}'.", field="ingestion.mode"))
    if not files:
        errors.append(_issue("At least one ingestion file is required.", field="ingestion.files"))

    logical_names: set[str] = set()
    for index, entry in enumerate(files):
        logical_name = str(entry.get("logical_name") or "").strip()
        if logical_name:
            if logical_name in logical_names:
                errors.append(_issue("Logical table names must be unique.", field=f"ingestion.files[{index}].logical_name"))
            logical_names.add(logical_name)
        try:
            resolved, resolution_warnings = _resolve_file_entry(dataset_dir, entry)
            for warning in resolution_warnings:
                warnings.append(_issue(warning, field=f"ingestion.files[{index}]", logical_name=logical_name))
            diagnostics["files"][logical_name or entry.get("role") or f"file_{index + 1}"] = preview_dataset_entry(
                dataset_dir,
                logical_name=logical_name or None,
                split_role=str(entry.get("role") or "") or None,
                joined=False,
                limit=10,
            )
            if resolved["format"] == "zip" and not entry.get("archive_member"):
                members = diagnostics["files"][logical_name or entry.get("role") or f"file_{index + 1}"].get("archive_members") or []
                if len(members) != 1:
                    warnings.append(
                        _issue(
                            "ZIP source contains multiple tabular members; set archive_member explicitly for deterministic loading.",
                            field=f"ingestion.files[{index}].archive_member",
                            logical_name=logical_name,
                        )
                    )
        except Exception as exc:
            errors.append(_issue(str(exc), field=f"ingestion.files[{index}]", logical_name=logical_name))

    target_col = str(config.task.get("target_col") or "").strip()
    if not target_col:
        errors.append(_issue("task.target_col is required.", field="task.target_col"))
    if not config.task.get("metric"):
        errors.append(_issue("task.metric is required.", field="task.metric"))
    if not config.task.get("type"):
        errors.append(_issue("task.type is required.", field="task.type"))

    try:
        if mode == "raw":
            tables, load_warnings, _ = _load_raw_tables(dataset_dir, config)
            for warning in load_warnings:
                warnings.append(_issue(warning, field="ingestion.files"))
            frame = load_raw_dataframe(dataset_dir, config)
            frame, prep_info = apply_declared_preparation(frame, config, max_rows=None)
            for warning in prep_info.get("warnings", []):
                warnings.append(_issue(warning, field="preparation"))
            if target_col and target_col not in frame.columns:
                errors.append(_issue(f"Target column '{target_col}' is missing after preparation.", field="task.target_col"))
            if _uses_relations(config):
                _, join_diagnostics, join_errors = _build_joined_dataframe(tables, config)
                diagnostics["joins"] = join_diagnostics
                for join_error in join_errors:
                    errors.append(_issue(join_error, field="relations.joins"))
                for join_info in join_diagnostics:
                    ratio = float(join_info.get("row_growth_ratio") or 1.0)
                    if ratio >= JOIN_ROW_GROWTH_ERROR_RATIO:
                        errors.append(_issue(
                            f"Join {join_info['left_table']} -> {join_info['right_table']} multiplied rows by {ratio:.2f}x.",
                            field="relations.joins",
                        ))
                    elif ratio >= JOIN_ROW_GROWTH_WARN_RATIO:
                        warnings.append(_issue(
                            f"Join {join_info['left_table']} -> {join_info['right_table']} increased rows by {ratio:.2f}x.",
                            field="relations.joins",
                        ))
            _validate_split_config(config.split, mode="raw")
        elif mode == "pre_split":
            prepared = _load_pre_split_frames(dataset_dir, config)
            _validate_pre_split_frames(prepared, config)
        else:
            pass
    except Exception as exc:
        errors.append(_issue(str(exc), field="dataset"))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }


def prepare_dataset(dataset_dir: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    config = load_dataset_config(dataset_dir)
    if config_path(dataset_dir) is None:
        return {"dataset": dataset_dir.name, "status": "skipped", "reason": "legacy dataset"}

    mode = config.ingestion.get("mode")
    warnings: list[str] = []
    join_diagnostics: list[dict[str, Any]] = []

    if mode == "raw":
        tables, load_warnings, _ = _load_raw_tables(dataset_dir, config)
        warnings.extend(load_warnings)
        df = load_raw_dataframe(dataset_dir, config)
        n_rows_source = len(df)
        if _uses_relations(config):
            _, join_diagnostics, join_errors = _build_joined_dataframe(tables, config)
            if join_errors:
                raise ValueError("; ".join(join_errors))
            for join_info in join_diagnostics:
                ratio = float(join_info.get("row_growth_ratio") or 1.0)
                if ratio >= JOIN_ROW_GROWTH_ERROR_RATIO:
                    raise ValueError(
                        f"Join {join_info['left_table']} -> {join_info['right_table']} multiplied rows by {ratio:.2f}x."
                    )
                if ratio >= JOIN_ROW_GROWTH_WARN_RATIO:
                    warnings.append(
                        f"Join {join_info['left_table']} -> {join_info['right_table']} increased rows by {ratio:.2f}x."
                    )
        df, prep_info = apply_declared_preparation(df, config, max_rows=max_rows)
        warnings.extend(prep_info.get("warnings", []))
        train, val, test, split_info = _split_dataframe(df, config)
    elif mode == "pre_split":
        if max_rows is not None:
            raise ValueError("--max-rows is only supported for raw ingestion datasets.")
        prepared = _load_pre_split_frames(dataset_dir, config)
        n_rows_source = sum(len(frame) for frame in prepared.values())
        prepared = _apply_preparation_to_splits(prepared, config)
        _validate_pre_split_frames(prepared, config)
        train = prepared["train"]
        val = prepared["val"]
        test = prepared["test"]
        split_info = {
            "split_strategy": "pre_split",
            "create_val_from_train_if_missing": bool(config.split.get("create_val_from_train_if_missing")),
        }
        prep_info = {
            "sampled": False,
            "dropped_columns": _combined_drop_columns(config),
            "warnings": [],
        }
    else:
        raise ValueError(f"Unsupported ingestion mode: {mode}")

    target = str(config.task["target_col"])
    out_dir = dataset_dir / "prepared"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    feature_count = max(train.shape[1] - 1, 0)
    meta = {
        "name": config.name,
        "suite": config.suite,
        "source": config.source,
        "dataset_notes": config.dataset_notes,
        "notes": config.notes or config.dataset_notes,
        "ingestion": {
            "mode": mode,
            "files": _meta_file_entries(config.ingestion.get("files") or []),
        },
        "raw_files": [entry.get("path", "") for entry in config.ingestion.get("files") or []],
        "input_formats": {
            (entry.get("logical_name") or entry.get("role") or f"file_{idx + 1}"): infer_format(
                entry.get("archive_member") or entry.get("path") or "",
                str(entry.get("format") or "auto"),
            )
            for idx, entry in enumerate(config.ingestion.get("files") or [])
            if entry.get("path") or entry.get("archive_member")
        },
        "task_type": config.task.get("type"),
        "target_col": target,
        "metric": config.task.get("metric"),
        "split_strategy": config.split.get("strategy"),
        "seed": config.split.get("seed", 42),
        "n_rows_source": n_rows_source,
        "n_rows_prepared": int(len(train) + len(val) + len(test)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_features": int(feature_count),
        "role": config.role,
        "forbidden_columns": list(config.task.get("forbidden_columns") or []),
        "dropped_columns": prep_info.get("dropped_columns", []),
        "join_diagnostics": join_diagnostics,
        "warnings": warnings,
    }
    if config.task.get("type") == "classification":
        meta["n_classes"] = int(pd.concat([train[target], val[target], test[target]], ignore_index=True).nunique(dropna=False))
        meta["class_distribution"] = {
            "all": _normalized_value_counts(pd.concat([train[target], val[target], test[target]], ignore_index=True)),
            "train": _normalized_value_counts(train[target]),
            "val": _normalized_value_counts(val[target]),
            "test": _normalized_value_counts(test[target]),
        }
    meta.update({k: v for k, v in prep_info.items() if k != "warnings"})
    meta.update(split_info)
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "dataset": config.name,
        "status": "ok",
        "reason": "",
        "prepared_dir": str(out_dir),
        "meta": meta,
        "warnings": warnings,
    }


def raw_inputs_available(dataset_dir: Path, config: DatasetConfig) -> bool:
    for entry in config.ingestion.get("files") or []:
        try:
            _resolve_file_entry(dataset_dir, entry)
        except Exception:
            return False
    return True


def apply_declared_preparation(
    df: pd.DataFrame,
    config: DatasetConfig,
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {"sampled": False, "warnings": []}
    prep = config.preparation or {}

    if prep.get("deduplicate", False):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        info["deduplicated"] = True
        info["duplicates_removed"] = before - len(df)

    rename_columns = dict(prep.get("rename_columns") or {})
    if rename_columns:
        df = df.rename(columns=rename_columns)

    target_col = str(config.task["target_col"])
    declared_drop = list(prep.get("drop_columns") or [])
    forbidden_columns = [col for col in config.task.get("forbidden_columns", []) or [] if col != target_col]
    if target_col in (config.task.get("forbidden_columns") or []):
        info["warnings"].append(f"Ignored forbidden target column '{target_col}'.")
    drop_columns = list(dict.fromkeys(declared_drop + forbidden_columns))
    dropped_columns = [col for col in drop_columns if col in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    info["dropped_columns"] = dropped_columns

    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' after preparation in {config.name}")

    mapping = prep.get("target_mapping")
    if mapping:
        df[target_col] = df[target_col].map(mapping).fillna(df[target_col])

    sampling = prep.get("sampling")
    if max_rows is not None:
        if not sampling or not sampling.get("allowed", False):
            raise ValueError(f"--max-rows is not allowed for dataset '{config.name}'")
        if len(df) > max_rows:
            seed = int(sampling.get("seed", config.split.get("seed", 42)))
            stratify = _safe_stratify_target(df, target_col, config.task.get("type"))
            if stratify is not None:
                df, _ = train_test_split(
                    df,
                    train_size=max_rows,
                    random_state=seed,
                    stratify=stratify,
                )
            else:
                df, _ = train_test_split(df, train_size=max_rows, random_state=seed)
            df = df.reset_index(drop=True)
            info.update({"sampled": True, "max_rows": max_rows})
    return df, info


def split_temporal(df: pd.DataFrame, split_cfg: dict[str, Any]):
    _validate_fraction_triplet(
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )
    timestamp_cfg = split_cfg["timestamp"]
    cols = list(timestamp_cfg["source_columns"])
    fmt = timestamp_cfg.get("format")
    technical_col = "__technical_timestamp__"
    built = df[cols[0]].astype(str) if len(cols) == 1 else df[cols].astype(str).agg(" ".join, axis=1)
    ts = pd.to_datetime(built, format=fmt, errors="coerce")
    if ts.isna().any():
        raise ValueError("Temporal split failed: timestamp parsing produced NaT values")
    df = df.copy()
    df[technical_col] = ts
    df = df.sort_values(technical_col).reset_index(drop=True)
    n_rows = len(df)
    n_train = int(n_rows * float(split_cfg["train_fraction"]))
    n_val = int(n_rows * float(split_cfg["val_fraction"]))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    bounds = {
        "train_start": str(train[technical_col].min()),
        "train_end": str(train[technical_col].max()),
        "val_start": str(val[technical_col].min()),
        "val_end": str(val[technical_col].max()),
        "test_start": str(test[technical_col].min()),
        "test_end": str(test[technical_col].max()),
    }
    for chunk in (train, val, test):
        chunk.drop(columns=[technical_col], inplace=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), {"temporal_boundaries": bounds}


def _normalize_source(source: Any) -> dict[str, Any]:
    src = dict(source or {})
    return {
        "title": str(src.get("title") or src.get("name") or src.get("provider") or ""),
        "url": str(src.get("url") or ""),
        "license": str(src.get("license") or ""),
        "citation": str(src.get("citation") or src.get("reference") or ""),
        "description": str(src.get("description") or ""),
        **{k: v for k, v in src.items() if k not in {"title", "name", "provider", "url", "license", "citation", "reference", "description"}},
    }


def _normalize_dataset_notes(dataset_notes: Any, legacy_notes: dict[str, Any]) -> dict[str, Any]:
    src = dict(dataset_notes or {})
    warnings = src.get("warnings")
    pitfalls = src.get("known_pitfalls")
    return {
        "short_description": str(src.get("short_description") or legacy_notes.get("description") or legacy_notes.get("desc") or ""),
        "llm_context": str(src.get("llm_context") or ""),
        "warnings": list(warnings or []),
        "known_pitfalls": list(pitfalls or []),
    }


def _normalize_ingestion(ingestion: Any, raw_data: Any) -> dict[str, Any]:
    if isinstance(ingestion, dict):
        mode = str(ingestion.get("mode") or "raw")
        files = [_normalize_file_entry(entry, index) for index, entry in enumerate(ingestion.get("files") or [])]
        return {"mode": mode, "files": files}

    legacy = dict(raw_data or {})
    legacy_files = list(legacy.get("files") or [])
    legacy_format = str(legacy.get("format") or "auto")
    legacy_options = dict(legacy.get("read_options") or {})
    files = []
    for index, file_name in enumerate(legacy_files):
        relative = PurePosixPath(str(file_name)).as_posix()
        files.append(
            {
                "logical_name": "table_1" if len(legacy_files) == 1 else f"table_{index + 1}",
                "role": "base" if index == 0 else "table",
                "source_type": "local",
                "url": "",
                "path": f"raw_data/{relative}",
                "format": legacy_format,
                "read_options": legacy_options,
                "optional": False,
                "archive_member": "",
            }
        )
    return {"mode": "raw", "files": files}


def _normalize_file_entry(entry: Any, index: int) -> dict[str, Any]:
    src = dict(entry or {})
    path = str(src.get("path") or "")
    normalized_path = _normalize_storage_path(path).as_posix() if path else ""
    archive_member = str(src.get("archive_member") or "")
    return {
        "logical_name": str(src.get("logical_name") or f"table_{index + 1}"),
        "role": str(src.get("role") or ("base" if index == 0 else "table")),
        "source_type": str(src.get("source_type") or "upload"),
        "url": str(src.get("url") or ""),
        "path": normalized_path,
        "format": str(src.get("format") or "auto"),
        "read_options": dict(src.get("read_options") or {}),
        "optional": bool(src.get("optional", False)),
        "archive_member": archive_member,
    }


def _normalize_relations(relations: Any, ingestion: dict[str, Any]) -> dict[str, Any]:
    src = dict(relations or {})
    joins = []
    for join in src.get("joins") or []:
        item = dict(join or {})
        joins.append(
            {
                "left_table": str(item.get("left_table") or ""),
                "right_table": str(item.get("right_table") or ""),
                "how": str(item.get("how") or "left"),
                "left_on": [str(value) for value in item.get("left_on") or []],
                "right_on": [str(value) for value in item.get("right_on") or []],
            }
        )
    base_table = str(src.get("base_table") or "")
    if not base_table and ingestion.get("files"):
        base_table = str((ingestion.get("files") or [])[0].get("logical_name") or "table_1")
    return {"base_table": base_table, "joins": joins}


def _normalize_task(task: Any) -> dict[str, Any]:
    src = dict(task or {})
    return {
        "type": str(src.get("type") or "classification"),
        "target_col": str(src.get("target_col") or src.get("target") or ""),
        "metric": str(src.get("metric") or src.get("metric_name") or ""),
        "forbidden_columns": [str(value) for value in src.get("forbidden_columns") or []],
    }


def _normalize_split(split: Any) -> dict[str, Any]:
    src = dict(split or {})
    return {
        "strategy": str(src.get("strategy") or "stratified_random"),
        "seed": int(src.get("seed", 42)),
        "train_fraction": float(src.get("train_fraction", 0.7)),
        "val_fraction": float(src.get("val_fraction", 0.15)),
        "test_fraction": float(src.get("test_fraction", 0.15)),
        "timestamp": dict(src.get("timestamp") or {}),
        "create_val_from_train_if_missing": bool(src.get("create_val_from_train_if_missing", False)),
        "val_fraction_from_train": float(src.get("val_fraction_from_train", 0.15)),
    }


def _normalize_preparation(preparation: Any) -> dict[str, Any]:
    src = dict(preparation or {})
    return {
        "drop_columns": list(src.get("drop_columns") or []),
        "rename_columns": dict(src.get("rename_columns") or {}),
        "target_mapping": dict(src.get("target_mapping") or {}),
        "sampling": src.get("sampling"),
        "deduplicate": bool(src.get("deduplicate", False)),
    }


def _legacy_raw_data_view(raw_data: Any, ingestion: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_data, dict):
        return dict(raw_data)
    files = [str(PurePosixPath(entry["path"]).relative_to("raw_data")) for entry in ingestion.get("files") or [] if entry.get("path")]
    first = (ingestion.get("files") or [{}])[0]
    return {
        "files": files,
        "format": first.get("format", "auto"),
        "read_options": dict(first.get("read_options") or {}),
    }


def _find_entry(
    config: DatasetConfig,
    *,
    logical_name: str | None,
    split_role: str | None,
) -> dict[str, Any] | None:
    for entry in config.ingestion.get("files") or []:
        if logical_name and entry.get("logical_name") == logical_name:
            return entry
        if split_role and entry.get("role") == split_role:
            return entry
    return None


def _load_raw_tables(
    dataset_dir: Path,
    config: DatasetConfig,
    *,
    preview_rows: int | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str], list[dict[str, Any]]]:
    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for entry in config.ingestion.get("files") or []:
        logical_name = str(entry.get("logical_name") or entry.get("role") or f"table_{len(tables) + 1}")
        resolved, resolution_warnings = _resolve_file_entry(dataset_dir, entry)
        warnings.extend(resolution_warnings)
        if resolved["format"] == "zip" and not entry.get("archive_member"):
            members = list_archive_members(Path(resolved["path"]))
            if len(members) == 1:
                auto_entry = dict(entry)
                auto_entry["archive_member"] = members[0]["member"]
                resolved, resolution_warnings = _resolve_file_entry(dataset_dir, auto_entry)
                warnings.extend(resolution_warnings)
            else:
                raise ValueError(
                    f"ZIP source '{entry.get('path')}' for '{logical_name}' requires archive_member because it contains multiple supported tables."
                )
        result = load_tabular_dataframe(
            Path(resolved["path"]),
            format_name=str(resolved["effective_format"]),
            read_options=dict(entry.get("read_options") or {}),
            nrows=preview_rows,
            preview_limit_bytes=PREVIEW_FULL_READ_LIMIT_BYTES,
        )
        warnings.extend(result.warnings)
        tables[logical_name] = result.dataframe
        diagnostics.append(
            {
                "logical_name": logical_name,
                "path": entry.get("path"),
                "format": result.format,
                "rows": int(result.dataframe.shape[0]),
                "cols": int(result.dataframe.shape[1]),
            }
        )
    return tables, warnings, diagnostics


def _load_pre_split_frames(dataset_dir: Path, config: DatasetConfig) -> dict[str, pd.DataFrame]:
    roles: dict[str, pd.DataFrame] = {}
    optional_roles = {str(entry.get("role")) for entry in config.ingestion.get("files") or [] if entry.get("optional")}
    for role in ("train", "val", "test"):
        entry = _find_entry(config, logical_name=None, split_role=role)
        if entry is None:
            if role == "val" and config.split.get("create_val_from_train_if_missing"):
                continue
            if role == "val" or role in optional_roles:
                continue
            raise ValueError(f"Missing required pre-split file with role '{role}'.")
        resolved, _ = _resolve_file_entry(dataset_dir, entry)
        if resolved["format"] == "zip" and not entry.get("archive_member"):
            members = list_archive_members(Path(resolved["path"]))
            if len(members) != 1:
                raise ValueError(
                    f"ZIP source '{entry.get('path')}' for role '{role}' requires archive_member."
                )
            entry = {**entry, "archive_member": members[0]["member"]}
            resolved, _ = _resolve_file_entry(dataset_dir, entry)
        result = load_tabular_dataframe(
            Path(resolved["path"]),
            format_name=str(resolved["effective_format"]),
            read_options=dict(entry.get("read_options") or {}),
        )
        roles[role] = result.dataframe
    if "val" not in roles and config.split.get("create_val_from_train_if_missing"):
        val_fraction = float(config.split.get("val_fraction_from_train", 0.15))
        if val_fraction <= 0 or val_fraction >= 1:
            raise ValueError("split.val_fraction_from_train must be in (0, 1).")
        target_col = str(config.task["target_col"])
        train_df = roles["train"]
        stratify = _safe_stratify_target(train_df, target_col, config.task.get("type"))
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_fraction,
            random_state=int(config.split.get("seed", 42)),
            stratify=stratify,
        )
        roles["train"] = train_df.reset_index(drop=True)
        roles["val"] = val_df.reset_index(drop=True)
    return roles


def _apply_preparation_to_splits(frames: dict[str, pd.DataFrame], config: DatasetConfig) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for role, frame in frames.items():
        prepared_frame, _ = apply_declared_preparation(frame.copy(), config, max_rows=None)
        prepared[role] = prepared_frame.reset_index(drop=True)
    return prepared


def _validate_pre_split_frames(frames: dict[str, pd.DataFrame], config: DatasetConfig) -> None:
    target_col = str(config.task["target_col"])
    if "train" not in frames or "test" not in frames or "val" not in frames:
        raise ValueError("Pre-split datasets require train, test, and val (or create_val_from_train_if_missing).")
    train_columns = list(frames["train"].columns)
    for role, frame in frames.items():
        if target_col not in frame.columns:
            raise ValueError(f"Target column '{target_col}' is missing from the {role} split.")
        if list(frame.columns) != train_columns:
            if set(frame.columns) != set(train_columns):
                raise ValueError(f"Split schemas differ between train and {role}.")
            frames[role] = frame[train_columns]


def _build_joined_dataframe(
    tables: dict[str, pd.DataFrame],
    config: DatasetConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    diagnostics: list[dict[str, Any]] = []
    errors: list[str] = []
    relations = config.relations or {}
    base_name = str(relations.get("base_table") or "")
    if not base_name:
        if len(tables) == 1:
            only_name = next(iter(tables))
            return tables[only_name].copy(), diagnostics, errors
        errors.append("relations.base_table is required for multi-table raw datasets.")
        return pd.DataFrame(), diagnostics, errors
    if base_name not in tables:
        errors.append(f"Base table '{base_name}' is not present in ingestion.files.")
        return pd.DataFrame(), diagnostics, errors

    frame = tables[base_name].copy()
    joins = relations.get("joins") or []
    for join in joins:
        left_table = str(join.get("left_table") or "")
        right_table = str(join.get("right_table") or "")
        left_on = [str(value) for value in join.get("left_on") or []]
        right_on = [str(value) for value in join.get("right_on") or []]
        how = str(join.get("how") or "left")
        if left_table != base_name:
            errors.append(
                f"Current implementation expects joins to start from base_table '{base_name}', got left_table '{left_table}'."
            )
            continue
        if right_table not in tables:
            errors.append(f"Right table '{right_table}' is not present in ingestion.files.")
            continue
        missing_left = [col for col in left_on if col not in frame.columns]
        missing_right = [col for col in right_on if col not in tables[right_table].columns]
        if missing_left:
            errors.append(f"Join keys {missing_left} are missing from left table '{left_table}'.")
            continue
        if missing_right:
            errors.append(f"Join keys {missing_right} are missing from right table '{right_table}'.")
            continue
        before_rows = len(frame)
        right = tables[right_table].copy()
        frame = frame.merge(
            right,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffixes=("", f"__{right_table}"),
        )
        after_rows = len(frame)
        diagnostics.append(
            {
                "left_table": left_table,
                "right_table": right_table,
                "how": how,
                "left_on": left_on,
                "right_on": right_on,
                "rows_before": int(before_rows),
                "rows_after": int(after_rows),
                "row_growth_ratio": round(after_rows / max(before_rows, 1), 6),
                "right_rows": int(len(right)),
            }
        )
    return frame, diagnostics, errors


def _resolve_file_entry(dataset_dir: Path, entry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    path_text = str(entry.get("path") or "").strip()
    if not path_text:
        raise ValueError(f"Entry '{entry.get('logical_name') or entry.get('role')}' is missing path.")
    normalized_path = _normalize_storage_path(path_text)
    absolute_path = (dataset_dir / normalized_path).resolve()
    try:
        absolute_path.relative_to(dataset_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe dataset path '{path_text}'.") from exc

    warnings: list[str] = []
    archive_member = str(entry.get("archive_member") or "").strip()
    if archive_member:
        if not is_safe_archive_member(archive_member):
            raise ValueError(f"Unsafe archive member path '{archive_member}'.")
        if not absolute_path.exists():
            raise FileNotFoundError(f"Missing archive file: {absolute_path}")
        if infer_format(absolute_path.name, str(entry.get("format") or "auto")) != "zip":
            raise ValueError("archive_member can only be used with ZIP sources.")
        extracted = _extract_archive_member(dataset_dir, absolute_path, archive_member)
        fmt = infer_format(archive_member, "auto")
        return {
            "path": str(extracted),
            "effective_format": fmt,
            "format": "zip",
            "archive_member": archive_member,
        }, warnings

    if absolute_path.exists():
        fmt = infer_format(absolute_path.name, str(entry.get("format") or "auto"))
        return {
            "path": str(absolute_path),
            "effective_format": fmt,
            "format": fmt,
            "archive_member": "",
        }, warnings

    if normalized_path.parts and normalized_path.parts[0] == "raw_data":
        member_name = PurePosixPath(*normalized_path.parts[1:]).as_posix()
        extracted = _extract_legacy_member_from_any_zip(dataset_dir / "raw_data", member_name)
        if extracted is not None:
            warnings.append(
                f"Materialized '{member_name}' from a ZIP archive in raw_data/ for compatibility."
            )
            fmt = infer_format(extracted.name, str(entry.get("format") or "auto"))
            return {
                "path": str(extracted),
                "effective_format": fmt,
                "format": fmt,
                "archive_member": "",
            }, warnings

    raise FileNotFoundError(f"Missing dataset file: {normalized_path.as_posix()}")


def _extract_archive_member(dataset_dir: Path, archive_path: Path, member_name: str) -> Path:
    if not is_safe_archive_member(member_name):
        raise ValueError(f"Unsafe zip member path: {member_name}")
    target_root = dataset_dir / "raw_data" / "_zip_cache" / archive_path.stem
    target = (target_root / PurePosixPath(member_name)).resolve()
    try:
        target.relative_to(target_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe zip member path: {member_name}") from exc
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = {PurePosixPath(info.filename).as_posix(): info for info in archive.infolist() if not info.is_dir()}
        if member_name not in members:
            raise FileNotFoundError(f"ZIP source '{archive_path.name}' does not contain '{member_name}'.")
        archive.extract(members[member_name], target_root)
    return target


def _extract_legacy_member_from_any_zip(raw_dir: Path, member_name: str) -> Path | None:
    if not raw_dir.exists():
        return None
    for archive_path in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            members = {PurePosixPath(info.filename).as_posix(): info for info in archive.infolist() if not info.is_dir()}
            if member_name not in members:
                continue
            target = (raw_dir / PurePosixPath(member_name)).resolve()
            try:
                target.relative_to(raw_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {member_name}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            archive.extract(members[member_name], raw_dir)
            return target
    return None


def _frame_preview(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    exact_rows_known: bool = False,
) -> dict[str, Any]:
    preview_df = df.copy()
    return {
        "columns": [str(col) for col in preview_df.columns],
        "rows": preview_df.where(pd.notnull(preview_df), None).values.tolist(),
        "shape": {
            "rows": int(preview_df.shape[0]) if exact_rows_known else None,
            "cols": int(preview_df.shape[1]),
        },
        "shape_is_approximate": not exact_rows_known,
        "dtypes": {str(col): str(dtype) for col, dtype in preview_df.dtypes.items()},
        "missing_counts": {str(col): int(count) for col, count in preview_df.isna().sum().items()},
        "target_distribution": _preview_target_distribution(preview_df, target_col),
        "warnings": [],
    }


def _preview_target_distribution(df: pd.DataFrame, target_col: str | None) -> dict[str, Any] | None:
    if not target_col or target_col not in df.columns:
        return None
    counts = df[target_col].value_counts(dropna=False).head(20)
    total = max(len(df), 1)
    return {str(key): {"count": int(value), "pct": round(float(value) / total, 4)} for key, value in counts.items()}


def _combined_drop_columns(config: DatasetConfig) -> list[str]:
    target_col = str(config.task.get("target_col") or "")
    declared = list(config.preparation.get("drop_columns") or [])
    forbidden = [col for col in config.task.get("forbidden_columns", []) or [] if col != target_col]
    return list(dict.fromkeys(declared + forbidden))


def _split_dataframe(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_cfg = config.split
    target_col = str(config.task["target_col"])
    strategy = str(split_cfg.get("strategy") or "stratified_random")
    if strategy == "stratified_random":
        _validate_split_config(split_cfg, mode="raw")
        return _split_random(df, target_col, config.task.get("type"), split_cfg)
    if strategy == "temporal":
        return split_temporal(df, split_cfg)
    raise ValueError(f"Unsupported split strategy: {strategy}")


def _split_random(
    df: pd.DataFrame,
    target_col: str,
    task_type: str | None,
    split_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_fraction = float(split_cfg["train_fraction"])
    val_fraction = float(split_cfg["val_fraction"])
    test_fraction = float(split_cfg["test_fraction"])
    _validate_fraction_triplet(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    seed = int(split_cfg.get("seed", 42))
    stratify = _safe_stratify_target(df, target_col, task_type)
    train, temp = train_test_split(
        df,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    relative_test_fraction = test_fraction / max(val_fraction + test_fraction, FRACTION_TOLERANCE)
    stratify_temp = _safe_stratify_target(temp, target_col, task_type) if stratify is not None else None
    val, test = train_test_split(
        temp,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=stratify_temp,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        {},
    )


def _validate_split_config(split_cfg: dict[str, Any], *, mode: str) -> None:
    strategy = str(split_cfg.get("strategy") or "")
    if mode == "raw" and strategy not in {"stratified_random", "temporal"}:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    if strategy in {"stratified_random", "temporal"}:
        _validate_fraction_triplet(
            train_fraction=float(split_cfg["train_fraction"]),
            val_fraction=float(split_cfg["val_fraction"]),
            test_fraction=float(split_cfg["test_fraction"]),
        )
    if strategy == "temporal":
        timestamp = dict(split_cfg.get("timestamp") or {})
        if not timestamp.get("source_columns"):
            raise ValueError("Temporal splits require split.timestamp.source_columns.")


def _validate_fraction_triplet(*, train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    fractions = [train_fraction, val_fraction, test_fraction]
    if any(value < 0 or value > 1 for value in fractions):
        raise ValueError("Split fractions must be in [0, 1].")
    total = sum(fractions)
    if abs(total - 1.0) > FRACTION_TOLERANCE:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.6f} "
            f"(train={train_fraction}, val={val_fraction}, test={test_fraction})."
        )


def _safe_stratify_target(
    df: pd.DataFrame,
    target_col: str,
    task_type: str | None,
) -> pd.Series | None:
    if str(task_type or "").lower() != "classification":
        return None
    if target_col not in df.columns or len(df) < 4:
        return None
    counts = df[target_col].value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return None
    return df[target_col]


def _normalized_value_counts(series: pd.Series) -> dict[str, float]:
    vc = series.value_counts(normalize=True, dropna=False)
    return {str(key): float(value) for key, value in vc.items()}


def _meta_file_entries(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in files:
        result.append(
            {
                "logical_name": entry.get("logical_name"),
                "role": entry.get("role"),
                "source_type": entry.get("source_type"),
                "url": entry.get("url"),
                "path": entry.get("path"),
                "format": entry.get("format"),
                "archive_member": entry.get("archive_member") or None,
            }
        )
    return result


def _uses_relations(config: DatasetConfig) -> bool:
    joins = (config.relations or {}).get("joins") or []
    return bool(joins)


def _normalize_storage_path(path_text: str) -> Path:
    raw = PurePosixPath(path_text)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"Unsafe relative path '{path_text}'.")
    if raw.parts and raw.parts[0] in {"raw_data", "prepared"}:
        normalized = raw
    else:
        normalized = PurePosixPath("raw_data") / raw
    return Path(normalized.as_posix())


def _issue(message: str, *, field: str | None = None, logical_name: str | None = None) -> dict[str, Any]:
    payload = {"message": message}
    if field:
        payload["field"] = field
    if logical_name:
        payload["logical_name"] = logical_name
    return payload


def _unique_safe_filename(root: Path, file_name: str) -> str:
    base = Path(file_name).name or "file"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base).strip("._")
    if not safe:
        safe = "file"
    candidate = safe
    stem = Path(safe).stem
    suffix = "".join(Path(safe).suffixes)
    counter = 2
    while (root / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _download_name_from_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    parsed = urllib.parse.urlparse(url)
    suffix = "".join(Path(parsed.path).suffixes)
    return f"download_{digest}{suffix}"


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Only http:// and https:// dataset URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("Dataset URL is missing a hostname.")
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        raise ValueError("Local or private network dataset URLs are not allowed.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve '{host}': {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("Local or private network dataset URLs are not allowed.")
