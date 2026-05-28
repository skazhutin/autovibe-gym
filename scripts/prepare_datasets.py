import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS_ROOT = Path("datasets")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    suite: str
    source: dict[str, Any]
    raw_data: dict[str, Any]
    task: dict[str, Any]
    split: dict[str, Any]
    preparation: dict[str, Any]
    role: str | None
    notes: dict[str, Any]


def discover_dataset_dirs(datasets_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    if not datasets_root.exists():
        return discovered
    for child in datasets_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "config.json").exists():
            discovered[child.name] = child
        elif (child / "meta.json").exists() and (child / "train.csv").exists():
            discovered[child.name] = child
    return dict(sorted(discovered.items()))


def load_dataset_config(dataset_dir: Path) -> DatasetConfig:
    cfg_path = dataset_dir / "config.json"
    if not cfg_path.exists():
        # legacy
        meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
        return DatasetConfig(
            name=meta.get("name", dataset_dir.name),
            suite="legacy",
            source={"type": "legacy"},
            raw_data={"files": []},
            task={"type": meta.get("task_type", "classification"), "target_col": meta["target_col"], "metric": meta.get("metric", "f1_weighted")},
            split={"strategy": meta.get("split_strategy", "stratified_random"), "seed": meta.get("seed", 42), "train_fraction": 0.7, "val_fraction": 0.15, "test_fraction": 0.15},
            preparation={},
            role=meta.get("role"),
            notes=meta.get("notes", {}),
        )
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return DatasetConfig(
        name=raw["name"], suite=raw.get("suite", "example_datasets"), source=raw.get("source", {}),
        raw_data=raw["raw_data"], task=raw["task"], split=raw["split"], preparation=raw.get("preparation", {}), role=raw.get("role"), notes=raw.get("notes", {}),
    )


def load_raw_dataframe(dataset_dir: Path, config: DatasetConfig) -> pd.DataFrame:
    files = config.raw_data.get("files", [])
    if not files:
        raise FileNotFoundError(f"No raw_data.files in config for {dataset_dir}")
    frames = []
    for file_name in files:
        path = dataset_dir / "raw_data" / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        opts = config.raw_data.get("read_options", {})
        fmt = config.raw_data.get("format", "csv")
        if fmt == "csv":
            frames.append(pd.read_csv(path, **opts))
        elif fmt in {"xlsx", "excel"}:
            frames.append(pd.read_excel(path, **opts))
        else:
            raise ValueError(f"Unsupported raw format: {fmt}")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def apply_declared_preparation(df: pd.DataFrame, config: DatasetConfig, *, max_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = {"sampled": False}
    prep = config.preparation or {}
    target_col = config.task["target_col"]

    mapping = prep.get("target_mapping")
    if mapping:
        df[target_col] = df[target_col].map(mapping)

    rename_columns = prep.get("rename_columns") or {}
    if rename_columns:
        df = df.rename(columns=rename_columns)

    drop_columns = prep.get("drop_columns") or []
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    sampling = prep.get("sampling")
    if max_rows is not None:
        if not sampling or not sampling.get("allowed", False):
            raise ValueError(f"--max-rows is not allowed for dataset '{config.name}'")
        if len(df) > max_rows:
            seed = sampling.get("seed", config.split.get("seed", 42))
            if config.task.get("type") == "classification":
                df, _ = train_test_split(df, train_size=max_rows, random_state=seed, stratify=df[target_col])
            else:
                df, _ = train_test_split(df, train_size=max_rows, random_state=seed)
            df = df.reset_index(drop=True)
            info.update({"sampled": True, "max_rows": max_rows})
    return df, info


def split_stratified_random(df: pd.DataFrame, target_col: str, seed: int, train_fraction: float, val_fraction: float, test_fraction: float):
    train, temp = train_test_split(df, test_size=val_fraction + test_fraction, random_state=seed, stratify=df[target_col])
    relative_test_fraction = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(temp, test_size=relative_test_fraction, random_state=seed, stratify=temp[target_col])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), {}


def split_temporal(df: pd.DataFrame, split_cfg: dict[str, Any]):
    timestamp_cfg = split_cfg["timestamp"]
    cols = timestamp_cfg["source_columns"]
    fmt = timestamp_cfg.get("format")
    technical_col = "__technical_timestamp__"
    if len(cols) == 1:
        built = df[cols[0]].astype(str)
    else:
        built = df[cols].astype(str).agg(" ".join, axis=1)
    ts = pd.to_datetime(built, format=fmt, errors="coerce")
    if ts.isna().any():
        raise ValueError("Temporal split failed: timestamp parsing produced NaT values")
    df = df.copy()
    df[technical_col] = ts
    df = df.sort_values(technical_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * split_cfg["train_fraction"])
    n_val = int(n * split_cfg["val_fraction"])
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    bounds = {"train_start": str(train[technical_col].min()), "train_end": str(train[technical_col].max()), "val_start": str(val[technical_col].min()), "val_end": str(val[technical_col].max()), "test_start": str(test[technical_col].min()), "test_end": str(test[technical_col].max())}
    for chunk in (train, val, test):
        chunk.drop(columns=[technical_col], inplace=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), {"temporal_boundaries": bounds}


def prepare_dataset(dataset_dir: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    config = load_dataset_config(dataset_dir)
    if not (dataset_dir / "config.json").exists():
        return {"dataset": dataset_dir.name, "status": "skipped", "reason": "legacy dataset"}
    df = load_raw_dataframe(dataset_dir, config)
    n_rows_source = len(df)
    if config.task["target_col"] not in df.columns:
        raise ValueError(f"Missing target column '{config.task['target_col']}' in {dataset_dir}")
    df, prep_info = apply_declared_preparation(df, config, max_rows=max_rows)
    split_cfg = config.split
    if split_cfg["strategy"] == "stratified_random":
        train, val, test, split_info = split_stratified_random(df, config.task["target_col"], split_cfg.get("seed", 42), split_cfg["train_fraction"], split_cfg["val_fraction"], split_cfg["test_fraction"])
    elif split_cfg["strategy"] == "temporal":
        train, val, test, split_info = split_temporal(df, split_cfg)
    else:
        raise ValueError(f"Unsupported split strategy: {split_cfg['strategy']}")
    out_dir = dataset_dir / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    target = config.task["target_col"]
    meta = {
        "name": config.name, "suite": config.suite, "source": config.source, "raw_files": config.raw_data.get("files", []),
        "task_type": config.task.get("type"), "target_col": target, "metric": config.task.get("metric"),
        "split_strategy": split_cfg.get("strategy"), "seed": split_cfg.get("seed", 42),
        "n_rows_source": n_rows_source, "n_rows_prepared": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": df.shape[1] - 1, "n_classes": int(df[target].nunique()),
        "class_distribution": {"all": df[target].value_counts(normalize=True).to_dict(), "train": train[target].value_counts(normalize=True).to_dict(), "val": val[target].value_counts(normalize=True).to_dict(), "test": test[target].value_counts(normalize=True).to_dict()},
        "sampled": prep_info.get("sampled", False), "role": config.role, "notes": config.notes,
    }
    meta.update(prep_info)
    meta.update(split_info)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dataset": config.name, "status": "ok", "reason": "", "prepared_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--suite")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    discovered = discover_dataset_dirs(DATASETS_ROOT)

    if args.list:
        print("name | suite | task_type | metric | split_strategy | raw_data_status | prepared_status | role")
        for name, ds_dir in discovered.items():
            if (ds_dir / "config.json").exists():
                cfg = load_dataset_config(ds_dir)
                raw_ok = all((ds_dir / "raw_data" / f).exists() for f in cfg.raw_data.get("files", [])) if cfg.raw_data.get("files") else False
                prepared_ok = (ds_dir / "prepared" / "meta.json").exists() or (ds_dir / "meta.json").exists()
                print(f"{name} | {cfg.suite} | {cfg.task.get('type')} | {cfg.task.get('metric')} | {cfg.split.get('strategy')} | {'ok' if raw_ok else 'missing'} | {'ok' if prepared_ok else 'missing'} | {cfg.role}")
            else:
                meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
                print(f"{name} | legacy | {meta.get('task_type','classification')} | {meta.get('metric')} | {meta.get('split_strategy','fixed')} | n/a | ok | {meta.get('role')}")
        return

    to_process: list[Path]
    if args.dataset:
        to_process = [DATASETS_ROOT / args.dataset]
    elif args.suite:
        to_process = [d for _, d in discovered.items() if (load_dataset_config(d).suite == args.suite)]
    else:
        to_process = [d for _, d in discovered.items() if (d / "config.json").exists()]

    summary = []
    for ds_dir in to_process:
        try:
            res = prepare_dataset(ds_dir, max_rows=args.max_rows)
        except Exception as exc:
            res = {"dataset": ds_dir.name, "status": "error", "reason": str(exc), "prepared_dir": str(ds_dir / 'prepared')}
        summary.append(res)

    print("dataset | status | reason | prepared_dir")
    for row in summary:
        print(f"{row['dataset']} | {row['status']} | {row.get('reason','')} | {row.get('prepared_dir','')}")


if __name__ == "__main__":
    main()
