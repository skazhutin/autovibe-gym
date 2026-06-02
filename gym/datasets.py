import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split


MetricFn = Callable[[pd.Series, pd.Series], float]


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    target_col: str
    metric_name: str | None = None
    task_type: str | None = None
    source: dict | str | None = None
    split_strategy: str | None = None
    role: str | None = None
    sampled: bool | None = None
    seed: int = 42
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, fallback_name: str) -> "DatasetMetadata":
        target_col = data.get("target_col") or data.get("target")
        if not target_col:
            raise ValueError("Dataset metadata must define 'target_col'.")
        return cls(
            name=str(data.get("name") or fallback_name),
            target_col=str(target_col),
            metric_name=data.get("metric_name") or data.get("metric"),
            task_type=data.get("task_type"),
            source=data.get("source"),
            split_strategy=data.get("split_strategy"),
            role=data.get("role"),
            sampled=data.get("sampled"),
            seed=int(data.get("seed", 42)),
            notes=dict(data.get("notes") or {}),
        )


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: DatasetMetadata

    @property
    def target_col(self) -> str:
        return self.metadata.target_col


def load_dataset_splits(
    *,
    dataset: str | None = None,
    dataset_dir: str | None = None,
    target_col: str | None = None,
    seed: int = 42,
) -> DatasetSplits:
    if bool(dataset) == bool(dataset_dir):
        raise ValueError("Pass exactly one of 'dataset' or 'dataset_dir'.")
    if dataset_dir:
        return load_splits_from_dir(dataset_dir)
    if not target_col:
        raise ValueError("'target_col' is required when loading a single CSV dataset.")
    return load_splits_from_csv(dataset or "", target_col=target_col, seed=seed)


def load_splits_from_csv(path: str, *, target_col: str, seed: int = 42) -> DatasetSplits:
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path)
    train, temp = train_test_split(df, test_size=0.3, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    metadata = DatasetMetadata(
        name=dataset_path.stem,
        target_col=target_col,
        seed=seed,
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        metadata=metadata,
    )


def load_splits_from_dir(dataset_dir: str) -> DatasetSplits:
    root = Path(dataset_dir)
    if (root / "prepared").exists() and not (root / "meta.json").exists():
        root = root / "prepared"
    metadata_path = root / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

    metadata = DatasetMetadata.from_dict(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        fallback_name=root.name,
    )
    return DatasetSplits(
        train=pd.read_csv(root / "train.csv"),
        val=pd.read_csv(root / "val.csv"),
        test=pd.read_csv(root / "test.csv"),
        metadata=metadata,
    )


def resolve_metric(metadata: DatasetMetadata, target_series: pd.Series) -> tuple[MetricFn, str]:
    if metadata.metric_name:
        return metric_from_name(metadata.metric_name), metadata.metric_name
    return infer_metric(target_series)


def infer_metric(target_series: pd.Series) -> tuple[MetricFn, str]:
    if target_series.nunique() <= 10:
        return (lambda y, p: f1_score(y, p, average="weighted", zero_division=0)), "f1_weighted"
    return (lambda y, p: -mean_squared_error(y, p) ** 0.5), "neg_rmse"


def metric_from_name(metric_name: str) -> MetricFn:
    normalized = metric_name.strip().lower()
    if normalized == "f1_weighted":
        return lambda y, p: f1_score(y, p, average="weighted", zero_division=0)
    if normalized == "f1_macro":
        return lambda y, p: f1_score(y, p, average="macro", zero_division=0)
    if normalized == "neg_rmse":
        return lambda y, p: -mean_squared_error(y, p) ** 0.5
    raise ValueError(f"Unsupported metric: {metric_name}")
