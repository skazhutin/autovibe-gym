"""Create immutable Paper V1 dataset snapshots and FANU reference values.

The command is intentionally result-blind with respect to the agent study.  It
materializes the already selected train/validation/test splits, applies only a
predeclared target transform, and evaluates one fixed dummy/reference pair.
Output paths are exclusive: an existing bundle is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from research.run_artifacts import canonical_hash, file_set_hash


FREEZE_SCHEMA_VERSION = "1.0"
REQUIRED_SPLIT_FILES = ("train.csv", "val.csv", "test.csv", "meta.json")


class FreezeInputError(ValueError):
    """Raised when selected inputs cannot satisfy the freeze contract."""


@dataclass(frozen=True)
class DatasetFreezeSpec:
    dataset_id: str
    source_relative_path: str
    target_col: str
    task_type: str
    metric: str
    metric_direction: str
    split_id: str
    split_seed: int
    target_transform: str
    drop_columns: list[str]
    split_policy: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetFreezeSpec":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != required:
            raise FreezeInputError(
                f"dataset freeze spec fields differ: expected {sorted(required)}, got {sorted(value)}"
            )
        spec = cls(**value)
        if spec.metric_direction not in {"higher", "lower"}:
            raise FreezeInputError(f"unsupported metric direction for {spec.dataset_id}")
        if spec.target_transform not in {"identity", "non_no_readmission_binary"}:
            raise FreezeInputError(f"unsupported target transform for {spec.dataset_id}")
        if not isinstance(spec.drop_columns, list) or any(
            not isinstance(column, str) for column in spec.drop_columns
        ):
            raise FreezeInputError(f"invalid drop_columns for {spec.dataset_id}")
        policy_type = spec.split_policy.get("type")
        if policy_type not in {"preserve_source", "group_shuffle"}:
            raise FreezeInputError(f"unsupported split policy for {spec.dataset_id}")
        if policy_type == "group_shuffle":
            required_policy = {
                "type",
                "group_column",
                "train_fraction",
                "validation_fraction",
                "test_fraction",
            }
            if set(spec.split_policy) != required_policy:
                raise FreezeInputError(f"invalid group split policy for {spec.dataset_id}")
            fractions = [
                float(spec.split_policy[key])
                for key in ("train_fraction", "validation_fraction", "test_fraction")
            ]
            if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
                raise FreezeInputError(f"group split fractions must sum to one for {spec.dataset_id}")
            if spec.split_policy["group_column"] not in spec.drop_columns:
                raise FreezeInputError(
                    f"group column must be removed from frozen features for {spec.dataset_id}"
                )
        return spec


def load_freeze_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise FreezeInputError("unsupported freeze source config")
    if not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise FreezeInputError("freeze source config has no datasets")
    for dataset in data["datasets"]:
        DatasetFreezeSpec.from_mapping(dataset)
    return data


def _source_paths(root: Path) -> list[Path]:
    paths = [root / name for name in REQUIRED_SPLIT_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FreezeInputError(f"missing source artifacts: {', '.join(missing)}")
    return paths


def _transform_target(frame: pd.DataFrame, spec: DatasetFreezeSpec) -> pd.DataFrame:
    if spec.target_col not in frame.columns:
        raise FreezeInputError(f"{spec.dataset_id} is missing target {spec.target_col!r}")
    result = frame.copy()
    if spec.target_transform == "non_no_readmission_binary":
        result[spec.target_col] = (result[spec.target_col].astype(str) != "NO").astype(int)
    return result


def _source_frames(source: Path) -> dict[str, pd.DataFrame]:
    return {
        split_name: pd.read_csv(source / f"{split_name}.csv")
        for split_name in ("train", "val", "test")
    }


def _apply_split_policy(
    frames: Mapping[str, pd.DataFrame], spec: DatasetFreezeSpec
) -> dict[str, pd.DataFrame]:
    if spec.split_policy["type"] == "preserve_source":
        return {name: frame.copy() for name, frame in frames.items()}

    combined = pd.concat(
        [frames[name] for name in ("train", "val", "test")],
        ignore_index=True,
    )
    group_column = str(spec.split_policy["group_column"])
    if group_column not in combined.columns:
        raise FreezeInputError(f"{spec.dataset_id} is missing group column {group_column!r}")
    train_fraction = float(spec.split_policy["train_fraction"])
    validation_fraction = float(spec.split_policy["validation_fraction"])
    test_fraction = float(spec.split_policy["test_fraction"])
    first = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=spec.split_seed,
    )
    train_indices, remainder_indices = next(
        first.split(combined, groups=combined[group_column])
    )
    remainder = combined.iloc[remainder_indices]
    second = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_fraction / (validation_fraction + test_fraction),
        random_state=spec.split_seed + 1,
    )
    validation_relative, test_relative = next(
        second.split(remainder, groups=remainder[group_column])
    )
    return {
        "train": combined.iloc[train_indices].sort_index().reset_index(drop=True),
        "val": remainder.iloc[validation_relative].sort_index().reset_index(drop=True),
        "test": remainder.iloc[test_relative].sort_index().reset_index(drop=True),
    }


def _prepare_frame(frame: pd.DataFrame, spec: DatasetFreezeSpec) -> pd.DataFrame:
    transformed = _transform_target(frame, spec)
    missing = [column for column in spec.drop_columns if column not in transformed.columns]
    if missing:
        raise FreezeInputError(
            f"{spec.dataset_id} is missing columns selected for removal: {missing}"
        )
    return transformed.drop(columns=spec.drop_columns)


def _group_split_audit(
    frames: Mapping[str, pd.DataFrame], spec: DatasetFreezeSpec
) -> dict[str, Any] | None:
    if spec.split_policy["type"] != "group_shuffle":
        return None
    group_column = str(spec.split_policy["group_column"])
    groups = {
        name: set(frame[group_column].astype(str))
        for name, frame in frames.items()
    }
    overlaps = {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise FreezeInputError(f"group leakage remained after splitting {spec.dataset_id}")
    return {
        "group_column_removed": group_column in spec.drop_columns,
        "unique_groups": {name: len(values) for name, values in groups.items()},
        "pairwise_group_overlap": overlaps,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _materialize_dataset(
    spec: DatasetFreezeSpec,
    *,
    datasets_root: Path,
    bundle_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source = (datasets_root / spec.source_relative_path).resolve()
    source_paths = _source_paths(source)
    destination = bundle_root / "datasets" / spec.dataset_id
    destination.mkdir(parents=True, exist_ok=False)

    source_meta = json.loads((source / "meta.json").read_text(encoding="utf-8-sig"))
    source_frames = _source_frames(source)
    frozen_frames = _apply_split_policy(source_frames, spec)
    group_split_audit = _group_split_audit(frozen_frames, spec)
    sizes: dict[str, int] = {}
    for split_name, source_frame in frozen_frames.items():
        destination_csv = destination / f"{split_name}.csv"
        frame = _prepare_frame(source_frame, spec)
        if (
            spec.target_transform == "identity"
            and not spec.drop_columns
            and spec.split_policy["type"] == "preserve_source"
        ):
            shutil.copyfile(source / f"{split_name}.csv", destination_csv)
        else:
            frame.to_csv(destination_csv, index=False, lineterminator="\n")
        sizes[split_name] = len(frame)

    frozen_meta = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "dataset_id": spec.dataset_id,
        "target_col": spec.target_col,
        "task_type": spec.task_type,
        "metric": spec.metric,
        "metric_direction": spec.metric_direction,
        "split_id": spec.split_id,
        "split_seed": spec.split_seed,
        "target_transform": spec.target_transform,
        "drop_columns": spec.drop_columns,
        "split_policy": spec.split_policy,
        "group_split_audit": group_split_audit,
        "source_relative_path": spec.source_relative_path,
        "source_artifact_hash": file_set_hash(source_paths),
        "source_metadata": source_meta,
        "rows": sizes,
    }
    _write_json_exclusive(destination / "meta.json", frozen_meta)
    frozen_paths = [destination / name for name in REQUIRED_SPLIT_FILES]
    return destination, {
        "dataset_id": spec.dataset_id,
        "dataset_hash": file_set_hash(frozen_paths),
        "source_artifact_hash": frozen_meta["source_artifact_hash"],
        "split_id": spec.split_id,
        "split_seed": spec.split_seed,
        "target_col": spec.target_col,
        "task_type": spec.task_type,
        "metric": spec.metric,
        "metric_direction": spec.metric_direction,
        "target_transform": spec.target_transform,
        "drop_columns": spec.drop_columns,
        "split_policy": spec.split_policy,
        "group_split_audit": group_split_audit,
        "rows": sizes,
    }


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = list(frame.select_dtypes(include="number").columns)
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _score(metric: str, truth: pd.Series, prediction: Any) -> float:
    if metric == "neg_rmse":
        return -float(mean_squared_error(truth, prediction) ** 0.5)
    if metric in {"f1_macro", "f1_weighted"}:
        return float(
            f1_score(
                truth,
                prediction,
                average="macro" if metric == "f1_macro" else "weighted",
                zero_division=0,
            )
        )
    raise FreezeInputError(f"unsupported reference metric: {metric}")


def evaluate_frozen_references(
    dataset_dir: Path,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(dataset["target_col"])
    train = pd.read_csv(dataset_dir / "train.csv")
    validation = pd.read_csv(dataset_dir / "val.csv")
    test = pd.read_csv(dataset_dir / "test.csv")
    features = train.drop(columns=[target])
    target_train = train[target]
    validation_features = validation.drop(columns=[target])
    test_features = test.drop(columns=[target])

    if dataset["task_type"] == "regression":
        dummy_model: Any = DummyRegressor(strategy="mean")
        reference_model: Any = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        dummy_model = DummyClassifier(strategy="prior")
        reference_model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
            class_weight="balanced",
        )

    dummy = Pipeline([("preprocess", _preprocessor(features)), ("model", dummy_model)])
    reference = Pipeline(
        [("preprocess", _preprocessor(features)), ("model", reference_model)]
    )
    dummy.fit(features, target_train)
    reference.fit(features, target_train)
    metric = str(dataset["metric"])
    dummy_validation = _score(metric, validation[target], dummy.predict(validation_features))
    reference_validation = _score(
        metric, validation[target], reference.predict(validation_features)
    )
    dummy_test = _score(metric, test[target], dummy.predict(test_features))
    reference_test = _score(metric, test[target], reference.predict(test_features))
    if dataset["metric_direction"] == "higher" and reference_test <= dummy_test:
        raise FreezeInputError(
            f"{dataset['dataset_id']} reference does not beat dummy on the frozen endpoint"
        )
    if dataset["metric_direction"] == "lower" and reference_test >= dummy_test:
        raise FreezeInputError(
            f"{dataset['dataset_id']} reference does not beat dummy on the frozen endpoint"
        )
    return {
        "dataset_id": dataset["dataset_id"],
        "metric_direction": dataset["metric_direction"],
        "dummy_score": dummy_test,
        "reference_score": reference_test,
        "validation_diagnostic": {
            "dummy_score": dummy_validation,
            "reference_score": reference_validation,
        },
    }


def build_freeze_bundle(
    config: Mapping[str, Any],
    *,
    datasets_root: Path,
    bundle_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle_root.exists():
        raise FileExistsError(f"Refusing to overwrite freeze bundle: {bundle_root}")
    bundle_root.mkdir(parents=True)
    datasets = []
    references = []
    for raw_spec in config["datasets"]:
        spec = DatasetFreezeSpec.from_mapping(raw_spec)
        destination, frozen = _materialize_dataset(
            spec, datasets_root=datasets_root, bundle_root=bundle_root
        )
        datasets.append(frozen)
        references.append(evaluate_frozen_references(destination, frozen))

    reference_payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "reference_pipeline_id": config["reference_pipeline_id"],
        "datasets": [
            {
                key: value
                for key, value in reference.items()
                if key != "validation_diagnostic"
            }
            for reference in references
        ],
    }
    reference_hash = canonical_hash(reference_payload)
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "kind": "autovibe-paper-v1-freeze-inputs",
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "reference_pipeline_id": config["reference_pipeline_id"],
        "reference_pipeline": config["reference_pipeline"],
        "datasets": datasets,
        "reference_hash": reference_hash,
        "validation_diagnostics": {
            item["dataset_id"]: item["validation_diagnostic"] for item in references
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    references_file = {**reference_payload, "reference_hash": reference_hash}
    _write_json_exclusive(bundle_root / "freeze_manifest.json", manifest)
    _write_json_exclusive(bundle_root / "fanu_reference_values.json", references_file)
    return manifest, references_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    manifest, references = build_freeze_bundle(
        load_freeze_config(args.config),
        datasets_root=args.datasets_root,
        bundle_root=args.bundle_root,
    )
    print(
        json.dumps(
            {
                "bundle_root": str(args.bundle_root.resolve()),
                "manifest_hash": manifest["manifest_hash"],
                "reference_hash": references["reference_hash"],
                "datasets": len(manifest["datasets"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
