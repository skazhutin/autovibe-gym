import json
from pathlib import Path

import pandas as pd
import pytest

from research.freeze_inputs import (
    DatasetFreezeSpec,
    _apply_split_policy,
    build_freeze_bundle,
)


def _write_source(root: Path, *, regression: bool) -> None:
    root.mkdir(parents=True)
    for split, offset in (("train", 0), ("val", 1), ("test", 2)):
        rows = []
        for index in range(200):
            value = (index * 7 + offset) % 200
            target = float(value * 2 + 1) if regression else (value % 2)
            rows.append(
                {
                    "number": value,
                    "category": "even" if value % 2 == 0 else "odd",
                    "target": target,
                }
            )
        pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)
    (root / "meta.json").write_text(json.dumps({"name": root.name}), encoding="utf-8")


def _config() -> dict:
    return {
        "schema_version": "1.0",
        "reference_pipeline_id": "fixture-reference-v1",
        "reference_pipeline": {"selection_rule": "fixed"},
        "datasets": [
            {
                "dataset_id": "regression",
                "source_relative_path": "regression",
                "target_col": "target",
                "task_type": "regression",
                "metric": "neg_rmse",
                "metric_direction": "higher",
                "split_id": "fixture-regression",
                "split_seed": 42,
                "target_transform": "identity",
                "drop_columns": [],
                "split_policy": {"type": "preserve_source"},
            },
            {
                "dataset_id": "classification",
                "source_relative_path": "classification",
                "target_col": "target",
                "task_type": "binary_classification",
                "metric": "f1_macro",
                "metric_direction": "higher",
                "split_id": "fixture-classification",
                "split_seed": 42,
                "target_transform": "identity",
                "drop_columns": [],
                "split_policy": {"type": "preserve_source"},
            },
        ],
    }


def test_build_freeze_bundle_is_hashed_complete_and_no_overwrite(tmp_path):
    sources = tmp_path / "sources"
    _write_source(sources / "regression", regression=True)
    _write_source(sources / "classification", regression=False)
    destination = tmp_path / "bundle"

    manifest, references = build_freeze_bundle(
        _config(), datasets_root=sources, bundle_root=destination
    )

    assert len(manifest["datasets"]) == 2
    assert len(references["datasets"]) == 2
    assert len(manifest["manifest_hash"]) == 64
    assert len(references["reference_hash"]) == 64
    assert (destination / "freeze_manifest.json").is_file()
    assert (destination / "fanu_reference_values.json").is_file()
    for dataset in manifest["datasets"]:
        assert len(dataset["dataset_hash"]) == 64
        assert dataset["rows"] == {"train": 200, "val": 200, "test": 200}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_freeze_bundle(_config(), datasets_root=sources, bundle_root=destination)


def test_readmission_transform_is_explicit_and_binary(tmp_path):
    source = tmp_path / "sources" / "readmission"
    source.mkdir(parents=True)
    for split in ("train", "val", "test"):
        feature = list(range(300))
        frame = pd.DataFrame(
            {
                "feature": feature,
                "readmitted": ["NO" if value % 3 == 0 else ">30" for value in feature],
            }
        )
        frame.to_csv(source / f"{split}.csv", index=False)
    (source / "meta.json").write_text("{}", encoding="utf-8")
    config = {
        "schema_version": "1.0",
        "reference_pipeline_id": "fixture-reference-v1",
        "reference_pipeline": {"selection_rule": "fixed"},
        "datasets": [
            {
                "dataset_id": "readmission",
                "source_relative_path": "readmission",
                "target_col": "readmitted",
                "task_type": "binary_classification",
                "metric": "f1_macro",
                "metric_direction": "higher",
                "split_id": "fixture-readmission",
                "split_seed": 42,
                "target_transform": "non_no_readmission_binary",
                "drop_columns": [],
                "split_policy": {"type": "preserve_source"},
            }
        ],
    }

    build_freeze_bundle(config, datasets_root=tmp_path / "sources", bundle_root=tmp_path / "bundle")

    frozen = pd.read_csv(tmp_path / "bundle" / "datasets" / "readmission" / "train.csv")
    assert set(frozen["readmitted"]) == {0, 1}


def test_group_split_prevents_entity_overlap_and_removes_ids(tmp_path):
    source = tmp_path / "sources" / "grouped"
    source.mkdir(parents=True)
    rows = []
    for patient in range(120):
        for encounter in range(2):
            rows.append(
                {
                    "patient_id": patient,
                    "encounter_id": patient * 2 + encounter,
                    "feature": patient % 2,
                    "target": patient % 2,
                }
            )
    frame = pd.DataFrame(rows)
    for split, part in zip(("train", "val", "test"), (frame.iloc[:80], frame.iloc[80:160], frame.iloc[160:])):
        part.to_csv(source / f"{split}.csv", index=False)
    (source / "meta.json").write_text("{}", encoding="utf-8")
    config = {
        "schema_version": "1.0",
        "reference_pipeline_id": "fixture-reference-v1",
        "reference_pipeline": {"selection_rule": "fixed"},
        "datasets": [
            {
                "dataset_id": "grouped",
                "source_relative_path": "grouped",
                "target_col": "target",
                "task_type": "binary_classification",
                "metric": "f1_macro",
                "metric_direction": "higher",
                "split_id": "fixture-grouped",
                "split_seed": 42,
                "target_transform": "identity",
                "drop_columns": ["patient_id", "encounter_id"],
                "split_policy": {
                    "type": "group_shuffle",
                    "group_column": "patient_id",
                    "train_fraction": 0.70,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                },
            }
        ],
    }

    raw_spec = config["datasets"][0]
    source_frames = {
        split: pd.read_csv(source / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    grouped = _apply_split_policy(
        source_frames,
        DatasetFreezeSpec.from_mapping(raw_spec),
    )
    patient_sets = {
        split: set(item["patient_id"])
        for split, item in grouped.items()
    }
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    manifest, _ = build_freeze_bundle(
        config,
        datasets_root=tmp_path / "sources",
        bundle_root=tmp_path / "bundle",
    )

    frozen = {
        split: pd.read_csv(tmp_path / "bundle" / "datasets" / "grouped" / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    assert all("patient_id" not in item and "encounter_id" not in item for item in frozen.values())
    assert sum(len(item) for item in frozen.values()) == len(frame)
    assert manifest["datasets"][0]["group_split_audit"]["pairwise_group_overlap"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
