from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard.server.app.services import dataset_store


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        datasets_dir=tmp_path / "datasets",
        uploads_dir=tmp_path / "uploads",
    )
    settings.datasets_dir.mkdir()
    settings.uploads_dir.mkdir()
    monkeypatch.setattr(dataset_store, "get_settings", lambda: settings)
    return settings


def _csv_bytes(rows: int = 20) -> bytes:
    df = pd.DataFrame(
        {
            "x": list(range(rows)),
            "segment": ["a" if i % 2 else "b" for i in range(rows)],
            "target": [0, 1] * (rows // 2),
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def test_sanitize_dataset_id_rejects_dangerous_names():
    assert dataset_store.sanitize_dataset_id("My Dataset 01") == "my-dataset-01"

    with pytest.raises(ValueError):
        dataset_store.sanitize_dataset_id("../secret")


def test_safe_child_blocks_path_traversal(tmp_path):
    base = tmp_path / "base"
    base.mkdir()

    with pytest.raises(ValueError):
        dataset_store._safe_child(base, "../escape.csv")


def test_archive_extraction_rejects_path_traversal(isolated_store):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.csv", "x,target\n1,0\n")

    uploaded = dataset_store.upload_file("bad.zip", buf.getvalue())

    with pytest.raises(ValueError, match="Unsafe archive member"):
        dataset_store.extract_upload_archive(uploaded["upload_id"], "uploaded/bad.zip")


def test_csv_upload_preview_returns_columns_rows_and_missing_counts(isolated_store):
    uploaded = dataset_store.upload_file("data.csv", _csv_bytes(rows=6))

    preview = dataset_store.preview_upload(uploaded["upload_id"], "uploaded/data.csv", limit=3)

    assert preview["columns"] == ["x", "segment", "target"]
    assert preview["shown"] == 3
    assert preview["total"] == 6
    assert preview["missing"]["target"] == 0


def test_create_dataset_from_one_raw_table_writes_splits_config_and_meta(isolated_store):
    uploaded = dataset_store.upload_file("raw.csv", _csv_bytes(rows=20))

    ds = dataset_store.create_from_config(
        {
            "id": "raw-demo",
            "name": "Raw Demo",
            "upload_id": uploaded["upload_id"],
            "task": {
                "task_type": "classification",
                "target_col": "target",
                "metric_name": "f1_macro",
                "metric_goal": "max",
            },
            "splits": {
                "mode": "raw_split",
                "raw_path": "uploaded/raw.csv",
                "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                "seed": 7,
                "shuffle": True,
                "stratify": "off",
            },
            "agent_notes": {"task_description": "Predict the target.", "visible_to_agent": True},
            "sources": [{"name": "unit"}],
            "tags": ["smoke"],
        }
    )

    root = isolated_store.datasets_dir / "raw-demo"
    assert ds["prepared"] is True
    assert (root / "raw" / "uploaded" / "raw.csv").exists()
    assert (root / "prepared" / "train.csv").exists()
    assert (root / "prepared" / "val.csv").exists()
    assert (root / "prepared" / "test.csv").exists()
    meta = json.loads((root / "prepared" / "meta.json").read_text("utf-8"))
    cfg = json.loads((root / "dataset_config.json").read_text("utf-8"))
    assert meta["target_col"] == "target"
    assert meta["metric_name"] == "f1_macro"
    assert cfg["status"] == "prepared"
    assert cfg["raw_files"][0]["path"].startswith("raw/uploaded/")


def test_create_dataset_from_prepared_files(isolated_store):
    uploaded = None
    upload_id = None
    for split in ("train", "val", "test"):
        uploaded = dataset_store.upload_file(f"{split}.csv", _csv_bytes(rows=10), upload_id)
        upload_id = uploaded["upload_id"]

    ds = dataset_store.create_from_config(
        {
            "id": "prepared-demo",
            "name": "Prepared Demo",
            "upload_id": upload_id,
            "task": {"task_type": "classification", "target_col": "target", "metric_name": "f1_macro"},
            "splits": {
                "mode": "prepared_files",
                "mapping": {
                    "train": "uploaded/train.csv",
                    "val": "uploaded/val.csv",
                    "test": "uploaded/test.csv",
                },
                "seed": 42,
            },
        }
    )

    cfg = dataset_store.get_dataset_config("prepared-demo")
    assert ds["status"] == "prepared"
    assert cfg is not None
    assert cfg["splits"]["train"]["rows"] == 10
    assert cfg["splits"]["val"]["source_path"] == "raw/uploaded/val.csv"


def test_create_dataset_can_split_validation_from_train(isolated_store):
    train = dataset_store.upload_file("train.csv", _csv_bytes(rows=20))
    upload_id = train["upload_id"]
    dataset_store.upload_file("test.csv", _csv_bytes(rows=8), upload_id)

    ds = dataset_store.create_from_config(
        {
            "id": "train-val-demo",
            "name": "Train Val Demo",
            "upload_id": upload_id,
            "task": {"task_type": "classification", "target_col": "target", "metric_name": "f1_macro"},
            "splits": {
                "mode": "prepared_files",
                "mapping": {"train": "uploaded/train.csv", "test": "uploaded/test.csv"},
                "create_val_from_train": True,
                "val_ratio": 0.25,
                "stratify": "off",
            },
        }
    )

    root = isolated_store.datasets_dir / "train-val-demo"
    assert ds["prepared"] is True
    assert pd.read_csv(root / "prepared" / "val.csv").shape[0] == 5


def test_old_prepared_meta_dataset_still_describes_correctly(isolated_store):
    root = isolated_store.datasets_dir / "legacy"
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    for split in ("train", "val", "test"):
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_csv(prepared / f"{split}.csv", index=False)
    (prepared / "meta.json").write_text(
        json.dumps({"name": "Legacy", "target_col": "target", "metric_name": "f1_macro", "seed": 9}),
        encoding="utf-8",
    )

    ds = dataset_store.get_dataset("legacy")

    assert ds is not None
    assert ds["prepared"] is True
    assert ds["name"] == "Legacy"
    assert ds["target"] == "target"
    assert ds["rows"] == 6
