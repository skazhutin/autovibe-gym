from __future__ import annotations

import io
import gzip
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


def test_gzip_extraction_enforces_decompressed_size_limit(isolated_store, monkeypatch):
    monkeypatch.setattr(dataset_store, "_MAX_EXTRACTED_BYTES", 100)
    uploaded = dataset_store.upload_file("large.csv.gz", gzip.compress(b"x" * 500))

    with pytest.raises(ValueError, match="maximum allowed size"):
        dataset_store.extract_upload_archive(uploaded["upload_id"], "uploaded/large.csv.gz")


def test_url_upload_rejects_local_and_private_targets(isolated_store):
    with pytest.raises(ValueError, match="localhost"):
        dataset_store.upload_from_url("http://localhost/data.csv")

    with pytest.raises(ValueError, match="private/internal"):
        dataset_store.upload_from_url("http://127.0.0.1/data.csv")


def test_csv_upload_preview_returns_columns_rows_and_missing_counts(isolated_store):
    uploaded = dataset_store.upload_file("data.csv", _csv_bytes(rows=6))

    preview = dataset_store.preview_upload(uploaded["upload_id"], "uploaded/data.csv", limit=3)

    assert preview["columns"] == ["x", "segment", "target"]
    assert preview["shown"] == 3
    assert preview["total"] == 6
    assert preview["missing"]["target"] == 0


def test_jsonl_upload_can_preview_and_create_raw_split(isolated_store):
    rows = [
        {"x": i, "segment": "a" if i % 2 else "b", "target": i % 2}
        for i in range(20)
    ]
    payload = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    uploaded = dataset_store.upload_file("raw.jsonl", payload)

    preview = dataset_store.preview_upload(uploaded["upload_id"], "uploaded/raw.jsonl", limit=4)
    ds = dataset_store.create_from_config(
        {
            "id": "jsonl-demo",
            "name": "JSONL Demo",
            "upload_id": uploaded["upload_id"],
            "task": {
                "task_type": "classification",
                "target_col": "target",
                "metric_name": "f1_macro",
            },
            "splits": {
                "mode": "raw_split",
                "raw_path": "uploaded/raw.jsonl",
                "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
                "seed": 11,
                "shuffle": True,
                "stratify": "off",
            },
        }
    )

    assert preview["columns"] == ["x", "segment", "target"]
    assert preview["shown"] == 4
    assert ds["prepared"] is True
    assert (isolated_store.datasets_dir / "jsonl-demo" / "prepared" / "train.csv").exists()


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


def test_failed_create_from_config_removes_partial_dataset_dir(isolated_store):
    uploaded = dataset_store.upload_file("raw.csv", _csv_bytes(rows=6))

    with pytest.raises(ValueError, match="Target column"):
        dataset_store.create_from_config(
            {
                "id": "bad-target",
                "name": "Bad Target",
                "upload_id": uploaded["upload_id"],
                "task": {"task_type": "classification", "target_col": "missing", "metric_name": "f1_macro"},
                "splits": {"mode": "raw_split", "raw_path": "uploaded/raw.csv"},
            }
        )

    assert not (isolated_store.datasets_dir / "bad-target").exists()


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


def test_legacy_root_dataset_edit_does_not_create_shadow_prepared_meta(isolated_store):
    root = isolated_store.datasets_dir / "legacy-root"
    root.mkdir()
    for split in ("train", "val", "test"):
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_csv(root / f"{split}.csv", index=False)
    (root / "meta.json").write_text(
        json.dumps({"name": "Legacy Root", "target_col": "target", "metric_name": "f1_macro"}),
        encoding="utf-8",
    )

    updated = dataset_store.update_meta("legacy-root", {"desc": "edited"})

    assert updated is not None
    assert updated["prepared"] is True
    assert updated["rows"] == 6
    assert not (root / "prepared" / "meta.json").exists()


def test_empty_source_displays_dash(isolated_store):
    root = isolated_store.datasets_dir / "empty-source"
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    for split in ("train", "val", "test"):
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_csv(prepared / f"{split}.csv", index=False)
    (prepared / "meta.json").write_text(
        json.dumps({"name": "Empty source", "target_col": "target", "metric_name": "f1_macro", "source": {"name": "", "url": ""}}),
        encoding="utf-8",
    )

    ds = dataset_store.get_dataset("empty-source")

    assert ds is not None
    assert ds["source"] == "-"


def test_config_yaml_metadata_fills_example_source_and_created_at(isolated_store):
    root = isolated_store.datasets_dir / "example-config"
    root.mkdir()
    created = "2026-05-27T14:28:51+03:00"
    (root / "config.yaml").write_text(
        "\n".join(
            [
                "name: Example Config",
                f"created_at: {created}",
                "source:",
                "  name: UCI Demo",
                "  url: https://archive.ics.uci.edu/",
                "raw_data:",
                "  files:",
                "    - data.csv",
                "task:",
                "  type: classification",
                "  target_col: target",
                "  metric: f1_macro",
                "split:",
                "  seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    ds = dataset_store.get_dataset("example-config")
    cfg = dataset_store.get_dataset_config("example-config")

    assert ds is not None
    assert cfg is not None
    assert ds["source"] == "UCI Demo"
    assert ds["createdAt"] == created
    assert ds["target"] == "target"
    assert ds["metric"] == "f1_macro"
    assert cfg["sources"][0]["name"] == "UCI Demo"
