import json
import io
import tempfile
import zipfile
from pathlib import Path
import unittest
from unittest import mock

import pandas as pd
import yaml

from gym.dataset_ingestion import (
    download_dataset_url,
    validate_dataset_config,
)
from scripts.prepare_datasets import (
    apply_declared_preparation,
    discover_dataset_dirs,
    load_dataset_config,
    load_raw_dataframe,
    prepare_dataset,
    split_temporal,
)
from gym.datasets import (
    DatasetMetadata,
    format_dataset_context,
    infer_metric,
    load_dataset_splits,
    load_splits_from_dir,
    metric_from_name,
    resolve_metric,
)


def _write_config(dataset_dir: Path, overrides: dict | None = None) -> None:
    config = {
        "name": "demo",
        "suite": "example_datasets",
        "source": {},
        "raw_data": {
            "files": ["data.csv"],
            "format": "csv",
            "read_options": {"sep": ","},
        },
        "task": {
            "type": "classification",
            "target_col": "y",
            "metric": "f1_macro",
        },
        "split": {
            "strategy": "stratified_random",
            "seed": 42,
            "train_fraction": 0.5,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
        },
        "preparation": {},
        "notes": {},
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    (dataset_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _write_modern_config(dataset_dir: Path, overrides: dict | None = None) -> None:
    config = {
        "name": "demo",
        "suite": "custom",
        "source": {
            "title": "Synthetic dataset",
            "url": "",
            "license": "",
            "citation": "",
            "description": "",
        },
        "dataset_notes": {
            "short_description": "Demo dataset",
            "llm_context": "Predict the target column.",
            "warnings": [],
            "known_pitfalls": [],
        },
        "ingestion": {
            "mode": "raw",
            "files": [
                {
                    "logical_name": "table_1",
                    "role": "base",
                    "source_type": "local",
                    "url": "",
                    "path": "raw_data/data.csv",
                    "format": "auto",
                    "read_options": {},
                    "optional": False,
                    "archive_member": "",
                }
            ],
        },
        "relations": {"base_table": "table_1", "joins": []},
        "task": {
            "type": "classification",
            "target_col": "y",
            "metric": "f1_macro",
            "forbidden_columns": [],
        },
        "split": {
            "strategy": "stratified_random",
            "seed": 42,
            "train_fraction": 0.5,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "create_val_from_train_if_missing": True,
            "val_fraction_from_train": 0.2,
        },
        "preparation": {
            "drop_columns": [],
            "rename_columns": {},
            "target_mapping": {},
        },
    }
    if overrides:
        for key, value in overrides.items():
            config[key] = value
    (dataset_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


class DatasetPipelineTests(unittest.TestCase):
    def test_metric_from_name_supports_f1_macro(self):
        fn = metric_from_name("f1_macro")
        self.assertAlmostEqual(fn(pd.Series([0, 1]), pd.Series([0, 1])), 1.0)

    def test_metric_resolution_infers_classification_and_regression(self):
        class_fn, class_name = infer_metric(pd.Series([0, 1, 1]))
        reg_fn, reg_name = infer_metric(pd.Series(range(11)))

        self.assertEqual(class_name, "f1_weighted")
        self.assertEqual(reg_name, "neg_rmse")
        self.assertAlmostEqual(class_fn(pd.Series([0, 1]), pd.Series([0, 1])), 1.0)
        self.assertLess(reg_fn(pd.Series([0, 2]), pd.Series([0, 0])), 0)

    def test_metric_from_name_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "Unsupported metric"):
            metric_from_name("auc")

    def test_resolve_metric_prefers_metadata_metric(self):
        metadata = DatasetMetadata(name="demo", target_col="y", metric_name="f1_macro")

        _, metric_name = resolve_metric(metadata, pd.Series(range(20)))

        self.assertEqual(metric_name, "f1_macro")

    def test_load_dataset_splits_accepts_dataset_root_with_prepared_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            prepared = root / "prepared"
            prepared.mkdir(parents=True)
            df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
            for name in ("train", "val", "test"):
                df.to_csv(prepared / f"{name}.csv", index=False)
            (prepared / "meta.json").write_text(json.dumps({"target_col": "y", "metric": "f1_macro"}))
            splits = load_splits_from_dir(str(root))
            self.assertEqual(splits.target_col, "y")

    def test_load_dataset_splits_requires_exactly_one_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_dataset_splits()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_dataset_splits(dataset="data.csv", dataset_dir="prepared")

    def test_csv_mode_requires_target_column(self):
        with self.assertRaisesRegex(ValueError, "target_col"):
            load_dataset_splits(dataset="data.csv")

    def test_load_splits_from_dir_requires_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_splits_from_dir(td)

    def test_load_dataset_config_reads_yaml_and_defaults_optional_fields(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            ds.mkdir()
            _write_config(ds, {"suite": "custom", "role": "smoke"})

            cfg = load_dataset_config(ds)

            self.assertEqual(cfg.name, "demo")
            self.assertEqual(cfg.suite, "custom")
            self.assertEqual(cfg.role, "smoke")

    def test_load_dataset_config_rejects_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            ds.mkdir()
            (ds / "config.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a mapping"):
                load_dataset_config(ds)

    def test_generic_csv_raw_loader_uses_config_read_options(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            (ds / "raw_data").mkdir(parents=True)
            (ds / "raw_data" / "data.csv").write_text("a;b\n1;0\n2;1\n")
            _write_config(
                ds,
                {
                    "raw_data": {"files": ["data.csv"], "format": "csv", "read_options": {"sep": ";"}},
                    "task": {"type": "classification", "target_col": "b", "metric": "f1_macro"},
                },
            )
            cfg = load_dataset_config(ds)
            df = load_raw_dataframe(ds, cfg)
            self.assertListEqual(list(df.columns), ["a", "b"])

    def test_raw_loader_extracts_configured_file_from_zip(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            with zipfile.ZipFile(raw_dir / "raw.zip", "w") as zf:
                zf.writestr("nested/data.csv", "a,b\n1,0\n2,1\n")
            _write_config(
                ds,
                {
                    "raw_data": {"files": ["nested/data.csv"], "format": "csv", "read_options": {"sep": ","}},
                    "task": {"type": "classification", "target_col": "b", "metric": "f1_macro"},
                },
            )
            cfg = load_dataset_config(ds)

            df = load_raw_dataframe(ds, cfg)

            self.assertListEqual(list(df.columns), ["a", "b"])
            self.assertTrue((raw_dir / "nested" / "data.csv").exists())

    def test_raw_loader_does_not_extract_unconfigured_zip_members(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            with zipfile.ZipFile(raw_dir / "raw.zip", "w") as zf:
                zf.writestr("data.csv", "x,y\n1,0\n2,1\n")
                zf.writestr("extra.txt", "do not extract")
            _write_config(ds)

            load_raw_dataframe(ds, load_dataset_config(ds))

            self.assertTrue((raw_dir / "data.csv").exists())
            self.assertFalse((raw_dir / "extra.txt").exists())

    def test_raw_loader_rejects_unsafe_zip_member_requested_by_config(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            with zipfile.ZipFile(raw_dir / "raw.zip", "w") as zf:
                zf.writestr("../data.csv", "x,y\n1,0\n2,1\n")
            _write_config(ds, {"raw_data": {"files": ["../data.csv"], "format": "csv", "read_options": {"sep": ","}}})

            with self.assertRaisesRegex(ValueError, "Unsafe (zip member|relative path)"):
                load_raw_dataframe(ds, load_dataset_config(ds))

    def test_raw_loader_reports_missing_configured_file(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            (ds / "raw_data").mkdir(parents=True)
            _write_config(ds)

            with self.assertRaises(FileNotFoundError):
                load_raw_dataframe(ds, load_dataset_config(ds))

    def test_raw_loader_rejects_unsupported_format(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            (ds / "raw_data").mkdir(parents=True)
            (ds / "raw_data" / "data.csv").write_text("x,y\n1,0\n", encoding="utf-8")
            _write_config(ds, {"raw_data": {"files": ["data.csv"], "format": "xml", "read_options": {}}})

            with self.assertRaisesRegex(ValueError, "Unsupported format"):
                load_raw_dataframe(ds, load_dataset_config(ds))

    def test_dataset_folder_config_is_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "abc"
            ds.mkdir()
            (ds / "config.yaml").write_text("{}")
            self.assertIn("abc", discover_dataset_dirs(Path(td)))

    def test_legacy_fixed_split_folder_is_discovered_and_skipped_by_prepare(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "legacy"
            ds.mkdir()
            pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(ds / "train.csv", index=False)
            (ds / "meta.json").write_text(json.dumps({"target_col": "y"}), encoding="utf-8")

            self.assertIn("legacy", discover_dataset_dirs(Path(td)))
            self.assertEqual(prepare_dataset(ds)["status"], "skipped")

    def test_prepare_dataset_writes_json_serializable_meta_for_numeric_labels(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            df = pd.DataFrame({
                "x": list(range(40)),
                "y": [0, 1] * 20,
            })
            df.to_csv(raw_dir / "data.csv", index=False)
            _write_config(ds)

            res = prepare_dataset(ds)
            self.assertEqual(res["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("class_distribution", meta)
            self.assertIsInstance(meta["class_distribution"]["all"], dict)
            # keys must be JSON-safe strings
            for key in meta["class_distribution"]["all"].keys():
                self.assertIsInstance(key, str)

    def test_prepare_dataset_allows_renaming_target_column(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            df = pd.DataFrame({
                "x": list(range(40)),
                "label_raw": [0, 1] * 20,
            })
            df.to_csv(raw_dir / "data.csv", index=False)
            _write_config(
                ds,
                {
                    "task": {"type": "classification", "target_col": "label", "metric": "f1_macro"},
                    "preparation": {"rename_columns": {"label_raw": "label"}},
                },
            )

            res = prepare_dataset(ds)
            self.assertEqual(res["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["target_col"], "label")

    def test_preparation_drops_columns_maps_target_and_samples_when_allowed(self):
        cfg = load_dataset_config_from_dict(
            {
                "task": {"type": "classification", "target_col": "label", "metric": "f1_macro"},
                "split": {"seed": 7},
                "preparation": {
                    "drop_columns": ["drop_me"],
                    "target_mapping": {"yes": 1, "no": 0},
                    "sampling": {"allowed": True, "seed": 7},
                },
            }
        )
        df = pd.DataFrame({
            "x": list(range(12)),
            "drop_me": list(range(12)),
            "label": ["yes", "no"] * 6,
        })

        prepared, info = apply_declared_preparation(df, cfg, max_rows=6)

        self.assertNotIn("drop_me", prepared.columns)
        self.assertSetEqual(set(prepared["label"]), {0, 1})
        self.assertEqual(len(prepared), 6)
        self.assertTrue(info["sampled"])

    def test_preparation_rejects_missing_target_after_renames_and_drops(self):
        cfg = load_dataset_config_from_dict(
            {
                "task": {"type": "classification", "target_col": "label", "metric": "f1_macro"},
                "preparation": {},
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing target column"):
            apply_declared_preparation(pd.DataFrame({"x": [1]}), cfg, max_rows=None)

    def test_preparation_rejects_sampling_when_config_disallows_it(self):
        cfg = load_dataset_config_from_dict({"preparation": {}})
        df = pd.DataFrame({"x": list(range(10)), "y": [0, 1] * 5})

        with self.assertRaisesRegex(ValueError, "--max-rows is not allowed"):
            apply_declared_preparation(df, cfg, max_rows=4)

    def test_prepare_dataset_rejects_invalid_split_fractions(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}).to_csv(raw_dir / "data.csv", index=False)
            _write_config(ds, {"split": {"train_fraction": 0.8, "val_fraction": 0.2, "test_fraction": 0.2}})

            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                prepare_dataset(ds)

    def test_prepare_dataset_rejects_unknown_split_strategy(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}).to_csv(raw_dir / "data.csv", index=False)
            _write_config(ds, {"split": {"strategy": "moon"}})

            with self.assertRaisesRegex(ValueError, "Unsupported split strategy"):
                prepare_dataset(ds)

    def test_temporal_split_sorts_by_timestamp_and_removes_technical_column(self):
        df = pd.DataFrame({
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04"],
            "target": [1, 0, 1, 0],
        })

        train, val, test, info = split_temporal(
            df,
            {
                "train_fraction": 0.5,
                "val_fraction": 0.25,
                "test_fraction": 0.25,
                "timestamp": {"source_columns": ["date"], "format": None},
            },
        )

        self.assertEqual(list(train["date"]), ["2024-01-01", "2024-01-02"])
        self.assertEqual(list(val["date"]), ["2024-01-03"])
        self.assertEqual(list(test["date"]), ["2024-01-04"])
        self.assertNotIn("__technical_timestamp__", train.columns)
        self.assertIn("temporal_boundaries", info)

    def test_temporal_split_rejects_unparseable_timestamps(self):
        df = pd.DataFrame({"date": ["not-a-date"], "target": [0]})

        with self.assertRaisesRegex(ValueError, "timestamp parsing"):
            split_temporal(
                df,
                {
                    "train_fraction": 1.0,
                    "val_fraction": 0.0,
                    "test_fraction": 0.0,
                    "timestamp": {"source_columns": ["date"], "format": None},
                },
            )

    def test_prepare_dataset_supports_tsv_and_txt_with_delimiter_detection(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            (raw_dir / "data.txt").write_text("a|y\n1|0\n2|1\n3|0\n4|1\n", encoding="utf-8")
            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "table_1",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/data.txt",
                                "format": "txt",
                                "read_options": {"sep": "|"},
                                "optional": False,
                                "archive_member": "",
                            }
                        ],
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            train = pd.read_csv(ds / "prepared" / "train.csv")
            self.assertListEqual(list(train.columns), ["a", "y"])

    def test_prepare_dataset_supports_excel(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            df = pd.DataFrame({"feature": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
            try:
                df.to_excel(raw_dir / "data.xlsx", index=False)
            except ImportError as exc:
                self.skipTest(str(exc))

            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "table_1",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/data.xlsx",
                                "format": "auto",
                                "read_options": {"sheet_name": 0},
                                "optional": False,
                                "archive_member": "",
                            }
                        ],
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["input_formats"]["table_1"], "xlsx")

    def test_prepare_dataset_supports_parquet(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            df = pd.DataFrame({"feature": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
            try:
                df.to_parquet(raw_dir / "data.parquet", index=False)
            except (ImportError, ValueError) as exc:
                self.skipTest(str(exc))

            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "table_1",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/data.parquet",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            }
                        ],
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["input_formats"]["table_1"], "parquet")

    def test_prepare_dataset_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            (raw_dir / "data.jsonl").write_text(
                "\n".join([
                    '{"feature": 1, "y": 0}',
                    '{"feature": 2, "y": 1}',
                    '{"feature": 3, "y": 0}',
                    '{"feature": 4, "y": 1}',
                ]),
                encoding="utf-8",
            )
            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "table_1",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/data.jsonl",
                                "format": "auto",
                                "read_options": {"lines": True},
                                "optional": False,
                                "archive_member": "",
                            }
                        ],
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["input_formats"]["table_1"], "jsonl")

    def test_prepare_dataset_supports_pre_split_and_creates_val_from_train(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            train_df = pd.DataFrame({"x": list(range(12)), "y": [0, 1] * 6})
            test_df = pd.DataFrame({"x": list(range(12, 16)), "y": [0, 1, 0, 1]})
            train_df.to_csv(raw_dir / "train.csv", index=False)
            test_df.to_csv(raw_dir / "test.csv", index=False)
            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "pre_split",
                        "files": [
                            {
                                "logical_name": "train",
                                "role": "train",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/train.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                            {
                                "logical_name": "test",
                                "role": "test",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/test.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                        ],
                    },
                    "split": {
                        "strategy": "pre_split",
                        "seed": 42,
                        "train_fraction": 0.7,
                        "val_fraction": 0.15,
                        "test_fraction": 0.15,
                        "create_val_from_train_if_missing": True,
                        "val_fraction_from_train": 0.25,
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            self.assertTrue((ds / "prepared" / "val.csv").exists())
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["split_strategy"], "pre_split")

    def test_prepare_dataset_supports_multi_table_join_and_forbidden_columns(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "accident_index": [1, 2, 3, 4],
                    "vehicle_reference": [11, 22, 33, 44],
                    "target": [0, 1, 0, 1],
                }
            ).to_csv(raw_dir / "casualties.csv", index=False)
            pd.DataFrame(
                {
                    "accident_index": [1, 2, 3, 4],
                    "weather": ["rain", "sun", "fog", "wind"],
                    "accident_severity": [3, 2, 1, 2],
                }
            ).to_csv(raw_dir / "collisions.csv", index=False)
            pd.DataFrame(
                {
                    "accident_index": [1, 2, 3, 4],
                    "vehicle_reference": [11, 22, 33, 44],
                    "vehicle_type": ["car", "van", "bike", "bus"],
                }
            ).to_csv(raw_dir / "vehicles.csv", index=False)

            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "casualties",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/casualties.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                            {
                                "logical_name": "collisions",
                                "role": "table",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/collisions.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                            {
                                "logical_name": "vehicles",
                                "role": "table",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/vehicles.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                        ],
                    },
                    "relations": {
                        "base_table": "casualties",
                        "joins": [
                            {
                                "left_table": "casualties",
                                "right_table": "collisions",
                                "how": "left",
                                "left_on": ["accident_index"],
                                "right_on": ["accident_index"],
                            },
                            {
                                "left_table": "casualties",
                                "right_table": "vehicles",
                                "how": "left",
                                "left_on": ["accident_index", "vehicle_reference"],
                                "right_on": ["accident_index", "vehicle_reference"],
                            },
                        ],
                    },
                    "task": {
                        "type": "classification",
                        "target_col": "target",
                        "metric": "f1_macro",
                        "forbidden_columns": ["accident_severity", "accident_index"],
                    },
                },
            )

            result = prepare_dataset(ds)

            self.assertEqual(result["status"], "ok")
            train = pd.read_csv(ds / "prepared" / "train.csv")
            self.assertNotIn("accident_index", train.columns)
            self.assertNotIn("accident_severity", train.columns)
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(len(meta["join_diagnostics"]), 2)

    def test_validate_dataset_reports_join_multiplication(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            pd.DataFrame({"id": [1, 2, 3], "y": [0, 1, 0]}).to_csv(raw_dir / "base.csv", index=False)
            pd.DataFrame({"id": [1] * 10 + [2, 3], "v": list(range(12))}).to_csv(raw_dir / "detail.csv", index=False)
            _write_modern_config(
                ds,
                {
                    "ingestion": {
                        "mode": "raw",
                        "files": [
                            {
                                "logical_name": "base",
                                "role": "base",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/base.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                            {
                                "logical_name": "detail",
                                "role": "table",
                                "source_type": "local",
                                "url": "",
                                "path": "raw_data/detail.csv",
                                "format": "auto",
                                "read_options": {},
                                "optional": False,
                                "archive_member": "",
                            },
                        ],
                    },
                    "relations": {
                        "base_table": "base",
                        "joins": [
                            {
                                "left_table": "base",
                                "right_table": "detail",
                                "how": "left",
                                "left_on": ["id"],
                                "right_on": ["id"],
                            }
                        ],
                    },
                },
            )

            validation = validate_dataset_config(ds)

            self.assertFalse(validation["ok"])
            self.assertTrue(any("multiplied rows" in item["message"] for item in validation["errors"]))

    def test_prepare_dataset_writes_dataset_notes_and_source_to_meta(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            raw_dir = ds / "raw_data"
            raw_dir.mkdir(parents=True)
            pd.DataFrame({"x": list(range(8)), "y": [0, 1] * 4}).to_csv(raw_dir / "data.csv", index=False)
            _write_modern_config(
                ds,
                {
                    "source": {
                        "title": "UCI Demo",
                        "url": "https://example.com/demo.csv",
                        "license": "CC-BY",
                        "citation": "Doe 2026",
                        "description": "Synthetic source",
                    },
                    "dataset_notes": {
                        "short_description": "Demo classification task",
                        "llm_context": "Task text may live here.",
                        "warnings": ["Do not leak ids."],
                        "known_pitfalls": ["Temporal drift matters."],
                    },
                },
            )

            prepare_dataset(ds)
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))

            self.assertEqual(meta["source"]["title"], "UCI Demo")
            self.assertEqual(meta["dataset_notes"]["llm_context"], "Task text may live here.")

    def test_format_dataset_context_includes_llm_notes_and_source(self):
        metadata = DatasetMetadata(
            name="demo",
            target_col="y",
            metric_name="f1_macro",
            source={"title": "UCI Demo", "license": "CC-BY"},
            dataset_notes={
                "short_description": "Demo task",
                "llm_context": "Predict churn from raw features.",
                "warnings": ["Avoid leakage columns."],
                "known_pitfalls": ["Class imbalance."],
            },
        )

        text = format_dataset_context(metadata)

        self.assertIn("[DATASET CONTEXT]", text)
        self.assertIn("Predict churn", text)
        self.assertIn("Avoid leakage", text)
        self.assertIn("UCI Demo", text)

    def test_download_dataset_url_saves_file_with_mocked_response(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            ds.mkdir()

            class FakeResponse(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            with mock.patch("gym.dataset_ingestion._validate_download_url"), mock.patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(b"a,b\n1,0\n2,1\n"),
            ):
                saved = download_dataset_url(ds, "https://example.com/data.csv")

            self.assertTrue((ds / saved["path"]).exists())
            self.assertEqual(saved["format"], "csv")


def load_dataset_config_from_dict(overrides: dict):
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "demo"
        ds.mkdir()
        _write_config(ds, overrides)
        return load_dataset_config(ds)


if __name__ == '__main__':
    unittest.main()
