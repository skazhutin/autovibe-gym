import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from scripts.prepare_datasets import discover_dataset_dirs, load_dataset_config, load_raw_dataframe, prepare_dataset
from gym.datasets import load_splits_from_dir, metric_from_name


class DatasetPipelineTests(unittest.TestCase):
    def test_metric_from_name_supports_f1_macro(self):
        fn = metric_from_name("f1_macro")
        self.assertAlmostEqual(fn(pd.Series([0, 1]), pd.Series([0, 1])), 1.0)

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

    def test_generic_csv_raw_loader_uses_config_read_options(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "demo"
            (ds / "raw_data").mkdir(parents=True)
            (ds / "raw_data" / "data.csv").write_text("a;b\n1;0\n2;1\n")
            (ds / "config.json").write_text(json.dumps({"name":"demo","suite":"example_datasets","source":{},"raw_data":{"files":["data.csv"],"format":"csv","read_options":{"sep":";"}},"task":{"type":"classification","target_col":"b","metric":"f1_macro"},"split":{"strategy":"stratified_random","seed":42,"train_fraction":0.7,"val_fraction":0.15,"test_fraction":0.15},"preparation":{},"notes":{}}))
            cfg = load_dataset_config(ds)
            df = load_raw_dataframe(ds, cfg)
            self.assertListEqual(list(df.columns), ["a", "b"])

    def test_dataset_folder_config_is_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "abc"
            ds.mkdir()
            (ds / "config.json").write_text("{}")
            self.assertIn("abc", discover_dataset_dirs(Path(td)))

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
            (ds / "config.json").write_text(json.dumps({
                "name": "demo",
                "suite": "example_datasets",
                "source": {},
                "raw_data": {"files": ["data.csv"], "format": "csv", "read_options": {"sep": ","}},
                "task": {"type": "classification", "target_col": "y", "metric": "f1_macro"},
                "split": {"strategy": "stratified_random", "seed": 42, "train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25},
                "preparation": {},
                "notes": {},
            }))

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
            (ds / "config.json").write_text(json.dumps({
                "name": "demo",
                "suite": "example_datasets",
                "source": {},
                "raw_data": {"files": ["data.csv"], "format": "csv", "read_options": {"sep": ","}},
                "task": {"type": "classification", "target_col": "label", "metric": "f1_macro"},
                "split": {"strategy": "stratified_random", "seed": 42, "train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25},
                "preparation": {"rename_columns": {"label_raw": "label"}},
                "notes": {},
            }))

            res = prepare_dataset(ds)
            self.assertEqual(res["status"], "ok")
            meta = json.loads((ds / "prepared" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["target_col"], "label")


if __name__ == '__main__':
    unittest.main()
