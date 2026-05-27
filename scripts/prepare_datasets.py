"""
Download and split all benchmark datasets into fixed train/val/test splits.
Run once; results are saved to datasets/<name>/ and reused by all experiments.

Usage:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --dataset wine_quality
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

DATASETS = {
    "wine_quality": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "sep": ";",
        "target": "quality",
        "metric": "f1_weighted",
        "description": "UCI Wine Quality (red). Predict wine quality score 3-8.",
    },
    "bank_marketing": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "sep": ";",
        "target": "y",
        "metric": "f1_weighted",
        "description": "UCI Bank Marketing. Predict if client subscribes to term deposit.",
        "zip_file": "bank.csv",
    },
    "heart_disease": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "sep": ",",
        "target": "target",
        "metric": "f1_weighted",
        "description": "UCI Heart Disease (Cleveland). Predict presence of heart disease (binary).",
        "header": None,
        "col_names": [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"
        ],
        "binary_target": True,
    },
}


def download_and_load(name: str, cfg: dict) -> pd.DataFrame:
    import urllib.request
    import zipfile

    raw_dir = os.path.join("datasets", "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    url = cfg["url"]
    ext = ".zip" if url.endswith(".zip") else ".csv"
    local = os.path.join(raw_dir, f"{name}{ext}")

    if not os.path.exists(local):
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, local)

    if ext == ".zip":
        with zipfile.ZipFile(local) as z:
            z.extract(cfg["zip_file"], raw_dir)
        local = os.path.join(raw_dir, cfg["zip_file"])

    kwargs = {"sep": cfg.get("sep", ",")}
    if "header" in cfg and cfg["header"] is None:
        kwargs["header"] = None
        kwargs["names"] = cfg["col_names"]

    df = pd.read_csv(local, **kwargs)

    if cfg.get("binary_target"):
        target = cfg["target"]
        df[target] = (df[target] > 0).astype(int)
        df = df.replace("?", np.nan)
        df = df.dropna()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

    return df


def split_and_save(df: pd.DataFrame, name: str, cfg: dict) -> None:
    target = cfg["target"]
    out_dir = os.path.join("datasets", name)
    os.makedirs(out_dir, exist_ok=True)

    train, temp = train_test_split(df, test_size=0.3, random_state=SEED, stratify=None)
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED)

    train.reset_index(drop=True).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.reset_index(drop=True).to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.reset_index(drop=True).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    meta = {
        "target_col": target,
        "metric": cfg["metric"],
        "description": cfg["description"],
        "seed": SEED,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_features": df.shape[1] - 1,
        "n_classes": int(df[target].nunique()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: train={len(train)}, val={len(val)}, test={len(test)} -> datasets/{name}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Prepare only this dataset")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for name, cfg in targets.items():
        print(f"\n[{name}]")
        try:
            df = download_and_load(name, cfg)
            split_and_save(df, name, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
