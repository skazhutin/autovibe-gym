"""
Aggregate all MLflow runs into a comparison table.

Usage:
    python -m experiments.compare
    python -m experiments.compare --experiment autovibe-gym --metric test_metric
"""
import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="autovibe-gym")
    parser.add_argument("--metric", default="test_metric")
    parser.add_argument("--output", default=None, help="Save CSV to this path")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[args.experiment],
        order_by=[f"metrics.{args.metric} DESC"],
    )

    if runs.empty:
        print("No runs found. Have you run any experiments yet?")
        return

    cols = {
        "params.experiment_type": "type",
        "params.model": "model",
        "params.dataset": "dataset",
        "params.mode": "mode",
        f"metrics.{args.metric}": args.metric,
        "metrics.checklist_coverage": "checklist_cov",
        "metrics.steps_used": "steps",
        "metrics.error_count": "errors",
        "metrics.input_tokens": "in_tokens",
        "metrics.output_tokens": "out_tokens",
        "metrics.elapsed_seconds": "elapsed_s",
    }

    available = {k: v for k, v in cols.items() if k in runs.columns}
    table = runs[list(available.keys())].rename(columns=available)
    table = table.sort_values(args.metric, ascending=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n=== Experiment: {args.experiment} | Sorted by: {args.metric} ===\n")
    print(table.to_string(index=False))

    if args.output:
        table.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
