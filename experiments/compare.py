"""
Aggregate all MLflow runs into a comparison table.

Usage:
    python -m experiments.compare
    python -m experiments.compare --experiment autovibe-gym --metric test_metric
"""
import argparse

import mlflow
import pandas as pd
from dotenv import load_dotenv

from experiments.mlflow_config import configure_mlflow_tracking

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="autovibe-gym")
    parser.add_argument("--metric", default="test_metric")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name")
    parser.add_argument("--output", default=None, help="Save CSV to this path")
    args = parser.parse_args()

    configure_mlflow_tracking(mlflow)

    filter_str = f"params.dataset = '{args.dataset}'" if args.dataset else None
    runs = mlflow.search_runs(
        experiment_names=[args.experiment],
        filter_string=filter_str,
    )

    if runs.empty:
        print("No runs found. Have you run any experiments yet?")
        return

    cols = {
        "params.experiment_type": "experiment_type",
        "params.model": "model",
        "params.dataset": "dataset",
        "params.mode": "mode",
        f"metrics.{args.metric}": args.metric,
        "metrics.final_test_metric": "final_test_metric",
        "metrics.best_validation_metric": "best_validation_metric",
        "metrics.best_val_metric": "best_val_metric",
        "tags.final_status": "final_status",
        "tags.null_reason": "null_reason",
        "tags.finalize_path": "finalize_path",
        "metrics.valid_submit": "valid_submit",
        "metrics.checklist_coverage": "checklist_cov",
        "metrics.steps_used": "steps",
        "metrics.attempts_used": "attempts",
        "metrics.error_count": "errors",
        "metrics.model_check_failure_count": "model_check_failure_count",
        "metrics.contract_feedback_count": "contract_feedback_count",
        "metrics.clean_runs_total": "clean_runs_total",
        "metrics.successful_clean_run": "successful_clean_run",
        "metrics.has_test_metric": "has_metric",
        "metrics.submit_failed": "submit_failed",
        "metrics.input_tokens": "in_tokens",
        "metrics.output_tokens": "out_tokens",
        "metrics.elapsed_seconds": "elapsed_s",
        "run_id": "mlflow_run_id",
    }

    available = {k: v for k, v in cols.items() if k in runs.columns}
    table = runs[list(available.keys())].rename(columns=available)
    if "best_validation_metric" not in table.columns and "best_val_metric" in table.columns:
        table["best_validation_metric"] = table["best_val_metric"]
    if args.metric in table.columns and "best_validation_metric" in table.columns:
        table["val_test_gap"] = table["best_validation_metric"] - table[args.metric]
    if "in_tokens" in table.columns or "out_tokens" in table.columns:
        in_tokens = table["in_tokens"] if "in_tokens" in table.columns else pd.Series(0, index=table.index)
        out_tokens = table["out_tokens"] if "out_tokens" in table.columns else pd.Series(0, index=table.index)
        table["total_tokens"] = in_tokens.fillna(0) + out_tokens.fillna(0)
        if args.metric in table.columns:
            denom = table["total_tokens"].replace(0, pd.NA) / 1000
            table["score_per_1k_tokens"] = table[args.metric] / denom
    sort_cols = [col for col in ["dataset", "model", "experiment_type"] if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, na_position="last")

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
