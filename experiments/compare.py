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
from experiments.modes import MODE_BY_EXPERIMENT_TYPE, MODE_BY_KEY

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="autovibe-gym")
    parser.add_argument("--metric", default="test_metric")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name")
    parser.add_argument("--output", default=None, help="Save CSV to this path")
    parser.add_argument(
        "--sort-by",
        choices=["matrix", "metric"],
        default="matrix",
        help="Sort by dataset/model/mode matrix order or selected metric descending.",
    )
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
        "params.requested_mode": "requested_mode",
        "params.batch_id": "batch_id",
        "params.product_mode": "product_mode",
        "params.mode_label": "mode_label",
        "params.mode_order": "mode_order",
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
    if "product_mode" not in table.columns:
        table["product_mode"] = table.get("experiment_type", "")
    else:
        table["product_mode"] = table["product_mode"].fillna(table.get("experiment_type", ""))
    if "requested_mode" not in table.columns:
        table["requested_mode"] = table["product_mode"]
    else:
        table["requested_mode"] = table["requested_mode"].fillna(table["product_mode"])
    if "batch_id" not in table.columns:
        table["batch_id"] = ""
    else:
        table["batch_id"] = table["batch_id"].fillna("")
    if "mode_label" not in table.columns:
        table["mode_label"] = table["product_mode"]
    else:
        table["mode_label"] = table["mode_label"].fillna(table["product_mode"])
    if "mode_order" not in table.columns:
        table["mode_order"] = pd.NA
    table["mode_order"] = pd.to_numeric(table["mode_order"], errors="coerce")
    if "experiment_type" in table.columns:
        def _infer_mode_order(row):
            if pd.notna(row["mode_order"]):
                return row["mode_order"]
            mode_key = row.get("product_mode") or row.get("experiment_type")
            spec = MODE_BY_KEY.get(str(mode_key)) or MODE_BY_EXPERIMENT_TYPE.get(str(mode_key))
            return spec.mode_order if spec else 999

        table["mode_order"] = table.apply(_infer_mode_order, axis=1)
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
    table["_batch_sort"] = table["batch_id"].where(
        table["batch_id"].astype(str).str.len() > 0,
        table["requested_mode"],
    )
    sort_cols = [
        col
        for col in ["dataset", "model", "_batch_sort", "mode_order", "experiment_type"]
        if col in table.columns
    ]
    sorted_by = "matrix"
    if args.sort_by == "metric" and args.metric in table.columns:
        table = table.sort_values(args.metric, ascending=False, na_position="last")
        sorted_by = args.metric
    elif args.sort_by == "metric":
        print(
            f"[compare] Metric column '{args.metric}' is absent; falling back to matrix sort."
        )
        if sort_cols:
            table = table.sort_values(sort_cols, na_position="last")
    elif sort_cols:
        table = table.sort_values(sort_cols, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n=== Experiment: {args.experiment} | Sorted by: {sorted_by} ===\n")
    printable = table.drop(columns=[c for c in ["_batch_sort"] if c in table.columns])
    print(printable.to_string(index=False))

    if args.output:
        printable.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
