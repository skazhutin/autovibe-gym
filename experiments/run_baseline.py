"""
Single-shot baseline: one LLM call, no iterative feedback, no checklist.

Usage:
    python -m experiments.run_baseline --dataset-dir datasets/example_dry_bean/prepared --mode local
"""
import argparse
import json
import os
import re
import time

import mlflow
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from experiments.mlflow_config import configure_mlflow_tracking
from gym.datasets import load_dataset_splits, resolve_metric
from gym.executor import CodeExecutor
from gym.llm import default_model_name, make_llm_client

if load_dotenv is not None:
    load_dotenv()

SYSTEM_PROMPT = """You are an expert data scientist. Solve the ML task completely in one Python code block.
Available variables: train_df, val_df, target_col, pd, np.
Do NOT access test data.
At the end of your code, assign your best trained model to a variable called `model`.
Output only a single ```python ... ``` block, nothing else."""


def extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", help="Directory with train/val/test CSV + meta.json")
    source.add_argument("--dataset", help="Single CSV file; requires --target")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--executor-backend", default=None)
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=60)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    max_tokens = args.max_tokens or (8192 if args.mode == "local" else 4096)
    splits = load_dataset_splits(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        target_col=args.target,
        seed=args.seed,
    )
    metric_fn, metric_name = resolve_metric(
        splits.metadata,
        splits.train[splits.target_col],
    )
    train, val, test = splits.train, splits.val, splits.test
    target_col = splits.target_col
    dataset_source = args.dataset_dir or args.dataset or ""
    dataset_name = splits.metadata.name or os.path.splitext(
        os.path.basename(dataset_source.rstrip("/\\"))
    )[0]
    model_name = args.model or default_model_name()
    run_name = args.run_name or f"baseline_{dataset_name}_{model_name.split('/')[-1]}"

    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name}\n\n"
        f"Training data shape: {train.shape}\n"
        f"Validation data shape: {val.shape}\n\n"
        f"Dataset statistics:\n{train.describe(include='all').to_string()}\n\n"
        "Variables available: train_df, val_df, target_col, pd, np\n"
        "Assign your best trained model to: model"
    )

    client = make_llm_client()
    configure_mlflow_tracking(mlflow)
    mlflow.set_experiment(args.experiment_name)
    started = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset_suite": splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": "baseline_single_shot",
            "max_tokens": max_tokens,
            "executor_backend": args.executor_backend or os.getenv("AUTOVIBE_EXECUTOR_BACKEND", "docker"),
        })

        response = client.complete(
            model=model_name,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task_prompt}],
        )
        code = extract_code(response.text)

        executor = CodeExecutor(
            timeout=args.sandbox_timeout,
            backend=args.executor_backend,
            docker_image=args.sandbox_image,
        )
        namespace = {
            "train_df": train.copy(),
            "val_df": val.copy(),
            "target_col": target_col,
            "pd": pd,
            "np": np,
        }
        stdout, stderr, namespace = executor.run(code, namespace)

        test_metric = None
        model_obj = namespace.get("model") or namespace.get("best_model")
        if model_obj is None:
            for value in namespace.values():
                if callable(getattr(value, "predict", None)):
                    model_obj = value
                    break
        if model_obj is not None:
            try:
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]
                preds = model_obj.predict(X_test)
                test_metric = float(metric_fn(y_test, preds))
            except Exception as exc:
                stderr += f"\n[submit error] {exc}"

        elapsed = round(time.time() - started, 1)
        summary = {
            "experiment_type": "baseline_single_shot",
            "model": model_name,
            "dataset": dataset_name,
            "test_metric": test_metric,
            "has_test_metric": test_metric is not None,
            "submit_failed": test_metric is None,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "elapsed_seconds": elapsed,
            "has_error": bool(stderr.strip()),
            "code_length": len(code),
        }

        metrics = {
            "has_test_metric": int(test_metric is not None),
            "submit_failed": int(test_metric is None),
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "elapsed_seconds": elapsed,
        }
        if test_metric is not None:
            metrics["test_metric"] = test_metric
        mlflow.log_metrics(metrics)

    print("\n=== Baseline Summary ===")
    print(json.dumps(summary, indent=2))
    if stderr.strip():
        print("\n[STDERR]", stderr[:500])


if __name__ == "__main__":
    main()
