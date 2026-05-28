"""
Single-shot baseline: one LLM call, no iterative feedback, no checklist.
Used as the comparison point for the Gym experiment.

Usage:
    python -m experiments.run_baseline --dataset-dir datasets/wine_quality --mode local
"""
import argparse
import json
import os
import re

import time

import mlflow
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from gym.agent import _default_client
from gym.datasets import load_dataset_splits, resolve_metric
from gym.executor import CodeExecutor

SYSTEM_PROMPT = """You are an expert data scientist. Solve the ML task completely in one Python code block.
Available variables: train_df, val_df, target_col.
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
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    max_tokens = args.max_tokens or (8192 if args.mode == "local" else 4096)

    splits = load_dataset_splits(dataset_dir=args.dataset_dir)
    metric_fn, metric_name = resolve_metric(splits.metadata, splits.train[splits.target_col])
    train, val, test = splits.train, splits.val, splits.test
    target_col   = splits.target_col
    dataset_name = splits.metadata.name or os.path.basename(args.dataset_dir.rstrip("/\\"))
    model_name   = args.model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
    run_name     = args.run_name or f"baseline_{dataset_name}_{model_name.split('/')[-1]}"

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

    llm = _default_client()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    _start = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment_type":        "baseline_single_shot",
            "mode":                   args.mode,
            "model":                  model_name,
            "dataset":                dataset_name,
            "max_tokens":             max_tokens,
            "dataset_suite":          splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role":           splits.metadata.role,
            "dataset_sampled":        str(splits.metadata.sampled),
        })

        response = llm.complete(
            model=model_name,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task_prompt}],
        )
        code = extract_code(response.text)

        executor = CodeExecutor(timeout=60)
        namespace = {
            "train_df":   train.copy(),
            "val_df":     val.copy(),
            "target_col": target_col,
            "pd":         pd,
        }
        stdout, stderr, namespace = executor.run(code, namespace)

        test_metric = None
        model_obj = namespace.get("model") or namespace.get("best_model")
        # Fallback: scan for any object with predict()
        if model_obj is None:
            for v in namespace.values():
                if callable(getattr(v, "predict", None)):
                    model_obj = v
                    break
        if model_obj is not None:
            try:
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]
                preds  = model_obj.predict(X_test)
                test_metric = float(metric_fn(y_test, preds))
            except Exception as e:
                stderr += f"\n[submit error] {e}"

        elapsed = round(time.time() - _start, 1)
        summary = {
            "experiment_type":  "baseline_single_shot",
            "model":            model_name,
            "dataset":          dataset_name,
            "test_metric":      test_metric,
            "input_tokens":     response.input_tokens,
            "output_tokens":    response.output_tokens,
            "elapsed_seconds":  elapsed,
            "has_error":        bool(stderr.strip()),
            "code_length":      len(code),
        }

        mlflow.log_metrics({
            "test_metric":     test_metric or 0.0,
            "input_tokens":    response.input_tokens,
            "output_tokens":   response.output_tokens,
            "elapsed_seconds": elapsed,
        })

    print("\n=== Baseline Summary ===")
    print(json.dumps(summary, indent=2))
    if stderr.strip():
        print("\n[STDERR]", stderr[:500])


if __name__ == "__main__":
    main()
