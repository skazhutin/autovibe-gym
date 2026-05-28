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

import mlflow
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import f1_score, mean_squared_error

from experiments.run_gym import load_dataset, get_metric_fn
from gym.executor import CodeExecutor

load_dotenv()

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
    train, val, test, meta = load_dataset(args.dataset_dir)
    metric_fn, metric_name = get_metric_fn(meta["metric"])
    target_col = meta["target_col"]
    dataset_name = os.path.basename(args.dataset_dir.rstrip("/\\"))
    model_name = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    run_name = args.run_name or f"baseline_{dataset_name}_{model_name.split('/')[-1]}"

    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name}\n\n"
        f"Training data shape: {train.shape}\n"
        f"Validation data shape: {val.shape}\n\n"
        f"Dataset statistics:\n{train.describe(include='all').to_string()}\n\n"
        "Variables available: train_df, val_df, target_col\n"
        "Assign your best model to: model"
    )

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "local"),
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset_suite": meta.get("suite", "legacy"),
            "dataset_split_strategy": meta.get("split_strategy"),
            "dataset_role": meta.get("role"),
            "dataset_sampled": str(meta.get("sampled")),
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": "baseline_single_shot",
            "max_tokens": max_tokens,
        })

        response = client.chat.completions.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ],
        )
        usage = response.usage
        code = extract_code(response.choices[0].message.content)

        executor = CodeExecutor(timeout=60)
        namespace = {
            "train_df": train.copy(),
            "val_df": val.copy(),
            "target_col": target_col,
            "pd": pd,
        }
        stdout, stderr, namespace = executor.run(code, namespace)

        test_metric = None
        model_obj = namespace.get("model") or namespace.get("best_model")
        if model_obj is not None:
            try:
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]
                preds = model_obj.predict(X_test)
                test_metric = metric_fn(y_test, preds)
            except Exception as e:
                stderr += f"\n[submit error] {e}"

        summary = {
            "experiment_type": "baseline_single_shot",
            "model": model_name,
            "dataset": dataset_name,
            "test_metric": test_metric,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "has_error": bool(stderr.strip()),
            "code_length": len(code),
        }

        mlflow.log_metrics({
            "test_metric": test_metric or 0.0,
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
        })

    print("\n=== Baseline Summary ===")
    print(json.dumps(summary, indent=2))
    if stderr.strip():
        print("\n[STDERR]", stderr[:500])


if __name__ == "__main__":
    main()
