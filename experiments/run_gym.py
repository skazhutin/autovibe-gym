"""
Gym experiment runner: iterative LLM agent with checklist feedback.

Usage:
    python -m experiments.run_gym --dataset-dir datasets/wine_quality --mode local
    python -m experiments.run_gym --dataset-dir datasets/wine_quality --mode cloud --model gpt-4o
"""
import argparse
import json
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, mean_squared_error

from gym import GymAgent, GymEnv

load_dotenv()

MODE_DEFAULTS = {
    "local": {"max_steps": 30, "max_tokens": 8192, "sandbox_timeout": 60},
    "cloud": {"max_steps": 15, "max_tokens": 4096, "sandbox_timeout": 30},
}


def load_dataset(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    meta_path = os.path.join(dataset_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    val = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
    test = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
    return train, val, test, meta


def get_metric_fn(metric_name: str):
    if metric_name == "f1_weighted":
        return lambda y, p: f1_score(y, p, average="weighted"), "f1_weighted"
    if metric_name == "neg_rmse":
        return lambda y, p: -(mean_squared_error(y, p) ** 0.5), "neg_rmse"
    raise ValueError(f"Unknown metric: {metric_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset dir with train/val/test/meta.json")
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL env var")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    max_steps = args.max_steps or defaults["max_steps"]
    max_tokens = args.max_tokens or defaults["max_tokens"]
    sandbox_timeout = args.sandbox_timeout or defaults["sandbox_timeout"]

    train, val, test, meta = load_dataset(args.dataset_dir)
    metric_fn, metric_name = get_metric_fn(meta["metric"])
    target_col = meta["target_col"]
    dataset_name = os.path.basename(args.dataset_dir.rstrip("/\\"))

    model_name = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    run_name = args.run_name or f"gym_{dataset_name}_{model_name.split('/')[-1]}"

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": "gym",
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "sandbox_timeout": sandbox_timeout,
        })

        env = GymEnv(
            train=train,
            val=val,
            test=test,
            target_col=target_col,
            metric_fn=metric_fn,
            metric_name=metric_name,
            max_steps=max_steps,
            sandbox_timeout=sandbox_timeout,
        )
        env.state.namespace = {
            "train_df": train.copy(),
            "val_df": val.copy(),
            "target_col": target_col,
            "pd": pd,
        }

        agent = GymAgent(env=env, model=model_name, max_tokens=max_tokens)
        summary = agent.run()

        mlflow.log_metrics({
            "test_metric": summary.get("test_metric") or 0.0,
            "checklist_coverage": summary["checklist_coverage"],
            "steps_used": summary["steps_used"],
            "error_count": summary["error_count"],
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
        })

    print("\n=== Run Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
