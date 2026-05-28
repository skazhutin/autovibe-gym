"""
Gym experiment runner: iterative LLM agent with checklist feedback.

Usage:
    python3 -m experiments.run_gym --dataset-dir datasets/wine_quality --mode local
    python3 -m experiments.run_gym --dataset-dir datasets/wine_quality --mode cloud --model gpt-4o
"""
import argparse
import json
import os

from gym import GymAgent, GymEnv
from gym.datasets import (
    DatasetSplits,
    load_dataset_splits,
    metric_from_name,
    resolve_metric,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

MODE_DEFAULTS = {
    "local": {"max_steps": 30, "max_tokens": 8192, "sandbox_timeout": 60},
    "cloud": {"max_steps": 15, "max_tokens": 4096, "sandbox_timeout": 30},
}


def load_dataset(dataset_dir: str):
    """Compatibility helper used by baseline runner."""
    splits = load_dataset_splits(dataset_dir=dataset_dir)
    meta = {
        "name": splits.metadata.name,
        "target_col": splits.target_col,
        "metric": splits.metadata.metric_name,
        "source": splits.metadata.source,
        "seed": splits.metadata.seed,
        "notes": splits.metadata.notes,
        "suite": splits.metadata.suite,
        "split_strategy": splits.metadata.split_strategy,
        "role": splits.metadata.role,
        "sampled": splits.metadata.sampled,
    }
    return splits.train, splits.val, splits.test, meta


def get_metric_fn(metric_name: str):
    return metric_from_name(metric_name), metric_name


def _dataset_name(splits: DatasetSplits, dataset_arg: str | None) -> str:
    if splits.metadata.name:
        return splits.metadata.name
    if dataset_arg:
        return os.path.splitext(os.path.basename(dataset_arg))[0]
    return "dataset"


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Single CSV file; requires --target")
    source.add_argument(
        "--dataset-dir",
        help="Directory with train.csv, val.csv, test.csv, meta.json",
    )
    parser.add_argument("--target", help="Target column for --dataset CSV mode")
    parser.add_argument("--seed", type=int, default=42)
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

    splits = load_dataset_splits(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        target_col=args.target,
        seed=args.seed,
    )
    metric_fn, metric_name = resolve_metric(
        splits.metadata, splits.train[splits.target_col]
    )

    dataset_name = _dataset_name(splits, args.dataset)
    model_name = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    run_name = args.run_name or f"gym_{dataset_name}_{model_name.split('/')[-1]}"

    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
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
            "dataset_suite": splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
        })

        env = GymEnv(
            train=splits.train,
            val=splits.val,
            test=splits.test,
            target_col=splits.target_col,
            metric_fn=metric_fn,
            metric_name=metric_name,
            max_steps=max_steps,
            sandbox_timeout=sandbox_timeout,
        )

        agent = GymAgent(env=env, model=model_name, max_tokens=max_tokens)
        summary = agent.run()
        mlflow.log_text(env.state.cell_history.to_markdown(), "cell_history.md")

        mlflow.log_metrics({
            "test_metric": summary.get("test_metric") or 0.0,
            "checklist_coverage": summary["checklist_coverage"],
            "steps_used": summary["steps_used"],
            "error_count": summary.get("error_count", summary.get("errors_count", 0)),
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
        })

    print("\n=== Run Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
