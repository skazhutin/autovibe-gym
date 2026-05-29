"""
Gym experiment runner: iterative LLM agent with checklist feedback.

Usage:
    python3 -m experiments.run_gym --dataset-dir datasets/wine_quality --mode local
    python3 -m experiments.run_gym --dataset-dir datasets/wine_quality --mode cloud --model gpt-4o
"""
import argparse
import json
import os

from experiments.mlflow_config import configure_mlflow_tracking
from gym import GymAgent, NotebookGymEnv
from gym.datasets import (
    DatasetSplits,
    load_dataset_splits,
    metric_from_name,
    resolve_metric,
)
from gym.llm import default_model_name

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


def _kernel_backend_label() -> str:
    backend = os.getenv("AUTOVIBE_KERNEL_BACKEND", "local").strip().lower()
    return "jupyter-docker" if backend == "docker" else "jupyter-local"


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
    parser.add_argument("--executor-backend", default=None, help="Legacy GymEnv option; ignored by the Jupyter backend.")
    parser.add_argument("--sandbox-image", default=None, help="Legacy GymEnv option; ignored by the Jupyter backend.")
    parser.add_argument(
        "--episode-mode",
        choices=["gym_with_checklist", "iterative_no_checklist"],
        default="gym_with_checklist",
    )
    parser.add_argument("--workspace-dir", default=None)
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
    model_name = args.model or default_model_name()
    run_name = args.run_name or f"gym_{dataset_name}_{model_name.split('/')[-1]}"

    import mlflow

    configure_mlflow_tracking(mlflow)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "mode": args.mode,
            "episode_mode": args.episode_mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": args.episode_mode,
            "protocol_version": NotebookGymEnv.protocol_version,
            "notebook_backend": "jupyter",
            "checklist_version": NotebookGymEnv.checklist_version,
            "feedback_policy_version": NotebookGymEnv.feedback_policy_version,
            "max_steps": max_steps,
            "max_agent_turns": max_steps,
            "max_tokens": max_tokens,
            "token_budget": max_tokens,
            "sandbox_timeout": sandbox_timeout,
            "executor_backend": _kernel_backend_label(),
            "kernel_backend": os.getenv("AUTOVIBE_KERNEL_BACKEND", "local").strip().lower(),
            "dataset_suite": splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
        })

        env = NotebookGymEnv(
            train=splits.train,
            val=splits.val,
            test=splits.test,
            target_col=splits.target_col,
            metric_fn=metric_fn,
            metric_name=metric_name,
            max_steps=max_steps,
            workspace_dir=args.workspace_dir,
            mode=args.episode_mode,
            kernel_timeout=sandbox_timeout,
        )

        agent = GymAgent(env=env, model=model_name, max_tokens=max_tokens)
        try:
            summary = agent.run()
        finally:
            env.close()

        has_test_metric = summary.get("final_test_metric") is not None
        metrics = {
            "checklist_coverage": summary["checklist_coverage"],
            "private_checklist_coverage": summary.get("private_checklist_coverage", 0),
            "steps_used": summary["steps_used"],
            "error_count": summary.get("error_count", summary.get("errors_count", 0)),
            "has_test_metric": int(has_test_metric),
            "valid_submit": int(bool(summary.get("valid_submit"))),
            "submit_failed": int(summary.get("submitted") and not has_test_metric),
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
            "notebook_cells_final": summary.get("notebook_cells_final", 0),
            "notebook_revisions_total": summary.get("notebook_revisions_total", 0),
            "cell_executions_total": summary.get("cell_executions_total", 0),
            "kernel_restarts_total": summary.get("kernel_restarts_total", 0),
            "clean_runs_total": summary.get("clean_runs_total", 0),
            "successful_clean_run": summary.get("successful_clean_run", 0),
            "validation_calls_total": summary.get("validation_calls_total", 0),
            "contract_feedback_count": summary.get("contract_feedback_count", 0),
            "model_check_failure_count": summary.get("model_check_failure_count", 0),
            "checklist_hints_shown_total": summary.get("checklist_hints_shown_total", 0),
        }
        if summary.get("best_validation_metric") is not None:
            metrics["best_validation_metric"] = summary["best_validation_metric"]
        if has_test_metric:
            metrics["final_test_metric"] = summary["final_test_metric"]
            metrics["test_metric"] = summary["final_test_metric"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(summary["episode_workspace"], artifact_path="episode")
        if summary.get("private_episode_dir"):
            mlflow.log_artifacts(
                summary["private_episode_dir"],
                artifact_path="episode_private",
            )

    print("\n=== Run Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
