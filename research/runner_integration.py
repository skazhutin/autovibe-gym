"""Opt-in integration of Paper V1 run artifacts with existing runners."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from research.budget import (
    BudgetedExecutor,
    BudgetedLLMClient,
    EpisodeBudget,
    EpisodeBudgetPolicy,
)
from research.run_artifacts import (
    RecordingExecutor,
    RecordingLLMClient,
    RunRecorder,
    canonical_hash,
    file_set_hash,
)


def add_research_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--research-run-dir", default=None)
    parser.add_argument("--research-experiment-id", default=None)
    parser.add_argument("--research-replicate-index", type=int, default=0)
    parser.add_argument("--research-arm", choices=["A", "B", "C"], default=None)
    parser.add_argument("--research-model-version", default="unversioned")
    parser.add_argument("--research-split-id", default=None)
    parser.add_argument("--research-condition-id", default=None)
    parser.add_argument("--research-run-id", default=None)
    parser.add_argument("--research-rerun-of", default=None)
    parser.add_argument("--research-rerun-reason", default=None)
    parser.add_argument("--research-total-token-limit", type=int, default=64_000)
    parser.add_argument("--research-max-output-tokens", type=int, default=4_096)
    parser.add_argument("--research-max-llm-calls", type=int, default=12)
    parser.add_argument("--research-max-code-executions", type=int, default=20)
    parser.add_argument("--research-max-tool-calls", type=int, default=20)
    parser.add_argument("--research-wall-clock-limit", type=float, default=1_800.0)


def create_episode_budget(args: argparse.Namespace) -> EpisodeBudget | None:
    if not args.research_run_dir:
        return None
    return EpisodeBudget(
        EpisodeBudgetPolicy(
            total_token_limit=args.research_total_token_limit,
            max_output_tokens_per_call=args.research_max_output_tokens,
            max_llm_calls=args.research_max_llm_calls,
            max_code_executions=args.research_max_code_executions,
            max_tool_calls=args.research_max_tool_calls,
            wall_clock_limit_seconds=args.research_wall_clock_limit,
        )
    )


def dataset_source_hash(*, dataset: str | None, dataset_dir: str | None) -> str:
    if dataset_dir:
        root = Path(dataset_dir)
        if (root / "prepared").is_dir() and not (root / "meta.json").is_file():
            root = root / "prepared"
        files = [root / name for name in ("meta.json", "train.csv", "val.csv", "test.csv")]
    elif dataset:
        files = [Path(dataset)]
    else:
        raise ValueError("dataset or dataset_dir is required for research provenance")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot hash missing dataset artifacts: {', '.join(missing)}")
    return file_set_hash(files)


def git_provenance(repo: str | Path | None = None) -> tuple[str, bool]:
    cwd = str(Path(repo).resolve()) if repo else None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def start_research_run(
    args: argparse.Namespace,
    *,
    arm: str,
    dataset_id: str,
    dataset_hash: str,
    split_seed: int,
    default_split_id: str,
    model_id: str,
    prompt_template: Any,
    decoding_config: Mapping[str, Any],
    budget_policy: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    episode_budget: EpisodeBudget | None = None,
) -> RunRecorder | None:
    if not args.research_run_dir:
        return None
    if not args.research_experiment_id:
        raise ValueError("--research-experiment-id is required with --research-run-dir")
    commit, dirty = git_provenance(Path(__file__).resolve().parents[1])
    provider = os.getenv("LLM_PROVIDER", "openai_compat")
    condition = {
        "experiment_id": args.research_experiment_id,
        "git_commit": commit,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "split_id": args.research_split_id or default_split_id,
        "split_seed": split_seed,
        "model_provider": provider,
        "model_id": model_id,
        "model_version": args.research_model_version,
        "decoding_config_hash": canonical_hash(decoding_config),
        "arm": args.research_arm or arm,
        "replicate_index": args.research_replicate_index,
        "budget_policy_hash": canonical_hash(budget_policy),
        "execution_policy_hash": canonical_hash(execution_policy),
        "prompt_template_hash": canonical_hash(prompt_template),
    }
    recorder = RunRecorder.create(
        args.research_run_dir,
        condition,
        expected_condition_id=args.research_condition_id,
        run_id=args.research_run_id,
        rerun_of=args.research_rerun_of,
        rerun_reason=args.research_rerun_reason,
        git_dirty=dirty,
    )
    if episode_budget is not None:
        episode_budget.set_event_sink(recorder.record_execution)
    return recorder


def wrap_llm(
    client: Any,
    recorder: RunRecorder | None,
    *,
    model: str,
    episode_budget: EpisodeBudget | None = None,
) -> Any:
    wrapped = client
    if recorder is not None:
        provider = str(recorder.manifest["condition"]["model_provider"])
        wrapped = RecordingLLMClient(wrapped, recorder, provider=provider, model=model)
    if episode_budget is not None:
        wrapped = BudgetedLLMClient(wrapped, episode_budget)
    return wrapped


def wrap_executor(
    executor: Any,
    recorder: RunRecorder | None,
    *,
    episode_budget: EpisodeBudget | None = None,
) -> Any:
    wrapped = RecordingExecutor(executor, recorder) if recorder else executor
    if episode_budget is not None:
        wrapped = BudgetedExecutor(wrapped, episode_budget)
    return wrapped


def finalize_research_run(
    recorder: RunRecorder | None,
    summary: dict[str, Any],
    *,
    notebook_events: list[Mapping[str, Any]] | None = None,
    episode_budget: EpisodeBudget | None = None,
) -> dict[str, Any]:
    if recorder is None:
        return summary
    if notebook_events:
        recorder.record_notebook_events(notebook_events)
    summary["research_condition_id"] = recorder.condition_id
    summary["research_run_id"] = recorder.run_id
    summary["research_manifest"] = str(recorder.manifest_path)
    if episode_budget is not None:
        summary["episode_budget"] = episode_budget.snapshot()
    recorder.finalize(summary)
    return summary


def research_mlflow_params(recorder: RunRecorder | None) -> dict[str, Any]:
    if recorder is None:
        return {}
    return {
        "research_experiment_id": recorder.manifest["experiment_id"],
        "research_condition_id": recorder.condition_id,
        "research_run_id": recorder.run_id,
        "research_manifest_schema": recorder.manifest["schema_version"],
    }
