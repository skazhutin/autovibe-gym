"""Common experiment runner with first-class product-mode selection.

Examples:
    python -m experiments.run --dataset-dir datasets/example_dry_bean/prepared --mode gym_with_checklist
    python -m experiments.run --dataset-dir datasets/example_dry_bean/prepared --mode all --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from experiments.modes import (
    ALL_REQUESTED_MODE,
    ProductMode,
    expand_requested_mode,
    mode_metadata_cli_args,
    normalize_mode_key,
)


def _new_batch_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _source_args(args: argparse.Namespace) -> list[str]:
    if args.dataset_dir:
        return ["--dataset-dir", args.dataset_dir]
    out = ["--dataset", args.dataset]
    if args.target:
        out.extend(["--target", args.target])
    return out


def build_command(
    args: argparse.Namespace,
    mode: ProductMode,
    *,
    requested_mode: str,
    batch_id: str | None,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        mode.module,
        *_source_args(args),
        "--mode",
        args.budget_mode,
        "--experiment-name",
        args.experiment_name,
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.max_tokens:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.sandbox_timeout:
        command.extend(["--sandbox-timeout", str(args.sandbox_timeout)])
    if args.run_name:
        run_name = args.run_name
        if requested_mode == ALL_REQUESTED_MODE:
            run_name = f"{run_name}_{mode.key}"
        command.extend(["--run-name", run_name])
    if args.max_steps and mode.module in {"experiments.run_gym", "experiments.run_fixed"}:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.shots and mode.key == "repeated_single_shot":
        command.extend(["--shots", str(args.shots)])
    if mode.episode_mode:
        command.extend(["--episode-mode", mode.episode_mode])
    if args.executor_backend and mode.module != "experiments.run_fixed":
        command.extend(["--executor-backend", args.executor_backend])
    if args.sandbox_image and mode.module != "experiments.run_fixed":
        command.extend(["--sandbox-image", args.sandbox_image])
    if args.workspace_dir:
        workspace = Path(args.workspace_dir)
        if requested_mode == ALL_REQUESTED_MODE:
            workspace = workspace / mode.key
        command.extend(["--workspace-dir", str(workspace)])
    command.extend(
        mode_metadata_cli_args(
            mode,
            requested_mode=requested_mode,
            batch_id=batch_id,
        )
    )
    command.extend(extra_args)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AutoVibe product mode or all product modes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", help="Directory with train/val/test CSV + meta.json")
    source.add_argument("--dataset", help="Single CSV file; requires --target")
    parser.add_argument("--target")
    parser.add_argument(
        "--mode",
        required=True,
        help=(
            "Product mode: single_shot, repeated_single_shot, iterative_no_checklist, "
            "gym_with_checklist, fixed_transitions, or all."
        ),
    )
    parser.add_argument(
        "--budget-mode",
        choices=["local", "cloud"],
        default="local",
        help="Runner budget preset passed to the underlying run_* script.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--executor-backend", default=None)
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args, extra = parser.parse_known_args()

    requested_mode = normalize_mode_key(args.mode)
    modes = expand_requested_mode(args.mode)
    batch_id = args.batch_id
    if requested_mode == ALL_REQUESTED_MODE:
        batch_id = batch_id or _new_batch_id()

    commands = [
        (
            mode,
            build_command(
                args,
                mode,
                requested_mode=requested_mode,
                batch_id=batch_id,
                extra_args=extra,
            ),
        )
        for mode in modes
    ]
    dataset_label = args.dataset_dir or args.dataset
    print(
        f"[run] requested_mode={requested_mode} dataset={dataset_label} "
        f"model={args.model or '(env default)'} batch_id={batch_id or '-'}"
    )
    print(f"[run] Planned {len(commands)} run(s)")

    results = []
    for index, (mode, command) in enumerate(commands, 1):
        label = f"{mode.mode_order}. {mode.key}"
        print(f"[run] [{index}/{len(commands)}] {label}")
        print("  " + " ".join(command))
        if args.dry_run:
            results.append(
                {
                    "product_mode": mode.key,
                    "requested_mode": requested_mode,
                    "batch_id": batch_id,
                    "command": command,
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                    "dry_run": True,
                }
            )
            continue

        started = time.time()
        completed = subprocess.run(command)
        elapsed = round(time.time() - started, 1)
        results.append(
            {
                "product_mode": mode.key,
                "requested_mode": requested_mode,
                "batch_id": batch_id,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "dry_run": False,
            }
        )
        status = "OK" if completed.returncode == 0 else f"FAILED ({completed.returncode})"
        print(f"  -> {status} in {elapsed}s")
        if completed.returncode != 0 and args.stop_on_failure:
            print(json.dumps(results, indent=2))
            sys.exit(completed.returncode)

    print("\n[run] Summary")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
