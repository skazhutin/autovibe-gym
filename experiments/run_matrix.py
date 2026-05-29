"""
Batch experiment matrix runner for notebook-era AutoVibe Gym.

Runs a configurable grid of (dataset × episode_mode) combinations and logs
each run to MLflow. Supports --dry-run for previewing the matrix without
executing experiments.

Usage:
    # Full default matrix (all example datasets × gym_with_checklist + iterative_no_checklist)
    python -m experiments.run_matrix --mode local

    # Custom datasets and modes
    python -m experiments.run_matrix \\
        --datasets datasets/example_student_dropout datasets/example_room_occupancy \\
        --episode-modes gym_with_checklist iterative_no_checklist \\
        --mode cloud --model gpt-4o

    # Preview without running
    python -m experiments.run_matrix --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

DEFAULT_EPISODE_MODES = ["gym_with_checklist", "iterative_no_checklist"]


def _discover_datasets(root: str = "datasets") -> list[str]:
    """Return all dataset directories that contain prepared/meta.json."""
    pattern = os.path.join(root, "*", "prepared", "meta.json")
    found = sorted(
        str(Path(p).parent.parent)
        for p in glob.glob(pattern)
    )
    return found


def _run_single(
    dataset_dir: str,
    episode_mode: str,
    mode: str,
    model: str | None,
    experiment_name: str,
    extra_args: list[str],
) -> int:
    """Run one gym experiment as a subprocess; return exit code."""
    cmd = [
        sys.executable, "-m", "experiments.run_gym",
        "--dataset-dir", dataset_dir,
        "--episode-mode", episode_mode,
        "--mode", mode,
        "--experiment-name", experiment_name,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(extra_args)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch experiment matrix runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directories to include. Default: auto-discover all prepared datasets.",
    )
    parser.add_argument(
        "--episode-modes",
        nargs="+",
        default=DEFAULT_EPISODE_MODES,
        dest="episode_modes",
        help=f"Episode modes to run. Default: {DEFAULT_EPISODE_MODES}",
    )
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "cloud"],
        help="Step budget and token budget preset (default: local).",
    )
    parser.add_argument("--model", default=None, help="LLM model override.")
    parser.add_argument(
        "--experiment-name",
        default="autovibe_matrix",
        dest="experiment_name",
        help="MLflow experiment name (default: autovibe_matrix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the planned runs without executing them.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        dest="stop_on_failure",
        help="Abort the matrix on the first non-zero exit code.",
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        dest="datasets_root",
        help="Root directory for dataset discovery (default: datasets).",
    )
    args, extra = parser.parse_known_args()

    datasets = args.datasets or _discover_datasets(args.datasets_root)
    if not datasets:
        print(
            f"[run_matrix] No datasets found under '{args.datasets_root}'. "
            "Prepare datasets first with scripts/prepare_datasets.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix = [(d, m) for d in datasets for m in args.episode_modes]
    total = len(matrix)

    print(f"[run_matrix] Matrix: {len(datasets)} dataset(s) x {len(args.episode_modes)} mode(s) = {total} run(s)")
    print(f"[run_matrix] mode={args.mode}  model={args.model or '(env default)'}  experiment={args.experiment_name}")
    print()

    for i, (dataset_dir, episode_mode) in enumerate(matrix, 1):
        dataset_label = Path(dataset_dir).name
        print(f"[run_matrix] [{i}/{total}] {dataset_label}  mode={episode_mode}")
        if args.dry_run:
            print(f"  -> (dry-run) python -m experiments.run_gym --dataset-dir {dataset_dir} --episode-mode {episode_mode} --mode {args.mode}")
            continue

        started = time.time()
        rc = _run_single(
            dataset_dir=dataset_dir,
            episode_mode=episode_mode,
            mode=args.mode,
            model=args.model,
            experiment_name=args.experiment_name,
            extra_args=extra,
        )
        elapsed = time.time() - started
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  -> {status}  ({elapsed:.0f}s)")

        if rc != 0 and args.stop_on_failure:
            print(f"[run_matrix] Aborting matrix on failure (--stop-on-failure).", file=sys.stderr)
            sys.exit(rc)

    if not args.dry_run:
        print(f"\n[run_matrix] Done. {total} run(s) complete.")


if __name__ == "__main__":
    main()
