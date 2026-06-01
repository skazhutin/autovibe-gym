"""Run the four AutoVibe product modes across datasets and models.

Usage:
    python -m experiments.run_all_modes_matrix \
      --datasets datasets/example_student_dropout/prepared datasets/example_dry_bean/prepared \
      --models deepseek-v4-flash gemma-4-26b \
      --mode cloud \
      --experiment-name autovibe-gym
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MODE_COMMANDS = {
    "single-shot": "experiments.run_baseline",
    "repeated single-shot": "experiments.run_multishot",
    "flexible gym": "experiments.run_gym",
    "fixed transitions": "experiments.run_fixed",
}


def _build_command(
    *,
    module: str,
    dataset: str,
    model: str,
    mode: str,
    experiment_name: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--dataset-dir",
        dataset,
        "--mode",
        mode,
        "--model",
        model,
        "--experiment-name",
        experiment_name,
    ]
    command.extend(extra_args)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all four AutoVibe product modes.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args, extra = parser.parse_known_args()

    plan = [
        (dataset, model, product_mode, module)
        for dataset in args.datasets
        for model in args.models
        for product_mode, module in MODE_COMMANDS.items()
    ]
    print(
        f"[run_all_modes_matrix] {len(args.datasets)} dataset(s) x "
        f"{len(args.models)} model(s) x 4 mode(s) = {len(plan)} run(s)"
    )

    results = []
    for index, (dataset, model, product_mode, module) in enumerate(plan, 1):
        command = _build_command(
            module=module,
            dataset=dataset,
            model=model,
            mode=args.mode,
            experiment_name=args.experiment_name,
            extra_args=extra,
        )
        label = f"{Path(dataset).name} | {model} | {product_mode}"
        print(f"[run_all_modes_matrix] [{index}/{len(plan)}] {label}")
        print("  " + " ".join(command))
        if args.dry_run:
            results.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "product_mode": product_mode,
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
                "dataset": dataset,
                "model": model,
                "product_mode": product_mode,
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

    print("\n[run_all_modes_matrix] Summary")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
