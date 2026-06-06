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
from experiments.modes import add_mode_metadata_args, mode_metadata_params
from gym.data_profile import build_dataset_card
from gym.datasets import load_dataset_splits, resolve_metric
from gym.executor import CodeExecutor
from gym.llm import make_llm_client
from gym.model_config import apply_model_reference
from gym.scoring import score_with_coercion

if load_dotenv is not None:
    load_dotenv()

SYSTEM_PROMPT = """You are an expert data scientist. Solve the ML task completely in one Python code block.
Available variables: train_df, val_df, target_col, pd, np.
Do NOT access test data.

Requirements for the final model:
- Wrap ALL preprocessing (encoding, scaling, imputation) inside a single
  scikit-learn Pipeline / ColumnTransformer and assign that fitted Pipeline to
  `model`, so `model.predict(df)` works on raw, unprocessed DataFrame rows.
  Do NOT transform features outside the model — the hidden test set is raw.
- `model` MUST be already FITTED. Call `model.fit(X, y)` on the training data
  (X = train_df.drop(columns=[target_col]), y = train_df[target_col]) before you
  finish. An unfitted estimator has no usable `.predict` and will be rejected.
- If you use GridSearchCV/RandomizedSearchCV: keep it small (cv<=3, few
  candidates), call `.fit(X, y)` on it, then assign `search.best_estimator_`
  (the refitted Pipeline) to `model` — do NOT assign the unfitted search object.
- As the LAST lines, verify it works and only then keep `model`:
      _ = model.predict(val_df.drop(columns=[target_col]).head())
- n_jobs=-1 is allowed.
- Target scikit-learn 1.7: rely on DEFAULT parameters and do NOT pass deprecated
  or removed arguments (e.g. `loss='auto'`, `multi_class=...`). If unsure about a
  parameter, omit it and use the estimator's defaults.

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
    parser.add_argument("--model", required=True, help="Model id or name from the shared model registry")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--executor-backend", default=None)
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--workspace-dir", default=None, help="Emit dashboard episode artifacts here.")
    parser.add_argument("--sandbox-timeout", type=int, default=60)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    add_mode_metadata_args(parser)
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
    model_name = apply_model_reference(args.model)
    run_name = args.run_name or f"baseline_{dataset_name}_{model_name.split('/')[-1]}"

    dataset_card = build_dataset_card(train, val, target_col, metric_name, max_chars=4500)
    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name}\n\n"
        f"{dataset_card}\n\n"
        "Variables available: train_df, val_df, target_col, pd, np\n"
        "Assign your best trained model to: model"
    )

    client = make_llm_client()
    configure_mlflow_tracking(mlflow)
    mlflow.set_experiment(args.experiment_name)
    started = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": "baseline_single_shot",
            "max_tokens": max_tokens,
            "executor_backend": args.executor_backend or os.getenv("AUTOVIBE_EXECUTOR_BACKEND", "docker"),
            **mode_metadata_params(args, "single_shot"),
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
        final_status = "no_candidate_found"
        null_reason = "No predict-capable model variable was produced."
        submit_failure_type = "no_candidate_found"
        finalize_path = "failed"
        submit_error = ""
        model_obj = namespace.get("model") or namespace.get("best_model")
        if model_obj is None:
            for value in namespace.values():
                if callable(getattr(value, "predict", None)):
                    model_obj = value
                    break
        if model_obj is not None:
            X_val = val.drop(columns=[target_col]).head(32)
            preflight_ok = True
            try:
                model_obj.predict(X_val)
            except Exception:
                # Safety net: the LLM defined the model but may not have fitted it.
                # Fit the chosen architecture on train and retry before rejecting.
                try:
                    model_obj.fit(train.drop(columns=[target_col]), train[target_col])
                    model_obj.predict(X_val)
                    finalize_path = "single_shot_autofit"
                except Exception as exc:
                    preflight_ok = False
                    final_status = "submit_blocked_preflight"
                    null_reason = f"{type(exc).__name__}: {exc}"
                    submit_failure_type = type(exc).__name__
                    finalize_path = "submit_preflight"
                    submit_error = null_reason
                    stderr += f"\n[submit preflight error] {submit_error}"
            if preflight_ok:
                try:
                    X_test = test.drop(columns=[target_col])
                    y_test = test[target_col]
                    preds = model_obj.predict(X_test)
                    test_metric = score_with_coercion(metric_fn, y_test, preds)
                    final_status = "submitted_clean"
                    null_reason = None
                    submit_failure_type = None
                    finalize_path = "single_shot_model"
                except Exception as exc:
                    final_status = "hidden_submit_failed"
                    null_reason = f"{type(exc).__name__}: {exc}"
                    submit_failure_type = type(exc).__name__
                    finalize_path = "hidden_test"
                    submit_error = null_reason
                    stderr += f"\n[submit error] {submit_error}"

        elapsed = round(time.time() - started, 1)
        summary = {
            "experiment_type": "baseline_single_shot",
            **mode_metadata_params(args, "single_shot"),
            "model": model_name,
            "dataset": dataset_name,
            "test_metric": test_metric,
            "has_test_metric": test_metric is not None,
            "submit_failed": test_metric is None,
            "valid_submit": test_metric is not None,
            "final_status": final_status,
            "null_reason": null_reason,
            "final_test_metric": test_metric,
            "submit_failure_type": submit_failure_type,
            "finalize_path": finalize_path,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "elapsed_seconds": elapsed,
            "has_error": bool(stderr.strip()),
            "code_length": len(code),
        }

        # Checklist coverage is a comparison metric for every mode (even though
        # single-shot gets no hints): measure it from the generated code.
        from experiments.dashboard_artifacts import checklist_coverage, write_episode_artifacts
        coverage = checklist_coverage(code, stdout, target_col)
        summary["checklist_coverage"] = coverage
        summary["steps_used"] = 1
        if args.workspace_dir:
            err_name = (submit_failure_type or "Error") if (test_metric is None and stderr.strip()) else None
            write_episode_artifacts(args.workspace_dir, code=code, stdout=stdout,
                                    stderr=stderr, error_name=err_name,
                                    target_col=target_col, coverage=coverage, steps=1)
            # Best-effort self-summary from the single-shot exchange (no hidden
            # score in scope), persisted for the dashboard «Мысли» tab. Generated
            # once the model produced a usable solution (a predict-capable
            # candidate), even if the hidden test later rejected it.
            if model_obj is not None:
                from gym.run_summary import generate_and_write

                generate_and_write(
                    client,
                    model_name,
                    args.workspace_dir,
                    conversation=[
                        {"role": "user", "content": task_prompt},
                        {"role": "assistant", "content": response.text},
                    ],
                    max_tokens=min(max_tokens, 700),
                )

        metrics = {
            "has_test_metric": int(test_metric is not None),
            "valid_submit": int(test_metric is not None),
            "submit_failed": int(test_metric is None),
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "elapsed_seconds": elapsed,
            "steps_used": 1,
        }
        if coverage is not None:
            metrics["checklist_coverage"] = coverage
        mlflow.set_tags({
            "final_status": final_status,
            "null_reason": null_reason or "",
            "finalize_path": finalize_path,
        })
        if test_metric is not None:
            metrics["test_metric"] = test_metric
            metrics["final_test_metric"] = test_metric
        mlflow.log_metrics(metrics)
        mlflow.log_text(code, "generated_solution.py")
        mlflow.log_text(stdout, "stdout.txt")
        mlflow.log_text(stderr, "stderr.txt")
        mlflow.log_text(submit_error, "submit_error.txt")
        mlflow.log_text(json.dumps(summary, indent=2), "summary.json")

    print("\n=== Baseline Summary ===")
    print(json.dumps(summary, indent=2))
    if stderr.strip():
        print("\n[STDERR]", stderr[:500])


if __name__ == "__main__":
    main()
