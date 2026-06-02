"""
Repeated single-shot: N independent attempts.

Each attempt is one LLM call and one fresh execution namespace. The only signal
shared between attempts is the best validation metric so far. The fair
iterative no-checklist control is `experiments.run_gym --episode-mode
iterative_no_checklist`.
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
from gym.llm import default_model_name, make_llm_client
from gym.protocol import Action
from gym.scoring import score_with_coercion

if load_dotenv is not None:
    load_dotenv()

MODE_DEFAULTS = {
    "local": {"max_attempts": 10, "max_tokens": 8192, "sandbox_timeout": 60},
    "cloud": {"max_attempts": 5, "max_tokens": 4096, "sandbox_timeout": 30},
}

SYSTEM_PROMPT = """You are an expert data scientist solving a supervised ML task.

You write a complete, self-contained Python solution in a single response.
The code will be executed once; you will not see stdout/stderr from it.

Available variables pre-loaded in the execution namespace:
  train_df   - training DataFrame
  val_df     - validation DataFrame
  target_col - target column name (string)
  pd, np     - pandas and numpy

Rules:
- Do NOT access test data; it is strictly hidden.
- Train your best model on train_df and evaluate on val_df if useful.
- Wrap ALL preprocessing (encoding, scaling, imputation) inside a single
  scikit-learn Pipeline / ColumnTransformer and assign that fitted Pipeline to
  `model`, so `model.predict(df)` works on raw, unprocessed DataFrame rows.
  Do NOT transform features outside the model — validation and test sets are raw.
- `model` MUST be already FITTED: call `model.fit(train_df.drop(columns=[target_col]),
  train_df[target_col])` before finishing. An unfitted estimator has no usable
  `.predict` and will be rejected.
- If you use GridSearchCV/RandomizedSearchCV: keep it small (cv<=3), call `.fit(X, y)`,
  then assign `search.best_estimator_` to `model` — not the unfitted search object.
- As the LAST line, verify: `_ = model.predict(val_df.drop(columns=[target_col]).head())`.
- Keep any hyperparameter search small (cv<=3); n_jobs=-1 is allowed.
- Target scikit-learn 1.7: rely on DEFAULT parameters and do NOT pass deprecated
  or removed arguments (e.g. `loss='auto'`, `multi_class=...`); omit a parameter
  if unsure and use defaults.
- Write only executable Python. Do not include markdown or explanations.
"""


def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _build_feedback(stdout: str, stderr: str, budget: int) -> str:
    parts = []
    if stdout.strip():
        parts.append(f"[OUTPUT]\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"[ERROR]\n{stderr.strip()}")
    parts.append(f"[BUDGET] {budget} shots remaining. Improve your solution or output SUBMIT.")
    return "\n\n".join(parts)


def _build_attempt_prompt(task_prompt: str, best_val: float | None, attempt: int) -> str:
    parts = [task_prompt]
    if best_val is not None:
        parts.append(
            f"\nPrevious best validation score across {attempt} attempt(s): {best_val:.4f}. "
            "Try to beat it with a different or improved approach."
        )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Repeated single-shot: N independent attempts, only best val score shared."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", help="Directory with train/val/test CSV + meta.json")
    source.add_argument("--dataset", help="Single CSV; requires --target")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--workspace-dir", default=None, help="Emit dashboard episode artifacts here.")
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--executor-backend", default=None)
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    add_mode_metadata_args(parser)
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    max_attempts = args.shots or defaults["max_attempts"]
    max_tokens = args.max_tokens or defaults["max_tokens"]
    sandbox_timeout = args.sandbox_timeout or defaults["sandbox_timeout"]

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
    target_col = splits.target_col
    train = splits.train
    val = splits.val
    test = splits.test
    dataset_source = args.dataset_dir or args.dataset or ""
    dataset_name = splits.metadata.name or os.path.splitext(
        os.path.basename(dataset_source.rstrip("/\\"))
    )[0]

    model_name = args.model or default_model_name()
    run_name = args.run_name or f"repeated_single_shot{max_attempts}_{dataset_name}_{model_name.split('/')[-1]}"

    client = make_llm_client()
    executor = CodeExecutor(
        timeout=sandbox_timeout,
        backend=args.executor_backend,
        docker_image=args.sandbox_image,
    )

    dataset_card = build_dataset_card(train, val, target_col, metric_name, max_chars=4500)
    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name} (higher is better)\n\n"
        f"{dataset_card}\n\n"
        "Workspace variables: train_df, val_df, target_col, pd, np\n"
        "Assign your trained model to variable: model"
    )

    configure_mlflow_tracking(mlflow)
    mlflow.set_experiment(args.experiment_name)
    started = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment_type": "repeated_single_shot",
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "max_attempts": max_attempts,
            "max_tokens": max_tokens,
            "sandbox_timeout": sandbox_timeout,
            "executor_backend": args.executor_backend or os.getenv("AUTOVIBE_EXECUTOR_BACKEND", "docker"),
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
            **mode_metadata_params(args, "repeated_single_shot"),
        })

        best_val: float | None = None
        best_model = None
        best_code = ""
        best_stdout = ""
        total_input_tokens = 0
        total_output_tokens = 0
        errors_count = 0
        attempt_log = []

        for attempt in range(max_attempts):
            prompt = _build_attempt_prompt(task_prompt, best_val, attempt)
            response = client.complete(
                model=model_name,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            try:
                action = Action.from_llm_response(response.text)
                code = action.code if action.type == "code" else ""
                parse_status = "ok"
                parse_error = None
            except Exception:
                code = _extract_code(response.text)
                parse_status = "fallback"
                parse_error = "Action parsing failed; extracted code fence/plain text."

            namespace = {
                "train_df": train.copy(),
                "val_df": val.copy(),
                "target_col": target_col,
                "pd": pd,
                "np": np,
            }
            stdout, stderr, namespace = executor.run(code, namespace)
            attempt_error = stderr.strip() or None
            if attempt_error:
                errors_count += 1

            model_obj = namespace.get("model") or namespace.get("best_model")
            if model_obj is None:
                for value in namespace.values():
                    if callable(getattr(value, "predict", None)):
                        model_obj = value
                        break

            val_metric = None
            raw_validation_ready = False
            preflight_error = None
            if model_obj is not None:
                X_val = val.drop(columns=[target_col])
                y_val = val[target_col]
                try:
                    try:
                        val_preds = model_obj.predict(X_val)
                    except Exception:
                        # Safety net: fit the LLM's model if it was left unfitted.
                        model_obj.fit(train.drop(columns=[target_col]), train[target_col])
                        val_preds = model_obj.predict(X_val)
                    raw_validation_ready = True
                    val_metric = score_with_coercion(metric_fn, y_val, val_preds)
                    if best_val is None or val_metric > best_val:
                        best_val = val_metric
                        best_model = model_obj
                        best_code = code
                        best_stdout = stdout
                except Exception as exc:
                    preflight_error = f"{type(exc).__name__}: {exc}"
                    attempt_error = (attempt_error or "") + f" [val_eval: {preflight_error}]"
                    errors_count += 1

            attempt_log.append({
                "attempt": attempt + 1,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "code_length": len(code),
                "parse_status": parse_status,
                "parse_error": parse_error,
                "execution_success": not bool(stderr.strip()),
                "val_metric": val_metric,
                "raw_validation_ready": raw_validation_ready,
                "submit_preflight_error": preflight_error,
                "error": attempt_error,
            })
            mlflow.log_text(code, f"attempt_{attempt + 1:02d}_solution.py")
            mlflow.log_text(stdout, f"attempt_{attempt + 1:02d}_stdout.txt")
            mlflow.log_text(stderr, f"attempt_{attempt + 1:02d}_stderr.txt")
            if val_metric is not None:
                mlflow.log_metric("val_metric", val_metric, step=attempt)

        test_metric = None
        final_status = "no_candidate_found"
        null_reason = "No raw-validation-ready model was produced."
        submit_failure_type = "no_candidate_found"
        finalize_path = "failed"
        if best_model is not None:
            try:
                best_model.predict(val.drop(columns=[target_col]).head(32))
            except Exception as exc:
                final_status = "submit_blocked_preflight"
                null_reason = f"{type(exc).__name__}: {exc}"
                submit_failure_type = type(exc).__name__
                finalize_path = "submit_preflight"
                errors_count += 1
                print(f"[submit preflight error] {exc}")
            else:
                try:
                    X_test = test.drop(columns=[target_col])
                    y_test = test[target_col]
                    test_preds = best_model.predict(X_test)
                    test_metric = score_with_coercion(metric_fn, y_test, test_preds)
                    final_status = "submitted_clean"
                    null_reason = None
                    submit_failure_type = None
                    finalize_path = "best_raw_validation_model"
                except Exception as exc:
                    final_status = "hidden_submit_failed"
                    null_reason = f"{type(exc).__name__}: {exc}"
                    submit_failure_type = type(exc).__name__
                    finalize_path = "hidden_test"
                    errors_count += 1
                    print(f"[submit error] {exc}")

        elapsed = round(time.time() - started, 1)
        mlflow.log_text(json.dumps(attempt_log, indent=2), "attempt_log.json")
        metrics = {
            "attempts_used": len(attempt_log),
            "error_count": errors_count,
            "has_test_metric": int(test_metric is not None),
            "valid_submit": int(test_metric is not None),
            "submit_failed": int(test_metric is None),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "elapsed_seconds": elapsed,
        }
        from experiments.dashboard_artifacts import checklist_coverage, write_episode_artifacts
        coverage = checklist_coverage(best_code, best_stdout, target_col) if best_code else None
        if coverage is not None:
            metrics["checklist_coverage"] = coverage
        metrics["steps_used"] = len(attempt_log)
        if args.workspace_dir and best_code:
            write_episode_artifacts(args.workspace_dir, code=best_code, stdout=best_stdout,
                                    target_col=target_col, coverage=coverage,
                                    steps=len(attempt_log))
        if best_val is not None:
            metrics["best_val_metric"] = best_val
            metrics["best_validation_metric"] = best_val
        if test_metric is not None:
            metrics["test_metric"] = test_metric
            metrics["final_test_metric"] = test_metric
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "final_status": final_status,
            "null_reason": null_reason or "",
            "finalize_path": finalize_path,
        })

    summary = {
        "experiment_type": "repeated_single_shot",
        **mode_metadata_params(args, "repeated_single_shot"),
        "model": model_name,
        "dataset": dataset_name,
        "attempts_used": len(attempt_log),
        "best_val_metric": best_val,
        "test_metric": test_metric,
        "has_test_metric": test_metric is not None,
        "submit_failed": test_metric is None,
        "valid_submit": test_metric is not None,
        "final_status": final_status,
        "null_reason": null_reason,
        "final_test_metric": test_metric,
        "submit_failure_type": submit_failure_type,
        "finalize_path": finalize_path,
        "errors_count": errors_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
    }
    print("\n=== Repeated Single-Shot Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
