"""
Repeated single-shot: N fully independent attempts.

Protocol (per TZ section "Режимы взаимодействия"):
  - Each attempt is a single LLM call → one code block executed → model evaluated on val.
  - Between attempts: agent sees ONLY the best validation metric so far. No traceback,
    no stdout, no checklist hints, no stage feedback.
  - Final: best attempt's model is evaluated on private test (once, irreversible).

This is the control condition that isolates "does knowing previous best val score help?"
from "does execution feedback help?" (multishot with feedback) and
"does checklist guidance help?" (gym / fixed transitions).

Usage:
    python -m experiments.run_multishot --dataset-dir datasets/student_dropout/prepared --mode local
    python -m experiments.run_multishot --dataset-dir datasets/student_dropout/prepared --shots 5
"""
import argparse
import json
import os
import time

import mlflow
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from experiments.run_gym import load_dataset, get_metric_fn
from gym.agent import _default_client
from gym.executor import CodeExecutor
from gym.protocol import Action

# Budget defaults — match run_gym.py for fair token-cost comparison
MODE_DEFAULTS = {
    "local":  {"max_attempts": 10, "max_tokens": 8192,  "sandbox_timeout": 60},
    "cloud":  {"max_attempts": 5,  "max_tokens": 4096,  "sandbox_timeout": 30},
}

SYSTEM_PROMPT = """You are an expert data scientist solving a supervised ML task.

You write a complete, self-contained Python solution in a single response.
The code will be executed once; you will not see any output from it.

Available variables pre-loaded in the execution namespace:
  train_df   — training DataFrame
  val_df     — validation DataFrame
  target_col — target column name (string)
  pd, np     — pandas and numpy

Rules:
- Do NOT access test data — it is strictly hidden.
- Train your best model on train_df, evaluate on val_df.
- Assign your final trained model to a variable called `model`.
  The model must have a .predict() method that works on raw DataFrame rows
  (same dtypes as train_df, without the target column).
- Do not print sensitive data or file paths.
- Write only executable Python. Do not include markdown or explanations.
"""


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
        description="Repeated single-shot: N independent attempts, only best val score shared between them."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", help="Directory with train/val/test CSV + meta.json")
    source.add_argument("--dataset", help="Single CSV (requires --target)")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of independent attempts (default: 10 local / 5 cloud)")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    max_attempts   = args.shots          or defaults["max_attempts"]
    max_tokens     = args.max_tokens     or defaults["max_tokens"]
    sandbox_timeout = args.sandbox_timeout or defaults["sandbox_timeout"]

    from gym.datasets import load_dataset_splits, resolve_metric
    splits = load_dataset_splits(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        target_col=args.target,
        seed=args.seed,
    )
    metric_fn, metric_name = resolve_metric(splits.metadata, splits.train[splits.target_col])
    target_col   = splits.target_col
    train        = splits.train
    val          = splits.val
    test         = splits.test
    dataset_name = splits.metadata.name or os.path.splitext(os.path.basename(
        (args.dataset_dir or args.dataset or "").rstrip("/\\")
    ))[0]

    model_name = args.model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
    run_name = args.run_name or f"repeated_ss_{dataset_name}_{model_name.split('/')[-1]}"

    llm      = _default_client()
    executor = CodeExecutor(timeout=sandbox_timeout)

    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name} (higher is better)\n\n"
        f"Training data shape: {train.shape}\n"
        f"Validation data shape: {val.shape}\n\n"
        f"Dataset statistics:\n{train.describe(include='all').to_string()}\n\n"
        "Workspace variables: train_df, val_df, target_col, pd, np\n"
        "Assign your trained model to variable: model"
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    _start = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment_type":        "repeated_single_shot",
            "mode":                   args.mode,
            "model":                  model_name,
            "dataset":                dataset_name,
            "max_attempts":           max_attempts,
            "max_tokens":             max_tokens,
            "sandbox_timeout":        sandbox_timeout,
            "dataset_suite":          splits.metadata.suite or "legacy",
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role":           splits.metadata.role,
            "dataset_sampled":        str(splits.metadata.sampled),
        })

        best_val    : float | None = None
        best_model  = None
        total_input_tokens  = 0
        total_output_tokens = 0
        errors_count        = 0
        attempt_log         = []  # [{attempt, val_metric, error}]

        for attempt in range(max_attempts):
            prompt = _build_attempt_prompt(task_prompt, best_val, attempt)

            response = llm.complete(
                model=model_name,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            total_input_tokens  += response.input_tokens
            total_output_tokens += response.output_tokens

            # Parse code from response (reuse Action parser for consistency)
            try:
                action = Action.from_llm_response(response.text)
                code = action.code if action.type == "code" else ""
            except Exception:
                code = response.text.strip()

            # Fresh namespace each attempt — no state shared
            namespace = {
                "train_df":   train.copy(),
                "val_df":     val.copy(),
                "target_col": target_col,
                "pd":         pd,
            }

            stdout, stderr, namespace = executor.run(code, namespace)
            attempt_error = stderr.strip() if stderr.strip() else None
            if attempt_error:
                errors_count += 1

            model_obj = namespace.get("model") or namespace.get("best_model")

            # Scan for any predict-able object if named variable not found
            if model_obj is None:
                for v in namespace.values():
                    if callable(getattr(v, "predict", None)):
                        model_obj = v
                        break

            val_metric = None
            if model_obj is not None:
                try:
                    X_val = val.drop(columns=[target_col])
                    y_val = val[target_col]
                    val_preds = model_obj.predict(X_val)
                    val_metric = float(metric_fn(y_val, val_preds))
                    if best_val is None or val_metric > best_val:
                        best_val   = val_metric
                        best_model = model_obj
                except Exception as e:
                    attempt_error = (attempt_error or "") + f" [val_eval: {e}]"
                    errors_count += 1

            attempt_log.append({
                "attempt":    attempt + 1,
                "val_metric": val_metric,
                "error":      attempt_error,
            })
            mlflow.log_metric("val_metric", val_metric or 0.0, step=attempt)

        # Private test — once, on best model
        test_metric = None
        if best_model is not None:
            try:
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]
                test_preds = best_model.predict(X_test)
                test_metric = float(metric_fn(y_test, test_preds))
            except Exception as e:
                errors_count += 1
                print(f"[submit error] {e}")

        elapsed = round(time.time() - _start, 1)

        mlflow.log_text(json.dumps(attempt_log, indent=2), "attempt_log.json")
        mlflow.log_metrics({
            "test_metric":    test_metric or 0.0,
            "best_val_metric": best_val   or 0.0,
            "attempts_used":  len(attempt_log),
            "error_count":    errors_count,
            "input_tokens":   total_input_tokens,
            "output_tokens":  total_output_tokens,
            "elapsed_seconds": elapsed,
        })

    summary = {
        "experiment_type": "repeated_single_shot",
        "model":           model_name,
        "dataset":         dataset_name,
        "attempts_used":   len(attempt_log),
        "best_val_metric": best_val,
        "test_metric":     test_metric,
        "errors_count":    errors_count,
        "input_tokens":    total_input_tokens,
        "output_tokens":   total_output_tokens,
        "elapsed_seconds": elapsed,
    }
    print("\n=== Repeated Single-Shot Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
