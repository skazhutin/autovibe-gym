"""
Multi-shot experiment: iterative LLM calls with execution feedback only.
No checklist hints — pure iteration as the control condition.

Ablation ladder:
  baseline   → 1 shot,  no feedback
  multishot  → N shots, execution feedback only   ← this script
  gym        → N shots, execution feedback + checklist hints

Usage:
    python -m experiments.run_multishot --dataset-dir datasets/wine_quality --mode local
    python -m experiments.run_multishot --dataset-dir datasets/wine_quality --shots 5
"""
import argparse
import json
import os
import re

import mlflow
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from experiments.run_gym import load_dataset, get_metric_fn
from gym.executor import CodeExecutor

load_dotenv()

SYSTEM_PROMPT = """You are an expert data scientist solving a supervised machine learning task.

You work in an iterative environment. Each turn you write Python code that gets executed.
Available variables in your namespace:
  train_df   — training DataFrame
  val_df     — validation DataFrame
  target_col — name of the target column (string)

Rules:
- Do NOT access test data — it is hidden until you are done.
- Write clean, executable Python inside a single ```python ... ``` block.
- Each round you will see the output of your previous code. Improve on it.
- When you are satisfied with your model, output exactly: SUBMIT
  (your best trained model must be assigned to variable `model` or `best_model`)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--shots", type=int, default=None,
                        help="Max LLM calls (default: 10 local / 5 cloud)")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    # Mode defaults
    if args.mode == "local":
        max_shots = args.shots or 10
        max_tokens = args.max_tokens or 8192
        sandbox_timeout = args.sandbox_timeout or 60
    else:
        max_shots = args.shots or 5
        max_tokens = args.max_tokens or 4096
        sandbox_timeout = args.sandbox_timeout or 30

    train, val, test, meta = load_dataset(args.dataset_dir)
    metric_fn, metric_name = get_metric_fn(meta["metric"])
    target_col = meta["target_col"]
    dataset_name = os.path.basename(args.dataset_dir.rstrip("/\\"))
    model_name = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    run_name = args.run_name or f"multishot{max_shots}_{dataset_name}_{model_name.split('/')[-1]}"

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "local"),
    )
    executor = CodeExecutor(timeout=sandbox_timeout)

    task_prompt = (
        f"Solve a supervised ML task.\n"
        f"Target column: '{target_col}'\n"
        f"Metric: {metric_name}\n\n"
        f"Training data shape: {train.shape}\n"
        f"Validation data shape: {val.shape}\n\n"
        f"Dataset statistics:\n{train.describe(include='all').to_string()}\n\n"
        "Variables available: train_df, val_df, target_col\n"
        "Assign your best model to: model"
    )

    namespace = {
        "train_df": train.copy(),
        "val_df": val.copy(),
        "target_col": target_col,
        "pd": pd,
    }

    messages = [{"role": "user", "content": task_prompt}]
    total_input_tokens = 0
    total_output_tokens = 0
    errors_count = 0
    shots_used = 0
    test_metric = None
    model_obj = None

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "mode": args.mode,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_type": f"multishot_{max_shots}",
            "max_shots": max_shots,
            "max_tokens": max_tokens,
            "sandbox_timeout": sandbox_timeout,
        })

        for shot in range(max_shots):
            shots_used = shot + 1

            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
            usage = response.usage
            if usage:
                total_input_tokens += usage.prompt_tokens
                total_output_tokens += usage.completion_tokens

            llm_text = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": llm_text})

            # Check for SUBMIT
            if "SUBMIT" in llm_text.upper():
                model_obj = namespace.get("model") or namespace.get("best_model")
                if model_obj is None:
                    feedback = (
                        "[ERROR] No variable named 'model' or 'best_model' found. "
                        "Train and assign your model first, then output SUBMIT."
                    )
                    messages.append({"role": "user", "content": feedback})
                    continue
                break

            code = _extract_code(llm_text)
            stdout, stderr, namespace = executor.run(code, namespace)
            if stderr.strip():
                errors_count += 1

            budget = max_shots - shots_used
            feedback = _build_feedback(stdout, stderr, budget)
            messages.append({"role": "user", "content": feedback})

            # Update model_obj from namespace each step (use latest)
            candidate = namespace.get("model") or namespace.get("best_model")
            if candidate is not None:
                model_obj = candidate

            if budget == 0:
                break

        # Evaluate on test set
        if model_obj is not None:
            try:
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]
                preds = model_obj.predict(X_test)
                test_metric = metric_fn(y_test, preds)
            except Exception as e:
                errors_count += 1
                print(f"[submit error] {e}")

        summary = {
            "experiment_type": f"multishot_{max_shots}",
            "model": model_name,
            "dataset": dataset_name,
            "test_metric": test_metric,
            "shots_used": shots_used,
            "errors_count": errors_count,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }

        mlflow.log_metrics({
            "test_metric": test_metric or 0.0,
            "shots_used": shots_used,
            "error_count": errors_count,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        })

    print("\n=== Multishot Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
