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
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from experiments.mlflow_config import configure_mlflow_tracking
from experiments.modes import add_mode_metadata_args, mode_metadata_params
from experiments.repeated_terminal import RepeatedSingleShotTerminalAdapter
from research.runner_integration import (
    add_research_artifact_args,
    add_research_v2_stopping_arg,
    budget_policy_payload,
    create_episode_budget,
    create_stopping_policy,
    dataset_source_hash,
    finalize_research_run,
    research_mlflow_params,
    start_research_run,
    wrap_executor,
    wrap_llm,
)
from research.budget import BudgetExhausted
from research.stopping import StoppingController
from research.submission import (
    HiddenEvaluationGate,
    prediction_backend_for_execution,
    validate_submission_candidate,
)
from gym.data_profile import build_dataset_card
from gym.candidate_store import CandidateBundleError
from gym.datasets import load_dataset_splits, resolve_metric
from gym.executor import CodeExecutor
from gym.llm import configured_temperature, make_llm_client
from gym.model_config import apply_model_reference
from gym.protocol import Action
from gym.scoring import score_with_coercion
from gym.terminal_contract import TERMINAL_CONTRACT_VERSION

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

TASK_PROMPT_TEMPLATE = """Solve a supervised ML task.
Target column: '{target_col}'
Metric: {metric_name} (higher is better)

{dataset_card}

Workspace variables: train_df, val_df, target_col, pd, np
Assign your trained model to variable: model"""

ATTEMPT_FEEDBACK_TEMPLATE = """Previous best validation score across {attempt} attempt(s): {best_val:.4f}. \
Try to beat it with a different or improved approach."""


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
            "\n"
            + ATTEMPT_FEEDBACK_TEMPLATE.format(
                attempt=attempt,
                best_val=best_val,
            )
        )
    return "\n".join(parts)


def _system_prompt_for_validation_boundary(*, labels_host_only: bool) -> str:
    if not labels_host_only:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT.replace(
            "  val_df     - validation DataFrame",
            "  val_df     - validation feature DataFrame; target labels are host-only",
        )
        .replace(
            "- Train your best model on train_df and evaluate on val_df if useful.",
            "- Train on train_df. Validation labels are host-only; the host evaluates submitted candidates.",
        )
        .replace(
            "- As the LAST line, verify: `_ = model.predict(val_df.drop(columns=[target_col]).head())`.",
            "- As the LAST line, verify: `_ = model.predict(val_df.head())`.",
        )
    )


def _symmetric_terminal_null_reason(final_status: str) -> str:
    return {
        "no_incumbent_for_finalization": (
            "No validated incumbent existed when protected finalization began."
        ),
        "invalid_candidate_artifact": (
            "The incumbent bundle failed integrity verification before replay."
        ),
        "protected_replay_failed": (
            "The immutable incumbent snapshot could not be replayed safely."
        ),
        "protected_replay_validation_failed": (
            "The replayed incumbent failed the common submission preflight."
        ),
        "protected_replay_validation_mismatch": (
            "The replayed validation metric did not match the frozen incumbent record."
        ),
        "protected_finalization_artifact_failed": (
            "The replayed model could not be stored and verified atomically."
        ),
        "hidden_submit_failed": (
            "The replayed artifact failed the single private hidden evaluation."
        ),
        "finalization_reserve_exhausted": (
            "The protected finalization reserve was exhausted before completion."
        ),
    }.get(
        final_status,
        "Symmetric terminal finalization produced no hidden score.",
    )


def _observe_repeated_boundary(
    controller: StoppingController,
    budget: Any,
    terminal: RepeatedSingleShotTerminalAdapter,
) -> None:
    incumbent = terminal.registry.incumbent()
    controller.observe_reserve_boundary(
        boundary_resources=budget.exploration_boundary_resources(),
        incumbent_candidate_id=(
            incumbent.candidate_id if incumbent is not None else None
        ),
    )


def _observe_repeated_candidate(
    controller: StoppingController,
    *,
    previous_incumbent: Any,
    record: Any,
    terminal: RepeatedSingleShotTerminalAdapter,
) -> None:
    incumbent = terminal.registry.incumbent()
    controller.observe_validation_attempt(
        eligible=True,
        improved=bool(
            incumbent is not None
            and incumbent.candidate_id == record.candidate_id
            and (
                previous_incumbent is None
                or previous_incumbent.candidate_id != record.candidate_id
            )
        ),
        candidate_id=record.candidate_id,
        incumbent_candidate_id=(
            incumbent.candidate_id if incumbent is not None else None
        ),
    )


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
    parser.add_argument("--model", required=True, help="Model id or name from the shared model registry")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--workspace-dir", default=None, help="Emit dashboard episode artifacts here.")
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--executor-backend", default=None)
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--experiment-name", default="autovibe-gym")
    parser.add_argument("--run-name", default=None)
    add_mode_metadata_args(parser)
    add_research_artifact_args(parser)
    add_research_v2_stopping_arg(parser)
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    max_attempts = args.shots or defaults["max_attempts"]
    max_tokens = args.max_tokens or defaults["max_tokens"]
    sandbox_timeout = args.sandbox_timeout or defaults["sandbox_timeout"]
    episode_budget = create_episode_budget(args)
    stopping_policy = create_stopping_policy(args, episode_budget)
    if episode_budget is not None:
        max_attempts = episode_budget.policy.max_llm_calls
        max_tokens = min(max_tokens, episode_budget.policy.max_output_tokens_per_call)

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
    validation_labels_host_only = bool(
        episode_budget is not None
        and episode_budget.has_validation_query_policy
    )
    agent_val = (
        val.drop(columns=[target_col])
        if validation_labels_host_only
        else val
    )
    system_prompt = _system_prompt_for_validation_boundary(
        labels_host_only=validation_labels_host_only
    )
    dataset_source = args.dataset_dir or args.dataset or ""
    dataset_name = splits.metadata.name or os.path.splitext(
        os.path.basename(dataset_source.rstrip("/\\"))
    )[0]

    model_name = apply_model_reference(args.model)
    run_name = args.run_name or f"repeated_single_shot{max_attempts}_{dataset_name}_{model_name.split('/')[-1]}"

    dataset_card = build_dataset_card(
        train,
        agent_val,
        target_col,
        metric_name,
        max_chars=4500,
    )
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        target_col=target_col,
        metric_name=metric_name,
        dataset_card=dataset_card,
    )
    if validation_labels_host_only:
        task_prompt += (
            "\nFeedback-validation labels are host-only and are not present in "
            "val_df. Each host validation access is counted."
        )

    execution_backend = args.executor_backend or os.getenv(
        "AUTOVIBE_EXECUTOR_BACKEND", "docker"
    )
    prediction_backend = prediction_backend_for_execution(execution_backend)
    recorder = start_research_run(
        args,
        arm="repeated_single_shot",
        dataset_id=dataset_name,
        dataset_hash=dataset_source_hash(dataset=args.dataset, dataset_dir=args.dataset_dir),
        split_seed=splits.metadata.seed,
        default_split_id=splits.metadata.split_strategy or f"seed-{args.seed}",
        model_id=model_name,
        prompt_template={
            "system": system_prompt,
            "task": TASK_PROMPT_TEMPLATE,
            "attempt_feedback": ATTEMPT_FEEDBACK_TEMPLATE,
        },
        decoding_config={
            "max_tokens": max_tokens,
            "temperature": configured_temperature(),
        },
        budget_policy=budget_policy_payload(
            episode_budget,
            fallback={
                "logical_llm_calls": max_attempts,
                "max_tokens_per_call": max_tokens,
            },
            stopping_policy=stopping_policy,
        ),
        execution_policy={
            "backend": execution_backend,
            "timeout_seconds": sandbox_timeout,
            "candidate_prediction_backend": prediction_backend,
            "candidate_prediction_network": (
                "none" if prediction_backend == "docker" else "host"
            ),
        },
        episode_budget=episode_budget,
    )
    client = wrap_llm(
        make_llm_client(),
        recorder,
        model=model_name,
        episode_budget=episode_budget,
    )
    executor = wrap_executor(
        CodeExecutor(
            timeout=sandbox_timeout,
            backend=execution_backend,
            docker_image=args.sandbox_image,
        ),
        recorder,
        episode_budget=episode_budget,
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
            "executor_backend": execution_backend,
            "dataset_split_strategy": splits.metadata.split_strategy,
            "dataset_role": splits.metadata.role,
            "dataset_sampled": str(splits.metadata.sampled),
            **mode_metadata_params(args, "repeated_single_shot"),
            **research_mlflow_params(recorder),
        })

        best_val: float | None = None
        best_model = None
        best_code = ""
        best_stdout = ""
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        errors_count = 0
        attempt_log = []
        attempt_records: list[dict] = []
        best_attempt_idx = -1
        budget_stop_reason = None
        stopping_controller = (
            StoppingController(stopping_policy)
            if stopping_policy is not None
            else None
        )
        hidden_gate = HiddenEvaluationGate(prediction_backend=prediction_backend)
        symmetric_terminal = None
        if episode_budget is not None and episode_budget.has_finalization_reserve:
            if (
                args.research_v2_metric_direction is None
                or args.research_v2_score_tolerance is None
            ):
                raise ValueError(
                    "An active V2 finalization reserve requires explicit "
                    "--research-v2-metric-direction and "
                    "--research-v2-score-tolerance values."
                )
            terminal_root = (
                recorder.run_dir / "private_terminal"
                if recorder is not None
                else Path(tempfile.mkdtemp(prefix="autovibe_rss_terminal_"))
            )
            metric_normalizer = (
                episode_budget.normalize_feedback_metric
                if episode_budget.has_validation_query_policy
                else float
            )
            symmetric_terminal = RepeatedSingleShotTerminalAdapter(
                private_dir=terminal_root,
                validation_features=val.drop(columns=[target_col]),
                validation_target=val[target_col],
                hidden_features=test.drop(columns=[target_col]),
                hidden_target=test[target_col],
                metric_fn=metric_fn,
                metric_name=metric_name,
                metric_direction=args.research_v2_metric_direction,
                score_tolerance=args.research_v2_score_tolerance,
                prediction_backend=prediction_backend,
                metric_normalizer=metric_normalizer,
                hidden_gate=hidden_gate,
            )

        for attempt in range(max_attempts):
            if stopping_controller is not None:
                _observe_repeated_boundary(
                    stopping_controller,
                    episode_budget,
                    symmetric_terminal,
                )
                if stopping_controller.decision is not None:
                    break
            prompt = _build_attempt_prompt(task_prompt, best_val, attempt)
            try:
                response = client.complete(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
            except BudgetExhausted as exc:
                budget_stop_reason = exc.reason
                if stopping_controller is not None:
                    incumbent = symmetric_terminal.registry.incumbent()
                    stopping_controller.observe_exploration_exhausted(
                        detail_code=exc.reason,
                        incumbent_candidate_id=(
                            incumbent.candidate_id if incumbent is not None else None
                        ),
                    )
                break
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_reasoning_tokens += response.reasoning_tokens

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
                "val_df": agent_val.copy(),
                "target_col": target_col,
                "pd": pd,
                "np": np,
            }
            try:
                stdout, stderr, namespace = executor.run(code, namespace)
            except BudgetExhausted as exc:
                budget_stop_reason = exc.reason
                if stopping_controller is not None:
                    incumbent = symmetric_terminal.registry.incumbent()
                    stopping_controller.observe_exploration_exhausted(
                        detail_code=exc.reason,
                        incumbent_candidate_id=(
                            incumbent.candidate_id if incumbent is not None else None
                        ),
                    )
                break
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
                validation = None
                try:
                    if symmetric_terminal is not None:
                        previous_incumbent = symmetric_terminal.registry.incumbent()
                        record = symmetric_terminal.validate_and_register(
                            model=model_obj,
                            source_code=code,
                            attempt_index=attempt,
                            before_validation=(
                                lambda: episode_budget.consume_validation_query(
                                    "repeated_single_shot_attempt"
                                )
                                if episode_budget.has_validation_query_policy
                                else None
                            ),
                            resource_snapshot={
                                "episode_budget": episode_budget.snapshot(),
                            },
                        )
                        val_metric = record.validation_metric
                        raw_validation_ready = True
                        incumbent = symmetric_terminal.registry.incumbent()
                        if stopping_controller is not None:
                            _observe_repeated_candidate(
                                stopping_controller,
                                previous_incumbent=previous_incumbent,
                                record=record,
                                terminal=symmetric_terminal,
                            )
                        if (
                            incumbent is not None
                            and incumbent.candidate_id == record.candidate_id
                        ):
                            best_val = val_metric
                            best_model = model_obj
                            best_code = code
                            best_stdout = stdout
                            best_attempt_idx = attempt
                    elif episode_budget is not None:
                        validation = validate_submission_candidate(
                            model_obj,
                            X_val,
                            prediction_backend=prediction_backend,
                        )
                        if not validation.valid:
                            raise RuntimeError(
                                f"{validation.error_type or 'SubmissionValidationError'}: "
                                f"{validation.error_message or 'candidate is not submit-ready'}"
                            )
                        val_preds = validation.predictions
                    else:
                        try:
                            val_preds = model_obj.predict(X_val)
                        except Exception:
                            # Preserve the product runner's existing autofit compatibility path.
                            model_obj.fit(train.drop(columns=[target_col]), train[target_col])
                        val_preds = model_obj.predict(X_val)
                    if symmetric_terminal is None:
                        raw_validation_ready = True
                        val_metric = score_with_coercion(metric_fn, y_val, val_preds)
                        if best_val is None or val_metric > best_val:
                            best_val = val_metric
                            best_model = model_obj
                            best_code = code
                            best_stdout = stdout
                            best_attempt_idx = attempt
                except Exception as exc:
                    if (
                        stopping_controller is not None
                        and isinstance(exc, CandidateBundleError)
                    ):
                        incumbent = symmetric_terminal.registry.incumbent()
                        stopping_controller.observe_unrecoverable_failure(
                            detail_code="candidate_bundle_failure",
                            incumbent_candidate_id=(
                                incumbent.candidate_id
                                if incumbent is not None
                                else None
                            ),
                        )
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
            # Extract a short exception label for the notebook error output.
            error_name = None
            if attempt_error:
                for line in reversed((stderr or "").strip().splitlines()):
                    line = line.strip()
                    if line and ":" in line and line.split(":", 1)[0].isidentifier():
                        error_name = line.split(":", 1)[0]
                        break
                error_name = error_name or (preflight_error.split(":", 1)[0] if preflight_error else "Error")
            attempt_records.append({
                "attempt": attempt + 1,
                "code": code,
                "stdout": stdout,
                "stderr": (attempt_error or stderr or ""),
                "error_name": error_name,
                "val_metric": val_metric,
            })
            mlflow.log_text(code, f"attempt_{attempt + 1:02d}_solution.py")
            mlflow.log_text(stdout, f"attempt_{attempt + 1:02d}_stdout.txt")
            mlflow.log_text(stderr, f"attempt_{attempt + 1:02d}_stderr.txt")
            if val_metric is not None:
                mlflow.log_metric("val_metric", val_metric, step=attempt)
            if stopping_controller is not None:
                _observe_repeated_boundary(
                    stopping_controller,
                    episode_budget,
                    symmetric_terminal,
                )
                if stopping_controller.decision is not None:
                    break

        if stopping_controller is not None and stopping_controller.decision is None:
            incumbent = symmetric_terminal.registry.incumbent()
            stopping_controller.observe_exploration_exhausted(
                detail_code="max_attempts",
                incumbent_candidate_id=(
                    incumbent.candidate_id if incumbent is not None else None
                ),
            )

        test_metric = None
        final_status = "no_candidate_found"
        null_reason = "No raw-validation-ready model was produced."
        submit_failure_type = "no_candidate_found"
        finalize_path = "failed"
        stopping_terminated = bool(
            stopping_controller is not None
            and stopping_controller.decision is not None
            and stopping_controller.decision.transition == "terminate"
        )
        if stopping_terminated:
            final_status = "stopped_unrecoverable_contract_failure"
            null_reason = (
                "The frozen stopping policy terminated the episode after an "
                "unrecoverable safety or contract failure."
            )
            submit_failure_type = stopping_controller.decision.detail_code
            finalize_path = "stopping_policy_terminate"
        elif symmetric_terminal is not None:
            terminal_result = None
            try:
                episode_budget.begin_finalization(
                    trigger=(
                        f"stopping:{stopping_controller.decision.reason}"
                        if stopping_controller is not None
                        and stopping_controller.decision is not None
                        else "repeated_single_shot_attempt_loop_complete"
                    )
                )

                def terminal_preflight_budget() -> None:
                    episode_budget.consume_tool_call("protected_replay_preflight")
                    if episode_budget.has_validation_query_policy:
                        episode_budget.consume_validation_query(
                            "protected_finalization"
                        )

                terminal_result = symmetric_terminal.finalize(
                    execute_code=executor.run,
                    namespace_factory=lambda: {
                        "train_df": train.copy(),
                        "val_df": (
                            val.drop(columns=[target_col]).copy()
                            if episode_budget.has_validation_query_policy
                            else val.copy()
                        ),
                        "target_col": target_col,
                        "pd": pd,
                        "np": np,
                    },
                    before_preflight=terminal_preflight_budget,
                    before_artifact_write=lambda: episode_budget.consume_tool_call(
                        "protected_artifact_write"
                    ),
                    before_hidden_evaluation=lambda: episode_budget.consume_tool_call(
                        "protected_hidden_evaluation"
                    ),
                    resource_snapshot=lambda: {
                        "terminal_contract_version": TERMINAL_CONTRACT_VERSION,
                        "episode_budget": episode_budget.snapshot(),
                    },
                )
            except BudgetExhausted as exc:
                final_status = "finalization_reserve_exhausted"
                null_reason = _symmetric_terminal_null_reason(final_status)
                submit_failure_type = type(exc).__name__
                finalize_path = "protected_incumbent_replay"
                errors_count += 1
            except Exception as exc:
                final_status = "protected_finalization_start_failed"
                null_reason = "The protected finalization phase could not start safely."
                submit_failure_type = type(exc).__name__
                finalize_path = "protected_incumbent_replay"
                errors_count += 1
            if terminal_result is not None:
                final_status = terminal_result.final_status
                finalize_path = "protected_incumbent_replay"
                if terminal_result.succeeded:
                    test_metric = terminal_result.hidden_metric
                    null_reason = None
                    submit_failure_type = None
                else:
                    null_reason = _symmetric_terminal_null_reason(
                        terminal_result.final_status
                    )
                    submit_failure_type = (
                        terminal_result.error_type or terminal_result.final_status
                    )
                    errors_count += 1
        elif best_model is not None:
            try:
                if episode_budget is not None:
                    validation = validate_submission_candidate(
                        best_model,
                        val.drop(columns=[target_col]),
                        prediction_backend=prediction_backend,
                    )
                    if not validation.valid:
                        raise RuntimeError(
                            f"{validation.error_type or 'SubmissionValidationError'}: "
                            f"{validation.error_message or 'candidate is not submit-ready'}"
                        )
                else:
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
                    if episode_budget is not None:
                        test_metric = hidden_gate.evaluate(best_model, X_test, y_test, metric_fn)
                    else:
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
        elif budget_stop_reason is not None:
            final_status = "budget_exhausted"
            null_reason = f"Episode stopped before a valid candidate: {budget_stop_reason}."
            submit_failure_type = "budget_exhausted"
            finalize_path = "budget_stop"

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
            "reasoning_tokens": total_reasoning_tokens,
            "elapsed_seconds": elapsed,
        }
        from experiments.dashboard_artifacts import checklist_coverage, write_attempts_episode
        coverage = checklist_coverage(best_code, best_stdout, target_col) if best_code else None
        if coverage is not None:
            metrics["checklist_coverage"] = coverage
        metrics["steps_used"] = len(attempt_log)
        # Always emit an episode — even a fully-failed run — so EVERY attempt
        # (code + error) is visible in the dashboard, not just the best one.
        if args.workspace_dir and attempt_records:
            for rec in attempt_records:
                rec["is_best"] = rec["attempt"] == best_attempt_idx + 1
            write_attempts_episode(args.workspace_dir, attempts=attempt_records,
                                   target_col=target_col, coverage=coverage,
                                   metric_name=metric_name)
        # Best-effort self-summary of the best attempt (validation metric is not
        # hidden), persisted as run_summary.json for the dashboard «Мысли» tab.
        # Generated once the model produced a usable solution (best_code is set
        # only for a raw-validation-ready model), even if the hidden test failed.
        if args.workspace_dir and best_code and episode_budget is None:
            from gym.run_summary import generate_and_write

            convo = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": f"```python\n{best_code}\n```"},
            ]
            if best_val is not None:
                convo.append({
                    "role": "user",
                    "content": f"Лучшая попытка дала {metric_name} на валидации = {best_val:.4f}.",
                })
            generate_and_write(
                client,
                model_name,
                args.workspace_dir,
                conversation=convo,
                solution_code=best_code,
                max_tokens=min(max_tokens, 700),
            )
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
        "reasoning_tokens": total_reasoning_tokens,
        "hidden_evaluations": int(hidden_gate.attempted),
        "budget_stop_reason": budget_stop_reason,
        "elapsed_seconds": elapsed,
    }
    if stopping_controller is not None:
        summary.update(
            {
                "stopping_policy": stopping_policy.to_dict(),
                "stopping_policy_state": stopping_controller.snapshot(),
                "stopping_events": list(stopping_controller.events),
                "stopping_reason": (
                    stopping_controller.decision.reason
                    if stopping_controller.decision is not None
                    else None
                ),
                "stopping_transition": (
                    stopping_controller.decision.transition
                    if stopping_controller.decision is not None
                    else None
                ),
            }
        )
    finalize_research_run(recorder, summary, episode_budget=episode_budget)
    print("\n=== Repeated Single-Shot Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
