from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cloudpickle
import numpy as np
import pandas as pd

from .candidates import CandidateRecord, CandidateRegistry
from .data_profile import (
    build_compact_profile,
    build_dataset_card,
    format_profile_for_agent,
    format_ydata_profile_for_agent,
    profile_config_from_env,
    run_ydata_profile,
)
from .feedback import FeedbackItem, FeedbackPolicy, NotebookChecklist
from .jupyter_kernel import (
    CellExecutionResult,
    KernelExecutionBackend,
    LocalJupyterKernelBackend,
)
from .modes import EpisodeMode, resolve_episode_mode
from .notebook import NotebookDocument
from .protocol import AGENT_STAGE_VALUES, Action, ActionParseError, Observation, coerce_action


def _default_kernel_backend() -> KernelExecutionBackend:
    mode = os.getenv("AUTOVIBE_KERNEL_BACKEND", "local").lower()
    if mode == "docker":
        from .jupyter_kernel import ContainerJupyterKernelBackend
        return ContainerJupyterKernelBackend()
    return LocalJupyterKernelBackend()


from .scoring import score_with_coercion as _score_with_coercion


MODEL_INTERFACE_MESSAGE = (
    "[MODEL CHECK] The selected candidate is not submit-ready because it does "
    "not provide a predict(X) method."
)

MODEL_RAW_INPUT_MESSAGE = (
    "[MODEL CHECK] The saved candidate cannot predict raw validation features. "
    "The submitted artifact must reproduce all required preprocessing when "
    "called on new raw rows."
)

MODEL_SERIALIZATION_MESSAGE = (
    "[MODEL CHECK] Candidate variable '{model_var}' cannot be serialized. Avoid "
    "local lambdas or non-serializable preprocessing objects; use a reproducible "
    "sklearn Pipeline/ColumnTransformer or top-level transformer functions."
)

NEEDS_CLEAN_RUN_MESSAGE = (
    "The current notebook has not been confirmed by a full clean run. "
    "Run restart_and_run_all first."
)

NEEDS_VALIDATION_MESSAGE = (
    "Submit requires a successful environment-controlled validate action for "
    "the current clean notebook revision."
)

HIDDEN_TEST_FAILURE_MESSAGE = (
    "Final submit failed on the hidden test split. The failure is recorded "
    "privately; hidden rows, labels, and score are not exposed."
)


@dataclass
class NotebookEnvState:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col: str
    metric_name: str
    step: int = 0
    max_steps: int = 20
    submitted: bool = False
    history: list[Observation] = field(default_factory=list)


@dataclass
class CandidateDiagnostic:
    step: int
    candidate_var: str
    source: str
    exists: bool | None = None
    has_predict: bool | None = None
    raw_val_predict_ok: bool | None = None
    prediction_length_ok: bool | None = None
    prediction_nan_free: bool | None = None
    serializable: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    validation_metric: float | None = None
    sample_rows: int | None = None
    model: Any = field(default=None, repr=False, compare=False)
    predictions: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "candidate_var": self.candidate_var,
            "source": self.source,
            "exists": self.exists,
            "has_predict": self.has_predict,
            "raw_val_predict_ok": self.raw_val_predict_ok,
            "prediction_length_ok": self.prediction_length_ok,
            "prediction_nan_free": self.prediction_nan_free,
            "serializable": self.serializable,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "validation_metric": self.validation_metric,
            "sample_rows": self.sample_rows,
        }


class NotebookGymEnv:
    """Real Jupyter-backed AutoVibe Gym environment."""

    protocol_version = "jupyter-v1"
    checklist_version = "generic-hidden-v1"
    feedback_policy_version = "selective-v1"

    _CANDIDATE_VAR_NAMES = (
        "model",
        "best_model",
        "final_model",
        "pipeline",
        "best_pipeline",
        "best_estimator",
        "estimator",
        "clf",
        "classifier",
        "best_clf",
        "best_classifier",
        "rf_model",
        "lgb_model",
        "xgb_model",
        "catboost_model",
        "trained_model",
        "final_pipeline_model",
        "reg",
        "regressor",
    )

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        metric_fn: Callable,
        metric_name: str = "score",
        max_steps: int = 20,
        workspace_dir: str | Path | None = None,
        private_dir: str | Path | None = None,
        mode: str | EpisodeMode | None = None,
        backend: KernelExecutionBackend | None = None,
        kernel_timeout: int = 60,
        enable_thoughts: bool = False,
        hint_cooldown: int = 2,
    ):
        self.enable_thoughts = enable_thoughts
        # Steps to wait between consecutive checklist hints (gym mode).
        self.hint_cooldown = max(1, int(hint_cooldown))
        self._feedback_policy = FeedbackPolicy(hint_cooldown_executions=self.hint_cooldown)
        self.scratchpad: list[dict[str, Any]] = []
        self.state = NotebookEnvState(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
            target_col=target_col,
            metric_name=metric_name,
            max_steps=max_steps,
        )
        self.metric_fn = metric_fn
        self.mode = resolve_episode_mode(mode)
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path(
            tempfile.mkdtemp(prefix="autovibe_episode_")
        ).resolve()
        self.private_dir = Path(private_dir).resolve() if private_dir else Path(
            tempfile.mkdtemp(prefix="autovibe_private_")
        ).resolve()
        self.backend = backend or _default_kernel_backend()
        self.kernel_timeout = kernel_timeout
        self.kernel = self.backend.create_session(self.workspace_dir)
        self.notebook = NotebookDocument.create(self.workspace_dir / "solution.ipynb")
        self.checklist = NotebookChecklist(target_col=target_col, policy=self._feedback_policy)
        self.candidates = CandidateRegistry()
        self._candidate_objects: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.feedback_trace: list[dict[str, Any]] = []
        self.validation_trajectory: list[dict[str, Any]] = []
        self.candidate_diagnostics: list[dict[str, Any]] = []
        self.private_summary: dict[str, Any] = {}
        self.dirty_since_clean_run = True
        self.last_clean_run_id: str | None = None
        self.notebook_revision_at_clean_run: int | None = None
        self.last_validated_candidate_id: str | None = None
        self.cell_executions_total = 0
        self.kernel_restarts_total = 0
        self.clean_runs_total = 0
        self.validation_calls_total = 0
        self.tool_calls_total = 0
        self.contract_feedback_count = 0
        self.model_check_failure_count = 0
        self.errors_count = 0
        self._model_check_seen: set[tuple[str, str | None, str]] = set()
        self._agent_trace_turn = 0
        self._start_time = time.time()
        self.current_stage: str | None = None
        self._accepted_agent_actions = 0
        self._active_action: Action | None = None

    def reset(self) -> dict:
        self.close()
        self._clear_generated_artifacts()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.workspace_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.state.train.to_csv(data_dir / "train.csv", index=False)
        self.state.val.to_csv(data_dir / "val.csv", index=False)

        self.notebook = NotebookDocument.create(self.workspace_dir / "solution.ipynb")
        self.kernel = self.backend.create_session(self.workspace_dir)
        self.kernel.start()
        self.kernel.inject_bootstrap_context(
            train_csv=data_dir / "train.csv",
            val_csv=data_dir / "val.csv",
            target_col=self.state.target_col,
        )

        self.state.step = 0
        self.state.submitted = False
        self.state.history = []
        self.checklist = NotebookChecklist(target_col=self.state.target_col, policy=self._feedback_policy)
        self.candidates = CandidateRegistry()
        self._candidate_objects = {}
        self.events = []
        self.feedback_trace = []
        self.validation_trajectory = []
        self.candidate_diagnostics = []
        self.scratchpad = []
        self.private_summary = {}
        self.dirty_since_clean_run = True
        self.last_clean_run_id = None
        self.notebook_revision_at_clean_run = None
        self.last_validated_candidate_id = None
        self.cell_executions_total = 0
        self.kernel_restarts_total = 0
        self.clean_runs_total = 0
        self.validation_calls_total = 0
        self.tool_calls_total = 0
        self.contract_feedback_count = 0
        self.model_check_failure_count = 0
        self.errors_count = 0
        self._model_check_seen = set()
        self._agent_trace_turn = 0
        self._start_time = time.time()
        self.current_stage = None
        self._accepted_agent_actions = 0
        self._active_action = None
        self._save_artifacts()
        return self._build_context_prompt()

    def close(self) -> None:
        kernel = getattr(self, "kernel", None)
        if kernel is not None:
            kernel.shutdown()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def step(self, action: Action | dict[str, Any] | str) -> Observation:
        if self.state.submitted:
            raise RuntimeError("Environment already finalized via submit().")
        try:
            parsed = coerce_action(action)
        except ActionParseError as exc:
            return self._reject_action_contract(
                "invalid_action",
                str(exc),
                action_type="invalid_action",
            )

        contract_error = self._validate_action_contract(parsed)
        if contract_error is not None:
            key, message = contract_error
            return self._reject_action_contract(
                key,
                message,
                action_type=parsed.type,
                stage=parsed.stage or None,
                thoughts=parsed.thoughts or None,
                model_var=parsed.model_var if parsed.type == "validate" else None,
                cell_id=parsed.cell_id,
            )

        self._accept_action(parsed)
        if self.state.step >= self.state.max_steps and parsed.type not in {"submit", "finalize"}:
            observation = self._observation(
                action=parsed.type,
                feedback_items=[
                    self._contract_feedback(
                        "step_budget_exhausted",
                        "Step budget exhausted. Submit an already validated candidate.",
                        severity="blocker",
                    )
                ],
                done=True,
                model_var=parsed.model_var if parsed.type == "validate" else None,
                cell_id=parsed.cell_id,
            )
            self._record_event(action=parsed.type, blocked="step_budget_exhausted")
            return self._record_observation(observation)
        self._consume_step(parsed)

        if parsed.type == "code":
            parsed = Action.add_cell_action(
                parsed.code,
                cell_type="code",
                execute=True,
                stage=parsed.stage,
                thoughts=parsed.thoughts,
            )
        if parsed.type == "think":
            return self._think(parsed)
        if parsed.type == "add_cell":
            return self._add_cell(parsed)
        if parsed.type == "update_cell":
            return self._update_cell(parsed)
        if parsed.type == "delete_cell":
            return self._delete_cell(parsed)
        if parsed.type == "move_cell":
            return self._move_cell(parsed)
        if parsed.type == "run_cell":
            return self._run_cell(parsed.cell_id or "")
        if parsed.type == "inspect_notebook":
            return self._inspect_notebook()
        if parsed.type == "restart_and_run_all":
            return self.restart_and_run_all()
        if parsed.type == "validate":
            return self.validate_candidate(parsed.model_var)
        if parsed.type == "submit":
            return self.submit_by_name(parsed.model_var)
        if parsed.type == "inspect_data":
            return self.inspect_data()
        if parsed.type == "profile_data":
            return self.profile_data(parsed.profile)
        if parsed.type == "list_candidates":
            return self.list_candidates()
        if parsed.type == "check_candidate":
            return self.check_candidate(parsed.model_var)
        if parsed.type == "quick_validate":
            return self.quick_validate(parsed.model_var)
        if parsed.type == "cleanlab_diagnose":
            return self.cleanlab_diagnose(
                parsed.model_var,
                source=parsed.source,
                max_issues=parsed.max_issues,
            )
        if parsed.type == "tune_hyperparameters":
            return self.tune_hyperparameters(
                parsed.model_var,
                search_space=parsed.search_space,
                n_trials=parsed.n_trials,
                timeout_sec=parsed.timeout_sec,
                scoring=parsed.scoring,
            )
        if parsed.type == "finalize":
            finalized = self.finalize(parsed.model_var)
            if finalized is not None:
                return finalized
            return self._finalize_failure_observation(
                final_status="no_candidate_found",
                null_reason="Environment was already submitted or no finalization action was possible.",
                finalize_path="failed",
                attempted_vars=[],
            )
        raise RuntimeError(f"Unsupported action type: {parsed.type}")

    def budget_remaining(self) -> int:
        return max(self.state.max_steps - self.state.step, 0)

    def _consume_step(self, action: Action) -> None:
        tool_actions = {
            "think",
            "submit",
            "inspect_data",
            "profile_data",
            "check_candidate",
            "quick_validate",
            "list_candidates",
            "cleanlab_diagnose",
            "tune_hyperparameters",
            "finalize",
        }
        if action.type in tool_actions:
            if action.type not in {"submit", "think"}:
                self.tool_calls_total += 1
            return
        else:
            self.state.step += 1

    def _validate_action_contract(self, action: Action) -> tuple[str, str] | None:
        if not action.stage:
            return (
                "missing_stage",
                "Every JSON action must include a non-empty 'stage' from the allowed stage enum.",
            )
        if action.stage not in AGENT_STAGE_VALUES:
            return (
                "unknown_stage",
                f"Unknown stage '{action.stage}'. Use one of the allowed stage values.",
            )
        has_thoughts = bool((action.thoughts or "").strip())
        if not self.enable_thoughts:
            if action.type == "think":
                return (
                    "think_disabled",
                    "Thoughts mode is disabled, so type 'think' is not allowed.",
                )
            if action.stage == "planning":
                return (
                    "planning_disabled",
                    "Thoughts mode is disabled, so stage 'planning' is not allowed.",
                )
            if has_thoughts:
                return (
                    "thoughts_disabled",
                    "Thoughts mode is disabled, so do not include 'thoughts'.",
                )
            return None

        if not has_thoughts:
            return (
                "missing_thoughts",
                "Thoughts mode is enabled, so every JSON action must include non-empty 'thoughts'.",
            )
        if self._accepted_agent_actions == 0:
            if action.type != "think" or action.stage != "planning":
                return (
                    "initial_planning_required",
                    "Thoughts mode is enabled, so the first action must be type 'think' with stage 'planning'.",
                )
            return None
        if action.stage == "planning":
            return (
                "planning_repeated",
                "Stage 'planning' is only allowed for the first thoughts-enabled think action.",
            )
        return None

    def _accept_action(self, action: Action) -> None:
        self.current_stage = action.stage
        self._active_action = action
        self._accepted_agent_actions += 1
        if self.enable_thoughts and action.thoughts:
            self._add_thought(action)

    def _reject_action_contract(
        self,
        key: str,
        message: str,
        *,
        action_type: str,
        stage: str | None = None,
        thoughts: str | None = None,
        model_var: str | None = None,
        cell_id: str | None = None,
    ) -> Observation:
        feedback = self._contract_feedback(key, message, severity="blocker", cell_id=cell_id)
        self._record_event(
            action=action_type,
            stage=stage,
            thoughts=thoughts,
            blocked=key,
        )
        observation = self._observation(
            action=action_type,
            feedback_items=[feedback],
            done=False,
            model_var=model_var,
            cell_id=cell_id,
            stage=stage,
            thoughts=thoughts,
        )
        return self._record_observation(observation)

    def restart_and_run_all(self) -> Observation:
        self.kernel.restart()
        self.kernel_restarts_total += 1
        self.kernel.inject_bootstrap_context(
            train_csv=self.workspace_dir / "data" / "train.csv",
            val_csv=self.workspace_dir / "data" / "val.csv",
            target_col=self.state.target_col,
        )
        self.notebook.clear_outputs()
        clean_run_id = str(uuid.uuid4())
        self.clean_runs_total += 1

        last_result: CellExecutionResult | None = None
        failed_cell_id: str | None = None
        for cell in self.notebook.notebook.cells:
            if cell.cell_type != "code":
                continue
            cell_id = str(cell.get("id"))
            result = self.kernel.execute_cell(
                str(cell.source),
                timeout=self.kernel_timeout,
            )
            self.cell_executions_total += 1
            self.notebook.set_cell_outputs(
                cell_id,
                outputs=result.outputs,
                execution_count=result.execution_count,
            )
            self._record_event(
                action="clean_run_cell",
                cell_id=cell_id,
                execution_result=result.to_dict(),
                clean_run_id=clean_run_id,
            )
            self._record_cell_execution(cell_id=cell_id, source=str(cell.source), result=result)
            last_result = result
            if not result.success:
                failed_cell_id = cell_id
                break

        if failed_cell_id is not None:
            self.dirty_since_clean_run = True
            self.last_clean_run_id = None
            self.notebook_revision_at_clean_run = None
            self.last_validated_candidate_id = None
            feedback = [
                self._contract_feedback(
                    "clean_run_failed",
                    f"Clean run failed at {failed_cell_id}. Fix the notebook and run it again.",
                    cell_id=failed_cell_id,
                )
            ]
            observation = self._observation(
                action="restart_and_run_all",
                stdout=last_result.compact_text() if last_result else "",
                feedback_items=feedback,
                cell_id=failed_cell_id,
            )
            self.errors_count += 1
            return self._record_observation(observation)

        self.dirty_since_clean_run = False
        self.last_clean_run_id = clean_run_id
        self.notebook_revision_at_clean_run = self.notebook.revision
        self.last_validated_candidate_id = None
        self.checklist.record_structural(
            "reproducible_solution",
            reason="successful restart_and_run_all",
            step=self.state.step,
        )
        self._record_event(action="restart_and_run_all", clean_run_id=clean_run_id)
        observation = self._observation(
            action="restart_and_run_all",
            stdout="[CLEAN RUN] Notebook executed successfully from a fresh kernel.",
        )
        return self._record_observation(observation)

    def validate_candidate(
        self, model_var: str = "model", *, allow_dirty: bool = False
    ) -> Observation:
        self.validation_calls_total += 1
        if model_var == "auto":
            return self._validate_auto_candidate(allow_dirty=allow_dirty)

        blocker = None if allow_dirty else self._clean_state_blocker()
        if blocker is not None:
            observation = self._observation(
                action="validate",
                feedback_items=[blocker],
                model_var=model_var,
            )
            return self._record_observation(observation)

        diagnostic = self._check_candidate_readiness(
            model_var,
            source="validate",
            sample_rows=None,
            compute_metric=True,
        )
        if not diagnostic.exists:
            feedback = self._contract_feedback(
                "candidate_missing",
                f"Candidate variable '{model_var}' is not available after the clean run.",
                severity="blocker",
            )
            self._record_event(
                action="validate",
                model_var=model_var,
                error=f"{diagnostic.error_type or 'NameError'}: {diagnostic.error_message or 'candidate missing'}",
            )
            observation = self._observation(
                action="validate",
                feedback_items=[feedback],
                model_var=model_var,
            )
            return self._record_observation(observation)

        # Candidate-readiness failures are counted centrally in
        # _record_candidate_diagnostic().
        if not diagnostic.has_predict:
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._contract_feedback(
                        "model_interface",
                        MODEL_INTERFACE_MESSAGE,
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        if not diagnostic.raw_val_predict_ok:
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._candidate_failure_feedback(diagnostic, source="validate")
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        if not diagnostic.prediction_length_ok or not diagnostic.prediction_nan_free:
            detail = "prediction length mismatch"
            if diagnostic.prediction_nan_free is False:
                detail = "predictions contain NaN-like values"
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._contract_feedback(
                        "raw_validation_prediction_invalid",
                        (
                            f"[MODEL CHECK] Candidate variable '{model_var}' is not submit-ready: {detail}. "
                            "The final candidate must return one non-null prediction per raw validation row."
                        ),
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        if not diagnostic.serializable:
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._contract_feedback(
                        "candidate_serialization_failed",
                        MODEL_SERIALIZATION_MESSAGE.format(model_var=model_var),
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        if diagnostic.validation_metric is None:
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._contract_feedback(
                        "validation_scoring_failed",
                        (
                            f"[MODEL CHECK] Candidate variable '{model_var}' produced predictions "
                            "on raw validation rows, but the validation metric could not be computed: "
                            f"{diagnostic.error_type or 'MetricError'}: "
                            f"{_clip(diagnostic.error_message or 'metric computation failed', 280)}.\n"
                            "Make sure predict() returns labels compatible with the target and metric."
                        ),
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        record = self._register_validated_candidate(diagnostic, model_var)
        self._record_event(action="validate", model_var=model_var, candidate=record.to_dict())
        observation = self._observation(
            action="validate",
            stdout=f"validation_{self.state.metric_name}={diagnostic.validation_metric:.6f}",
            model_var=model_var,
            validation_metric=diagnostic.validation_metric,
        )
        return self._record_observation(observation)

    def submit_by_name(
        self, model_var: str = "model", *, allow_dirty: bool = False
    ) -> Observation:
        if model_var == "auto":
            return self._submit_auto_candidate(allow_dirty=allow_dirty)

        blocker = None if allow_dirty else self._clean_state_blocker()
        if blocker is not None:
            observation = self._observation(
                action="submit",
                feedback_items=[blocker],
                model_var=model_var,
            )
            return self._record_observation(observation)

        candidate = self.candidates.latest()
        mismatch = (
            candidate is None
            or not candidate.validation_success
            or candidate.model_var != model_var
        )
        if not allow_dirty:
            mismatch = mismatch or (
                candidate.notebook_revision != self.notebook.revision
                or candidate.clean_run_id != self.last_clean_run_id
            )
        if mismatch:
            observation = self._observation(
                action="submit",
                feedback_items=[
                    self._contract_feedback(
                        "validation_required",
                        NEEDS_VALIDATION_MESSAGE,
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        model = self._candidate_objects.get(candidate.candidate_id)
        if model is None and candidate.artifact_path:
            with open(candidate.artifact_path, "rb") as fh:
                model = cloudpickle.load(fh)

        try:
            X_test = self.state.test.drop(columns=[self.state.target_col])
            y_test = self.state.test[self.state.target_col]
            preds = model.predict(X_test)
            score = _score_with_coercion(self.metric_fn, y_test, preds)
            candidate.submitted = True
            self.state.submitted = True
            self.private_summary["final_test_metric"] = score
            self.private_summary["valid_submit"] = True
            self.private_summary["submit_failure_type"] = None
            self.private_summary.setdefault(
                "final_status",
                "submitted_dirty_finalize" if allow_dirty else "submitted_clean",
            )
            self.private_summary.setdefault("null_reason", None)
            self.private_summary.setdefault(
                "reproducibility_level",
                "raw_val_ready_only" if allow_dirty else "clean_replay",
            )
            self.private_summary.setdefault("finalize_path", "agent_submit")
            self._record_event(
                action="submit",
                model_var=model_var,
                candidate_id=candidate.candidate_id,
                private={"final_test_metric": score},
            )
            observation = self._observation(
                action="submit",
                done=True,
                submitted=True,
                model_var=model_var,
            )
            return self._record_observation(observation)
        except Exception as exc:
            self.state.submitted = True
            self.private_summary["valid_submit"] = False
            self.private_summary["final_test_metric"] = None
            self.private_summary["submit_failure_type"] = type(exc).__name__
            self.private_summary["final_status"] = "hidden_submit_failed"
            self.private_summary["null_reason"] = (
                "Candidate passed validation but failed during private hidden-test submission."
            )
            self.private_summary.setdefault("finalize_path", "agent_submit")
            self._record_event(
                action="submit_failed",
                model_var=model_var,
                candidate_id=candidate.candidate_id,
                private={"failure_type": type(exc).__name__},
            )
            observation = self._observation(
                action="submit",
                feedback_items=[
                    self._contract_feedback(
                        "hidden_submit_failed",
                        HIDDEN_TEST_FAILURE_MESSAGE,
                        severity="blocker",
                    )
                ],
                done=True,
                submitted=True,
                model_var=model_var,
            )
            return self._record_observation(observation)

    def finalize(self, model_var: str = "auto") -> Observation | None:
        """Best-effort host-controlled finalization when the agent stops without
        a valid submit (e.g. the step budget is exhausted).

        Implements the TZ fallback contract: if the agent never submitted, the
        environment selects a candidate itself. It submits an already-validated
        candidate for the current clean run, or otherwise performs a clean
        ``restart_and_run_all`` and validates the first predict-capable
        candidate variable found in the notebook before submitting it.

        Returns the terminal :class:`Observation`, or ``None`` when no
        reproducible candidate could be produced (e.g. the clean run fails).
        """
        if self.state.submitted:
            return None

        attempted_vars: list[str] = []
        self.private_summary["finalized_by_host"] = True
        self.private_summary["finalize_attempted_vars"] = attempted_vars

        # 1. Reuse an existing validated candidate for the current clean run.
        candidate = self.candidates.latest()
        if (
            candidate is not None
            and candidate.validation_success
            and not self.dirty_since_clean_run
            and candidate.clean_run_id == self.last_clean_run_id
            and candidate.notebook_revision == self.notebook.revision
            and (model_var == "auto" or candidate.model_var == model_var)
        ):
            self.private_summary["finalize_path"] = "validated_clean_candidate"
            self.private_summary["reproducibility_level"] = "clean_replay"
            return self.submit_by_name(candidate.model_var)

        # 2. Live-kernel fallback (tried before any restart, which would wipe the
        #    kernel). The agent often has a working fitted estimator in the live
        #    kernel even when the accumulated notebook cannot replay cleanly.
        #    The candidate is still tested on raw validation and raw test rows, so
        #    raw-input correctness is preserved; only the full-notebook
        #    reproducibility guarantee is relaxed to avoid a null outcome.
        live = self._finalize_from_candidate_vars(
            allow_dirty=True,
            model_var=model_var,
            attempted_vars=attempted_vars,
            finalize_path="live_kernel_candidate",
        )
        if live is not None:
            return live

        # 3. Last resort: a clean restart_and_run_all, then validate + submit.
        if self.dirty_since_clean_run or self.last_clean_run_id is None:
            self.restart_and_run_all()
            if self.dirty_since_clean_run:
                return self._finalize_failure_observation(
                    final_status="clean_run_failed",
                    null_reason="Clean restart_and_run_all failed during host finalization.",
                    finalize_path="failed",
                    attempted_vars=attempted_vars,
                )
        replay = self._finalize_from_candidate_vars(
            allow_dirty=False,
            model_var=model_var,
            attempted_vars=attempted_vars,
            finalize_path="clean_replay_candidate",
        )
        if replay is not None:
            return replay
        return self._finalize_failure_observation(
            final_status="no_candidate_found",
            null_reason="No discovered candidate could predict raw validation rows and serialize.",
            finalize_path="failed",
            attempted_vars=attempted_vars,
        )

    def _finalize_from_candidate_vars(
        self,
        *,
        allow_dirty: bool,
        model_var: str,
        attempted_vars: list[str],
        finalize_path: str,
    ) -> Observation | None:
        names = [model_var] if model_var != "auto" else self._candidate_var_order()
        for name in names:
            if name not in attempted_vars:
                attempted_vars.append(name)
            self.validate_candidate(name, allow_dirty=allow_dirty)
            latest = self.candidates.latest()
            if (
                latest is not None
                and latest.validation_success
                and latest.model_var == name
            ):
                self.private_summary["finalize_path"] = finalize_path
                self.private_summary["reproducibility_level"] = (
                    "raw_val_ready_only" if allow_dirty else "clean_replay"
                )
                return self.submit_by_name(name, allow_dirty=allow_dirty)
        return None

    def get_summary(self) -> dict:
        error_count = sum(1 for observation in self.state.history if observation.stderr.strip())
        final_test_metric = self.private_summary.get("final_test_metric")
        has_test_metric = final_test_metric is not None
        final_status = self.private_summary.get("final_status")
        if final_status is None:
            if has_test_metric:
                final_status = "submitted_clean"
            elif self.state.submitted:
                final_status = "hidden_submit_failed"
            else:
                final_status = "no_candidate_found"
        null_reason = self.private_summary.get("null_reason")
        if null_reason is None and not has_test_metric:
            null_reason = "No hidden-test metric was produced."
        finalize_path = self.private_summary.get("finalize_path")
        if finalize_path is None:
            finalize_path = "not_attempted" if not self.private_summary.get("finalized_by_host") else "failed"
        summary = {
            "steps_used": self.state.step,
            "thoughts_enabled": self.enable_thoughts,
            "thoughts_count": len(self.scratchpad),
            "current_stage": self.current_stage,
            "checklist_coverage": self.checklist.coverage(),
            "private_checklist_coverage": self.checklist.coverage(),
            "error_count": error_count,
            "errors_count": error_count,
            "submitted": self.state.submitted,
            "valid_submit": bool(self.private_summary.get("valid_submit", False)),
            "test_metric": final_test_metric,
            "final_test_metric": final_test_metric,
            "has_test_metric": has_test_metric,
            "submit_failed": not has_test_metric,
            "submit_failure_type": self.private_summary.get("submit_failure_type"),
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            "notebook_cells_final": len(self.notebook.notebook.cells),
            "notebook_revisions_total": self.notebook.revision,
            "cell_executions_total": self.cell_executions_total,
            "kernel_restarts_total": self.kernel_restarts_total,
            "clean_runs_total": self.clean_runs_total,
            "successful_clean_run": int(not self.dirty_since_clean_run and self.last_clean_run_id is not None),
            "validation_calls_total": self.validation_calls_total,
            "tool_calls_total": self.tool_calls_total,
            "best_validation_metric": self._best_validation_metric(),
            "contract_feedback_count": self.contract_feedback_count,
            "model_check_failure_count": self.model_check_failure_count,
            "candidate_diagnostics_total": len(self.candidate_diagnostics),
            "checklist_hints_shown_total": self.checklist.hints_shown_total,
            "process_guidance_hints_shown_total": self.checklist.process_guidance_hints_shown_total,
            "protocol_version": self.protocol_version,
            "notebook_backend": "jupyter",
            "checklist_version": self.checklist_version,
            "feedback_policy_version": self.feedback_policy_version,
            "episode_workspace": str(self.workspace_dir),
            "private_episode_dir": str(self.private_dir),
            "final_status": final_status,
            "null_reason": null_reason,
            "finalize_path": finalize_path,
        }
        summary.update(self.private_summary)
        summary["test_metric"] = final_test_metric
        summary["final_test_metric"] = final_test_metric
        summary["has_test_metric"] = has_test_metric
        summary["submit_failed"] = not has_test_metric
        summary["final_status"] = final_status
        summary["null_reason"] = null_reason
        summary["finalize_path"] = finalize_path
        return summary

    def _think(self, action: Action) -> Observation:
        self._record_event(
            action="think",
            non_mutating=True,
            thoughts=action.thoughts,
            stage=action.stage,
        )
        observation = self._observation(
            action="think",
            stdout="[THINK] Thought recorded. Notebook, kernel, data, validation, and submission state were not changed.",
            stage=action.stage,
            thoughts=action.thoughts,
        )
        return self._record_observation(observation)

    def _add_cell(self, action: Action) -> Observation:
        # Weak models sometimes emit add_cell with an empty 'source' (or put the
        # code under 'code'). Refuse to create blank code cells and nudge the
        # agent instead of silently piling up empty cells and wasting budget.
        if action.cell_type != "markdown" and not (action.source or "").strip():
            self._record_event(action="add_cell", blocked="empty_source")
            observation = self._observation(
                action="add_cell",
                feedback_items=[self._contract_feedback(
                    "empty_cell",
                    "Your add_cell had an empty 'source', so no cell was created. "
                    "Put your Python code in the 'source' field (a non-empty string) "
                    "and send the action again.",
                    severity="warning",
                )],
            )
            return self._record_observation(observation)
        # Pre-validate Python syntax before adding/executing code cells so weak
        # models get an immediate, actionable error instead of a kernel SyntaxError
        # after the cell has already been added.  Also catches the common failure
        # mode where a model puts explanatory prose into 'source' instead of code.
        if action.cell_type != "markdown":
            try:
                ast.parse((action.source or "").strip())
            except SyntaxError as _e:
                _is_prose = not any(
                    kw in (action.source or "")
                    for kw in ("import ", "def ", "class ", " = ", "(", "[",
                               "print(", "return ", "for ", "if ", "while ")
                )
                _hint = (
                    " Your 'source' looks like explanatory text, not Python code."
                    " Move explanations to a markdown cell (cell_type='markdown')"
                    " or include them in the 'notes' field — code cells must"
                    " contain executable Python only."
                    if _is_prose else ""
                )
                self._record_event(action="add_cell", blocked="syntax_error_pre_check")
                observation = self._observation(
                    action="add_cell",
                    feedback_items=[self._contract_feedback(
                        "syntax_error_pre_check",
                        f"SyntaxError: {_e} — cell was NOT added.{_hint}",
                        severity="blocker",
                    )],
                )
                return self._record_observation(observation)
        if action.cell_type == "markdown":
            cell_id = self.notebook.add_markdown_cell(action.source)
            result = None
        else:
            cell_id = self.notebook.add_code_cell(action.source)
            result = self._execute_and_store_cell(cell_id) if action.execute else None
        self._mark_dirty()
        self._record_event(
            action="add_cell",
            cell_id=cell_id,
            source_after=action.source,
            executed=bool(result),
            execution_result=result.to_dict() if result else None,
        )
        observation = self._observation_from_execution(
            "add_cell",
            cell_id=cell_id,
            result=result,
            source=action.source,
        )
        return self._record_observation(observation)

    def _update_cell(self, action: Action) -> Observation:
        edit = self.notebook.update_cell(action.cell_id or "", action.source)
        result = None
        if action.execute and self.notebook.get_cell(action.cell_id or "").cell_type == "code":
            result = self._execute_and_store_cell(action.cell_id or "")
        self._mark_dirty()
        self._record_event(
            action="update_cell",
            cell_id=action.cell_id,
            source_before=edit["before"]["source"],
            source_after=action.source,
            executed=bool(result),
            execution_result=result.to_dict() if result else None,
        )
        observation = self._observation_from_execution(
            "update_cell",
            cell_id=action.cell_id,
            result=result,
            source=action.source,
        )
        return self._record_observation(observation)

    def _delete_cell(self, action: Action) -> Observation:
        edit = self.notebook.delete_cell(action.cell_id or "")
        self._mark_dirty()
        self._record_event(
            action="delete_cell",
            cell_id=action.cell_id,
            source_before=edit["before"]["source"],
        )
        observation = self._observation(
            action="delete_cell",
            stdout=f"Deleted cell {action.cell_id}.",
            cell_id=action.cell_id,
        )
        return self._record_observation(observation)

    def _move_cell(self, action: Action) -> Observation:
        moved = self.notebook.move_cell(action.cell_id or "", action.new_position or 0)
        self._mark_dirty()
        self._record_event(action="move_cell", cell_id=action.cell_id, move=moved)
        observation = self._observation(
            action="move_cell",
            stdout=f"Moved cell {action.cell_id} to position {action.new_position}.",
            cell_id=action.cell_id,
        )
        return self._record_observation(observation)

    def _run_cell(self, cell_id: str) -> Observation:
        result = self._execute_and_store_cell(cell_id)
        self.dirty_since_clean_run = True
        self.last_validated_candidate_id = None
        source = str(self.notebook.get_cell(cell_id).source)
        self._record_event(
            action="run_cell",
            cell_id=cell_id,
            executed=True,
            execution_result=result.to_dict(),
        )
        observation = self._observation_from_execution(
            "run_cell",
            cell_id=cell_id,
            result=result,
            source=source,
        )
        return self._record_observation(observation)

    def _inspect_notebook(self) -> Observation:
        cells = self.notebook.list_cells()
        text = json.dumps(cells, indent=2)
        self._record_event(action="inspect_notebook")
        observation = self._observation(action="inspect_notebook", stdout=text)
        return self._record_observation(observation)

    def _execute_and_store_cell(self, cell_id: str) -> CellExecutionResult:
        cell = self.notebook.get_cell(cell_id)
        if cell.cell_type != "code":
            raise TypeError(f"Cell {cell_id} is not a code cell.")
        result = self.kernel.execute_cell(str(cell.source), timeout=self.kernel_timeout)
        self.cell_executions_total += 1
        self.notebook.set_cell_outputs(
            cell_id,
            outputs=result.outputs,
            execution_count=result.execution_count,
        )
        if not result.success:
            self.errors_count += 1
        self._record_cell_execution(cell_id=cell_id, source=str(cell.source), result=result)
        return result

    def _observation_from_execution(
        self,
        action: str,
        *,
        cell_id: str | None,
        result: CellExecutionResult | None,
        source: str,
    ) -> Observation:
        feedback_items: list[FeedbackItem] = []
        hints: list[str] = []
        stdout = ""
        if result is not None:
            stdout = result.compact_text()
            if not result.success:
                feedback_items.append(
                    FeedbackItem(
                        channel="runtime",
                        key="cell_error",
                        message=f"{result.error_name}: {result.error_value}",
                        severity="blocker",
                        visible_to_agent=True,
                        cell_id=cell_id,
                    )
                )
            else:
                model_feedback = self._model_validation_feedback_items()
                feedback_items.extend(model_feedback)
            if self.mode.checklist_feedback_enabled:
                checklist_items = self.checklist.record_execution(
                    source=source,
                    stdout=stdout,
                    cell_id=cell_id,
                    step=self.state.step,
                    execution_success=result.success,
                    has_runtime_error=not result.success,
                    has_contract_blocker=any(
                        item.channel == "contract" and item.severity == "blocker"
                        for item in feedback_items
                    ),
                )
                hints = [item.message for item in checklist_items]
                feedback_items.extend(checklist_items)
        return self._observation(
            action=action,
            stdout=stdout,
            hints=hints,
            feedback_items=feedback_items,
            cell_id=cell_id,
        )

    def _observation(
        self,
        *,
        action: str,
        stdout: str = "",
        feedback_items: list[FeedbackItem] | None = None,
        hints: list[str] | None = None,
        cell_id: str | None = None,
        done: bool = False,
        submitted: bool = False,
        model_var: str | None = None,
        validation_metric: float | None = None,
        final_status: str | None = None,
        null_reason: str | None = None,
        finalize_path: str | None = None,
        stage: str | None = None,
        thoughts: str | None = None,
    ) -> Observation:
        feedback_items = feedback_items or []
        stderr = "\n".join(
            item.message
            for item in feedback_items
            if item.visible_to_agent and item.channel == "contract"
        )
        active = self._active_action
        observation_stage = stage if stage is not None else (active.stage if active else self.current_stage)
        observation_thoughts = thoughts if thoughts is not None else (
            active.thoughts if active and self.enable_thoughts else None
        )
        return Observation(
            action=action,
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
            stage=observation_stage,
            thoughts=observation_thoughts,
            stdout=stdout,
            stderr=stderr,
            hints=hints or [],
            checklist_coverage=self.checklist.coverage(),
            done=done or self.state.step >= self.state.max_steps,
            submitted=submitted,
            model_var=model_var,
            cell_id=cell_id,
            notebook_status=self._notebook_status(),
            feedback_items=[item.to_dict() for item in feedback_items],
            validation_metric=validation_metric,
            final_status=final_status,
            null_reason=null_reason,
            finalize_path=finalize_path,
        )

    def _record_observation(self, observation: Observation) -> Observation:
        if observation.stage is None:
            observation.stage = self.current_stage
        self.state.history.append(observation)
        self.feedback_trace.append(self._visible_observation_dict(observation))
        self._save_artifacts()
        return observation

    def _add_thought(self, action: Action) -> None:
        """Append a visible agent thought to the persistent scratchpad."""
        thought = {
            "step": self.state.step + 1,
            "type": action.type,
            "stage": action.stage,
            "thoughts": str(action.thoughts).strip(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.scratchpad.append(thought)

    def scratchpad_digest(self, max_notes: int = 30, max_chars: int = 4000) -> str:
        """Render accumulated visible thoughts to re-inject into the agent context."""
        if not self.scratchpad:
            return ""
        lines = [
            f"- (step {n['step']}, {n.get('stage') or 'unknown'}) {n['thoughts']}"
            for n in self.scratchpad[-max_notes:]
        ]
        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[-max_chars:]
        return "[YOUR THOUGHTS SO FAR]\n" + body

    def _record_event(self, **event: Any) -> None:
        event_record = {
            "event_id": str(uuid.uuid4()),
            "step": self.state.step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notebook_revision": self.notebook.revision,
        }
        if "stage" not in event and self.current_stage:
            event_record["stage"] = self.current_stage
        if "thoughts" not in event and self._active_action and self.enable_thoughts:
            event_record["thoughts"] = self._active_action.thoughts
        event_record.update(event)
        self.events.append(_json_safe(event_record))

    def _save_artifacts(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.notebook.save()
        final_ipynb = self.workspace_dir / "final_notebook.ipynb"
        shutil.copyfile(self.notebook.path, final_ipynb)
        self.notebook.export_python(self.workspace_dir / "final_notebook.py")
        _write_json(
            self.workspace_dir / "notebook_events.json",
            [self._public_event(event) for event in self.events],
        )
        _write_json(self.workspace_dir / "feedback_trace.json", self.feedback_trace)
        _write_json(
            self.workspace_dir / "validation_trajectory.json",
            [self._public_candidate_record(record) for record in self.validation_trajectory],
        )
        _write_json(self.workspace_dir / "episode_summary.json", self.get_public_summary())
        if self.enable_thoughts:
            _write_json(self.workspace_dir / "scratchpad.json", self.scratchpad)
        _write_json(self.private_dir / "notebook_events_private.json", self.events)
        _write_json(
            self.private_dir / "feedback_trace_private.json",
            [observation.to_private_dict() for observation in self.state.history],
        )
        _write_json(
            self.private_dir / "validation_trajectory_private.json",
            self.validation_trajectory,
        )
        _write_json(
            self.private_dir / "candidate_diagnostics_private.json",
            self.candidate_diagnostics,
        )
        _write_json(
            self.private_dir / "checklist_private.json",
            self.checklist.to_private_dict(),
        )
        _write_json(self.private_dir / "episode_summary.json", self.get_summary())

    def _visible_observation_dict(self, observation: Observation) -> dict[str, Any]:
        data = observation.to_dict()
        data.pop("test_metric", None)
        data.pop("checklist_coverage", None)
        data.pop("action", None)
        return _json_safe(data)

    def get_public_summary(self) -> dict:
        private_keys = {
            "checklist_coverage",
            "private_checklist_coverage",
            "test_metric",
            "final_test_metric",
            "valid_submit",
            "submit_failure_type",
            "private_episode_dir",
        }
        return {
            key: value
            for key, value in self.get_summary().items()
            if key not in private_keys
        }

    def _public_candidate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        public = dict(record)
        public.pop("artifact_path", None)
        return public

    def _public_event(self, event: dict[str, Any]) -> dict[str, Any]:
        public = copy.deepcopy(event)
        public.pop("private", None)
        if "type" not in public and "action" in public:
            public["type"] = public.pop("action")
        else:
            public.pop("action", None)
        candidate = public.get("candidate")
        if isinstance(candidate, dict):
            candidate.pop("artifact_path", None)
        return public

    def _mark_dirty(self) -> None:
        self.dirty_since_clean_run = True
        self.last_validated_candidate_id = None
        self.candidates.clear()
        self._candidate_objects.clear()

    def _clean_state_blocker(self) -> FeedbackItem | None:
        if (
            self.dirty_since_clean_run
            or self.last_clean_run_id is None
            or self.notebook_revision_at_clean_run != self.notebook.revision
        ):
            return self._contract_feedback(
                "clean_run_required",
                NEEDS_CLEAN_RUN_MESSAGE,
                severity="blocker",
            )
        return None

    def _contract_feedback(
        self,
        key: str,
        message: str,
        *,
        severity: str = "warning",
        cell_id: str | None = None,
    ) -> FeedbackItem:
        self.contract_feedback_count += 1
        return FeedbackItem(
            channel="contract",
            key=key,
            message=message,
            severity=severity,  # type: ignore[arg-type]
            visible_to_agent=True,
            cell_id=cell_id,
        )

    def _register_validated_candidate(
        self,
        diagnostic: CandidateDiagnostic,
        model_var: str,
    ) -> CandidateRecord:
        candidate_id = str(uuid.uuid4())
        artifact_path = self.private_dir / "artifacts" / f"{candidate_id}.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as fh:
            cloudpickle.dump(diagnostic.model, fh)
        record = CandidateRecord(
            candidate_id=candidate_id,
            model_var=model_var,
            notebook_revision=self.notebook.revision,
            clean_run_id=self.last_clean_run_id or "",
            validation_metric=diagnostic.validation_metric,
            validation_success=True,
            raw_inference_ready=True,
            artifact_path=str(artifact_path),
        )
        self.candidates.add(record)
        self._candidate_objects[candidate_id] = diagnostic.model
        self.last_validated_candidate_id = candidate_id
        self.validation_trajectory.append(record.to_dict())
        self.checklist.record_structural(
            "baseline_candidate_created",
            reason="candidate exists after clean run",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "baseline_first",
            reason="candidate exists",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "raw_row_inference_ready",
            reason="raw validation inference succeeded",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "serialization_reproducibility",
            reason="candidate serialized with cloudpickle",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "candidate_validation_before_submit",
            reason="environment validation succeeded",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "validation_evaluated",
            reason="environment validation succeeded",
            step=self.state.step,
        )
        self.checklist.record_structural(
            "submit_ready_artifact",
            reason="raw validation inference succeeded",
            step=self.state.step,
        )
        return record

    def _discover_predictable_kernel_vars(self) -> list[str]:
        marker = f"__AUTOVIBE_VARS_{uuid.uuid4().hex}__"
        source = f"""
import json as _autovibe_json
_autovibe_names = []
for _autovibe_name, _autovibe_obj in list(globals().items()):
    if _autovibe_name.startswith("_"):
        continue
    try:
        if not isinstance(_autovibe_obj, type) and callable(getattr(_autovibe_obj, "predict", None)):
            _autovibe_names.append(_autovibe_name)
    except Exception:
        pass
print({marker!r} + _autovibe_json.dumps(_autovibe_names, ensure_ascii=False))
""".strip()
        try:
            result = self.kernel.execute_cell(source, store_history=False, timeout=min(self.kernel_timeout, 20))
        except Exception as exc:
            self._record_event(
                action="discover_predictable_kernel_vars_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return []
        if not result.success:
            self._record_event(
                action="discover_predictable_kernel_vars_failed",
                error=f"{result.error_name}: {result.error_value}",
            )
            return []
        for line in result.stdout.splitlines():
            if marker in line:
                payload = line.split(marker, 1)[1]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    return []
                if isinstance(data, list):
                    return [str(name) for name in data if isinstance(name, str)]
        return []

    def _candidate_var_order(self) -> list[str]:
        discovered = self._discover_predictable_kernel_vars()
        ordered: list[str] = []
        for name in list(self._CANDIDATE_VAR_NAMES) + discovered:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _kernel_var_summary(self, model_var: str) -> dict[str, Any]:
        marker = f"__AUTOVIBE_VAR_{uuid.uuid4().hex}__"
        source = f"""
import json as _autovibe_json
_autovibe_var = {model_var!r}
_autovibe_exists = _autovibe_var in globals()
_autovibe_has_predict = False
if _autovibe_exists:
    try:
        _autovibe_has_predict = callable(getattr(globals()[_autovibe_var], "predict", None))
    except Exception:
        _autovibe_has_predict = False
print({marker!r} + _autovibe_json.dumps({{
    "exists": _autovibe_exists,
    "has_predict": _autovibe_has_predict,
}}, ensure_ascii=False))
""".strip()
        result = self.kernel.execute_cell(source, store_history=False, timeout=min(self.kernel_timeout, 20))
        if not result.success:
            return {
                "exists": None,
                "has_predict": None,
                "error_type": result.error_name or "RuntimeError",
                "error_message": result.error_value or result.stderr,
            }
        for line in result.stdout.splitlines():
            if marker in line:
                try:
                    return json.loads(line.split(marker, 1)[1])
                except json.JSONDecodeError:
                    break
        return {
            "exists": None,
            "has_predict": None,
            "error_type": "RuntimeError",
            "error_message": "Could not parse kernel variable summary.",
        }

    def _check_candidate_readiness(
        self,
        model_var: str,
        *,
        source: str,
        sample_rows: int | None = 32,
        compute_metric: bool = False,
    ) -> CandidateDiagnostic:
        diagnostic = CandidateDiagnostic(
            step=self.state.step,
            candidate_var=model_var,
            source=source,
            sample_rows=sample_rows,
        )
        try:
            summary = self._kernel_var_summary(model_var)
            diagnostic.exists = summary.get("exists")
            diagnostic.has_predict = summary.get("has_predict")
            if summary.get("error_type"):
                diagnostic.error_type = str(summary.get("error_type"))
                diagnostic.error_message = str(summary.get("error_message") or "")
                self._record_candidate_diagnostic(diagnostic)
                return diagnostic
            if not diagnostic.exists or not diagnostic.has_predict:
                self._record_candidate_diagnostic(diagnostic)
                return diagnostic

            try:
                model = self._load_candidate_from_kernel(model_var)
                diagnostic.model = model
            except Exception as exc:
                diagnostic.serializable = False
                diagnostic.error_type = type(exc).__name__
                diagnostic.error_message = _clip(str(exc), 800)
                self._record_candidate_diagnostic(diagnostic)
                return diagnostic

            try:
                cloudpickle.dumps(model)
                diagnostic.serializable = True
            except Exception as exc:
                diagnostic.serializable = False
                diagnostic.error_type = type(exc).__name__
                diagnostic.error_message = _clip(str(exc), 800)
                self._record_candidate_diagnostic(diagnostic)
                return diagnostic

            X_val = self.state.val.drop(columns=[self.state.target_col])
            y_val = self.state.val[self.state.target_col]
            if sample_rows is not None and sample_rows > 0:
                X_eval = X_val.head(sample_rows)
                y_eval = y_val.head(sample_rows)
            else:
                X_eval = X_val
                y_eval = y_val
            try:
                preds = model.predict(X_eval)
                diagnostic.predictions = preds
                diagnostic.raw_val_predict_ok = True
                diagnostic.prediction_length_ok = _prediction_length(preds) == len(X_eval)
                diagnostic.prediction_nan_free = _predictions_nan_free(preds)
                if compute_metric and diagnostic.prediction_length_ok and diagnostic.prediction_nan_free:
                    try:
                        diagnostic.validation_metric = _score_with_coercion(
                            self.metric_fn,
                            y_eval,
                            preds,
                        )
                    except Exception as exc:
                        diagnostic.error_type = type(exc).__name__
                        diagnostic.error_message = _clip(str(exc), 800)
            except Exception as exc:
                diagnostic.raw_val_predict_ok = False
                diagnostic.error_type = type(exc).__name__
                diagnostic.error_message = _clip(str(exc), 800)
        except Exception as exc:
            diagnostic.error_type = type(exc).__name__
            diagnostic.error_message = _clip(str(exc), 800)

        self._record_candidate_diagnostic(diagnostic)
        return diagnostic

    def _record_candidate_diagnostic(self, diagnostic: CandidateDiagnostic) -> None:
        data = diagnostic.to_dict()
        self.candidate_diagnostics.append(_json_safe(data))
        self._append_private_jsonl("candidate_diagnostics_private.jsonl", data)
        if (
            diagnostic.exists is False
            or diagnostic.has_predict is False
            or diagnostic.raw_val_predict_ok is False
            or diagnostic.prediction_length_ok is False
            or diagnostic.prediction_nan_free is False
            or diagnostic.serializable is False
            or _validation_scoring_failed(diagnostic)
        ):
            self.model_check_failure_count += 1

    def _candidate_failure_feedback(
        self,
        diagnostic: CandidateDiagnostic,
        *,
        source: str,
    ) -> FeedbackItem:
        if diagnostic.serializable is False and diagnostic.raw_val_predict_ok is not False:
            return self._contract_feedback(
                "candidate_serialization_failed",
                MODEL_SERIALIZATION_MESSAGE.format(model_var=diagnostic.candidate_var),
                severity="blocker",
            )
        error_type = diagnostic.error_type or "RuntimeError"
        error_message = _clip(diagnostic.error_message or "prediction failed", 280)
        if source == "validate":
            instruction = (
                "The final candidate must reproduce all preprocessing when called "
                "on raw validation or hidden-test rows."
            )
        else:
            instruction = (
                "The final artifact must include all preprocessing needed when "
                "predict() is called on raw rows."
            )
        return self._contract_feedback(
            "raw_validation_readiness",
            (
                f"[MODEL CHECK] Candidate variable '{diagnostic.candidate_var}' cannot "
                f"predict raw validation features: {error_type}: {error_message}.\n"
                f"{instruction}"
            ),
            severity="blocker",
        )

    def _model_validation_feedback_items(self) -> list[FeedbackItem]:
        items: list[FeedbackItem] = []
        names = self._discover_predictable_kernel_vars()
        ordered = [name for name in self._candidate_var_order() if name in names]
        for model_var in ordered:
            diagnostic = self._check_candidate_readiness(
                model_var,
                source="auto_model_check",
                sample_rows=32,
                compute_metric=False,
            )
            broken = (
                diagnostic.has_predict is False
                or diagnostic.raw_val_predict_ok is False
                or diagnostic.prediction_length_ok is False
                or diagnostic.prediction_nan_free is False
                or diagnostic.serializable is False
            )
            if not broken:
                continue
            message_hash = hashlib.sha256(
                (diagnostic.error_message or str(diagnostic.to_dict())).encode("utf-8", errors="ignore")
            ).hexdigest()[:12]
            signature = (diagnostic.candidate_var, diagnostic.error_type, message_hash)
            if signature in self._model_check_seen:
                continue
            self._model_check_seen.add(signature)
            items.append(self._candidate_failure_feedback(diagnostic, source="auto_model_check"))
        return items[:3]

    def _validate_auto_candidate(self, *, allow_dirty: bool) -> Observation:
        blocker = None if allow_dirty else self._clean_state_blocker()
        if blocker is not None:
            observation = self._observation(
                action="validate",
                feedback_items=[blocker],
                model_var="auto",
            )
            return self._record_observation(observation)

        attempted: list[str] = []
        failures: list[dict[str, Any]] = []
        for name in self._candidate_var_order():
            attempted.append(name)
            diagnostic = self._check_candidate_readiness(
                name,
                source="validate_auto",
                sample_rows=None,
                compute_metric=True,
            )
            if self._diagnostic_is_submit_ready(diagnostic):
                record = self._register_validated_candidate(diagnostic, name)
                self._record_event(
                    action="validate",
                    model_var="auto",
                    selected_model_var=name,
                    attempted_vars=attempted,
                    candidate=record.to_dict(),
                )
                observation = self._observation(
                    action="validate",
                    stdout=(
                        f"selected_model_var={name}\n"
                        f"validation_{self.state.metric_name}={diagnostic.validation_metric:.6f}"
                    ),
                    model_var=name,
                    validation_metric=diagnostic.validation_metric,
                )
                return self._record_observation(observation)
            failures.append(
                {
                    "candidate_var": name,
                    "error_type": diagnostic.error_type,
                    "error_message": diagnostic.error_message,
                    "has_predict": diagnostic.has_predict,
                    "raw_val_predict_ok": diagnostic.raw_val_predict_ok,
                    "serializable": diagnostic.serializable,
                }
            )

        self._record_event(action="validate_auto_failed", attempted_vars=attempted, private={"failures": failures})
        observation = self._observation(
            action="validate",
            feedback_items=[
                self._contract_feedback(
                    "no_raw_validation_ready_candidate",
                    (
                        "[MODEL CHECK] No discovered candidate variable could predict raw validation rows "
                        "and serialize. Put all preprocessing inside a fitted Pipeline/estimator, then validate again."
                    ),
                    severity="blocker",
                )
            ],
            model_var="auto",
        )
        return self._record_observation(observation)

    def _submit_auto_candidate(self, *, allow_dirty: bool) -> Observation:
        candidate = self.candidates.latest()
        if (
            candidate is not None
            and candidate.validation_success
            and (allow_dirty or (
                not self.dirty_since_clean_run
                and candidate.clean_run_id == self.last_clean_run_id
                and candidate.notebook_revision == self.notebook.revision
            ))
        ):
            return self.submit_by_name(candidate.model_var, allow_dirty=allow_dirty)

        validation = self._validate_auto_candidate(allow_dirty=allow_dirty)
        latest = self.candidates.latest()
        if latest is not None and latest.validation_success:
            return self.submit_by_name(latest.model_var, allow_dirty=allow_dirty)
        observation = self._observation(
            action="submit",
            feedback_items=[
                self._contract_feedback(
                    "validation_required",
                    "Auto-submit could not find a raw-validation-ready candidate. Validate or finalize after fixing the candidate.",
                    severity="blocker",
                )
            ],
            model_var="auto",
        )
        return self._record_observation(observation)

    def _diagnostic_is_submit_ready(self, diagnostic: CandidateDiagnostic) -> bool:
        return bool(
            diagnostic.exists
            and diagnostic.has_predict
            and diagnostic.raw_val_predict_ok
            and diagnostic.prediction_length_ok
            and diagnostic.prediction_nan_free
            and diagnostic.serializable
            and diagnostic.validation_metric is not None
        )

    def inspect_data(self) -> Observation:
        profile = build_compact_profile(
            self.state.train,
            self.state.val,
            self.state.target_col,
            self.state.metric_name,
        )
        self._write_private_json("data_inspection_private.json", profile)
        text = build_dataset_card(
            self.state.train,
            self.state.val,
            self.state.target_col,
            self.state.metric_name,
            max_chars=5500,
        )
        self._record_event(action="inspect_data")
        observation = self._observation(action="inspect_data", stdout=text)
        return self._record_observation(observation)

    def profile_data(self, profile: str = "compact") -> Observation:
        compact = build_compact_profile(
            self.state.train,
            self.state.val,
            self.state.target_col,
            self.state.metric_name,
        )
        private_payload: dict[str, Any] = {"compact": compact, "backend": "compact"}
        cfg = profile_config_from_env()
        ydata_result: dict[str, Any] = {}
        wants_ydata = profile == "ydata" or bool(cfg["enable_ydata"])
        if wants_ydata:
            ydata_result = run_ydata_profile(
                self.state.train,
                self.state.target_col,
                self.private_dir,
                max_rows=int(cfg["max_rows"]),
                max_cols=int(cfg["max_cols"]),
                timeout_sec=int(cfg["timeout_sec"]),
                minimal=bool(cfg["minimal"]),
            )
            private_payload["ydata"] = ydata_result
        private_payload["backend"] = "compact+ydata" if ydata_result.get("success") else "compact"
        self._write_private_json("data_profile_private.json", private_payload)
        summary_text = (
            format_ydata_profile_for_agent(compact, ydata_result, max_chars=5500)
            if wants_ydata
            else format_profile_for_agent(compact, max_chars=5500)
        )
        (self.private_dir / "data_profile_summary.txt").write_text(summary_text, encoding="utf-8")
        self._record_event(action="profile_data", profile=profile, ydata_success=ydata_result.get("success"))
        observation = self._observation(action="profile_data", stdout=summary_text)
        return self._record_observation(observation)

    def list_candidates(self) -> Observation:
        discovered = self._discover_predictable_kernel_vars()
        ordered = [name for name in self._candidate_var_order() if name in discovered]
        private = {"discovered": discovered, "ordered": ordered}
        self._record_event(action="list_candidates", private=private)
        text = "[CANDIDATES]\n" + (
            "\n".join(f"- {name}" for name in ordered)
            if ordered
            else "No predict-capable candidate variables were discovered."
        )
        observation = self._observation(action="list_candidates", stdout=text)
        return self._record_observation(observation)

    def check_candidate(self, model_var: str = "auto") -> Observation:
        names = self._candidate_var_order() if model_var == "auto" else [model_var]
        diagnostics = [
            self._check_candidate_readiness(
                name,
                source="check_candidate",
                sample_rows=32,
                compute_metric=False,
            )
            for name in names
        ]
        ready = [diag for diag in diagnostics if self._diagnostic_is_check_ready(diag)]
        selected = ready[0].candidate_var if ready else None
        lines = ["[CANDIDATE CHECK]"]
        if selected:
            lines.append(f"selected_model_var={selected}")
        for diag in diagnostics[:12]:
            lines.append(
                (
                    f"- {diag.candidate_var}: exists={diag.exists} has_predict={diag.has_predict} "
                    f"raw_validation_ready={diag.raw_val_predict_ok} "
                    f"prediction_length_ok={diag.prediction_length_ok} "
                    f"prediction_nan_free={diag.prediction_nan_free} "
                    f"serializable={diag.serializable}"
                )
            )
            if diag.error_type:
                lines.append(f"  issue={diag.error_type}: {_clip(diag.error_message or '', 220)}")
        self._record_event(action="check_candidate", model_var=model_var, selected_model_var=selected)
        observation = self._observation(action="check_candidate", stdout=_clip("\n".join(lines), 5000), model_var=selected or model_var)
        return self._record_observation(observation)

    def _diagnostic_is_check_ready(self, diagnostic: CandidateDiagnostic) -> bool:
        return bool(
            diagnostic.exists
            and diagnostic.has_predict
            and diagnostic.raw_val_predict_ok
            and diagnostic.prediction_length_ok
            and diagnostic.prediction_nan_free
            and diagnostic.serializable
        )

    def quick_validate(self, model_var: str = "auto") -> Observation:
        names = self._candidate_var_order() if model_var == "auto" else [model_var]
        diagnostics: list[CandidateDiagnostic] = []
        for name in names:
            diag = self._check_candidate_readiness(
                name,
                source="quick_validate",
                sample_rows=None,
                compute_metric=True,
            )
            diagnostics.append(diag)
            if self._diagnostic_is_submit_ready(diag):
                text = (
                    "[QUICK VALIDATE]\n"
                    f"selected_model_var={name}\n"
                    f"validation_{self.state.metric_name}={diag.validation_metric:.6f}\n"
                    "raw_validation_ready=true\n"
                    "serializable=true\n"
                    "This does not create a clean submit-ready candidate; run restart_and_run_all and validate/finalize before submit."
                )
                self._record_event(action="quick_validate", model_var=model_var, selected_model_var=name)
                observation = self._observation(
                    action="quick_validate",
                    stdout=text,
                    model_var=name,
                    validation_metric=diag.validation_metric,
                )
                return self._record_observation(observation)
        lines = ["[QUICK VALIDATE]", "No candidate could be scored on raw validation rows."]
        for diag in diagnostics[:8]:
            if diag.error_type:
                lines.append(f"- {diag.candidate_var}: {diag.error_type}: {_clip(diag.error_message or '', 220)}")
            else:
                lines.append(f"- {diag.candidate_var}: raw_validation_ready={diag.raw_val_predict_ok}")
        observation = self._observation(action="quick_validate", stdout="\n".join(lines), model_var=model_var)
        return self._record_observation(observation)

    def cleanlab_diagnose(
        self,
        model_var: str = "auto",
        *,
        source: str = "validation_or_cv",
        max_issues: int = 20,
    ) -> Observation:
        if os.getenv("AUTOVIBE_ENABLE_CLEANLAB", "0") != "1":
            text = (
                "[CLEANLAB DIAGNOSTIC] cleanlab diagnostics are disabled. "
                "Set AUTOVIBE_ENABLE_CLEANLAB=1 and install cleanlab to enable this optional tool."
            )
            self._record_event(action="cleanlab_diagnose", enabled=False)
            return self._record_observation(self._observation(action="cleanlab_diagnose", stdout=text, model_var=model_var))
        try:
            from cleanlab.filter import find_label_issues
        except ImportError:
            text = (
                "[CLEANLAB DIAGNOSTIC] cleanlab is unavailable, so label-issue detection cannot run."
            )
            self._record_event(action="cleanlab_diagnose", enabled=True, available=False)
            return self._record_observation(self._observation(action="cleanlab_diagnose", stdout=text, model_var=model_var))

        if not self._looks_like_classification():
            text = (
                "[CLEANLAB DIAGNOSTIC] This target does not look like a bounded classification label, "
                "so label-issue detection was skipped."
            )
            return self._record_observation(self._observation(action="cleanlab_diagnose", stdout=text, model_var=model_var))

        selected = self._select_candidate_with_predict_proba(model_var)
        if selected is None:
            text = (
                "[CLEANLAB DIAGNOSTIC] The selected candidate does not provide predict_proba, "
                "so label-issue detection cannot run. Try a probabilistic classifier or use quick_validate/check_candidate first."
            )
            return self._record_observation(self._observation(action="cleanlab_diagnose", stdout=text, model_var=model_var))

        selected_name, model = selected
        max_rows = min(_int_env("AUTOVIBE_CLEANLAB_MAX_ROWS", 5000), len(self.state.val))
        X_val = self.state.val.drop(columns=[self.state.target_col]).head(max_rows)
        y_val = self.state.val[self.state.target_col].head(max_rows)
        try:
            pred_probs = model.predict_proba(X_val)
            issues = find_label_issues(labels=y_val.to_numpy(), pred_probs=pred_probs, return_indices_ranked_by="self_confidence")
            issue_indices = [int(i) for i in list(issues)[:max_issues]]
            rows = [
                {
                    "validation_index": idx,
                    "label": _json_safe(y_val.iloc[idx]),
                    "predicted_label": _json_safe(model.classes_[int(np.argmax(pred_probs[idx]))]) if hasattr(model, "classes_") else None,
                    "self_confidence": float(pred_probs[idx].max()),
                }
                for idx in issue_indices
                if idx < len(y_val)
            ]
            payload = {
                "model_var": selected_name,
                "source": source,
                "rows_evaluated": int(len(y_val)),
                "issue_count": int(len(issue_indices)),
                "issues": rows,
            }
            self._write_private_json("cleanlab_diagnostics_private.json", payload)
            self._write_cleanlab_csv(rows)
            lines = [
                "[CLEANLAB DIAGNOSTIC]",
                f"model_var={selected_name}",
                f"Possible label issues detected: {len(issue_indices)} / {len(y_val)} validation rows.",
                "Top suspicious rows are listed by validation index. Treat this as a diagnostic signal, not ground truth.",
            ]
            for row in rows[:max_issues]:
                lines.append(
                    f"- validation_index={row['validation_index']} label={row['label']} predicted={row['predicted_label']} confidence={row['self_confidence']:.4f}"
                )
            self._record_event(action="cleanlab_diagnose", model_var=selected_name, issue_count=len(issue_indices))
            observation = self._observation(action="cleanlab_diagnose", stdout=_clip("\n".join(lines), 5000), model_var=selected_name)
            return self._record_observation(observation)
        except Exception as exc:
            text = f"[CLEANLAB DIAGNOSTIC] cleanlab failed: {type(exc).__name__}: {_clip(str(exc), 240)}"
            self._record_event(action="cleanlab_diagnose_failed", model_var=selected_name, error=text)
            return self._record_observation(self._observation(action="cleanlab_diagnose", stdout=text, model_var=selected_name))

    def tune_hyperparameters(
        self,
        model_var: str,
        *,
        search_space: dict[str, Any],
        n_trials: int,
        timeout_sec: int,
        scoring: str = "metric",
    ) -> Observation:
        del scoring
        n_trials = min(max(int(n_trials), 1), 30)
        timeout_sec = min(max(int(timeout_sec), 1), 120)
        diagnostic = self._check_candidate_readiness(
            model_var,
            source="tune_hyperparameters_preflight",
            sample_rows=32,
            compute_metric=False,
        )
        if not self._diagnostic_is_check_ready(diagnostic):
            observation = self._observation(
                action="tune_hyperparameters",
                feedback_items=[self._candidate_failure_feedback(diagnostic, source="tune_hyperparameters")],
                model_var=model_var,
            )
            return self._record_observation(observation)
        if not search_space:
            text = "[TUNING]\nNo search_space was provided; tuning skipped."
            return self._record_observation(self._observation(action="tune_hyperparameters", stdout=text, model_var=model_var))

        started = time.time()
        trials: list[dict[str, Any]] = []
        best_model = None
        best_metric = None
        best_params: dict[str, Any] = {}
        X_train = self.state.train.drop(columns=[self.state.target_col])
        y_train = self.state.train[self.state.target_col]
        X_val = self.state.val.drop(columns=[self.state.target_col])
        y_val = self.state.val[self.state.target_col]
        try:
            from sklearn.base import clone
        except Exception as exc:
            text = f"[TUNING]\nCannot clone the candidate: {type(exc).__name__}: {_clip(str(exc), 240)}"
            return self._record_observation(self._observation(action="tune_hyperparameters", stdout=text, model_var=model_var))

        for trial_idx in range(n_trials):
            if time.time() - started >= timeout_sec:
                break
            params = _sample_search_params(search_space, random.Random(42 + trial_idx))
            try:
                model = clone(diagnostic.model)
                model.set_params(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                metric = _score_with_coercion(self.metric_fn, y_val, preds)
                trials.append({"trial": trial_idx + 1, "params": params, "validation_metric": metric, "success": True})
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_model = model
                    best_params = params
            except Exception as exc:
                trials.append(
                    {
                        "trial": trial_idx + 1,
                        "params": params,
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error_message": _clip(str(exc), 300),
                    }
                )
        private = {
            "base_model_var": model_var,
            "new_model_var": "tuned_model",
            "trials": trials,
            "best_validation_metric": best_metric,
            "best_params": best_params,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        self._write_private_json("tuning_diagnostics_private.json", private)
        if best_model is None:
            text = "[TUNING]\nNo trial completed successfully within the configured caps."
            return self._record_observation(self._observation(action="tune_hyperparameters", stdout=text, model_var=model_var))
        self._inject_model_into_kernel("tuned_model", best_model)
        tuned_diag = self._check_candidate_readiness(
            "tuned_model",
            source="tune_hyperparameters",
            sample_rows=None,
            compute_metric=True,
        )
        lines = [
            "[TUNING]",
            f"base_model_var={model_var}",
            "new_model_var=tuned_model",
            f"trials_completed={sum(1 for t in trials if t.get('success'))}",
            f"best_validation_metric={best_metric:.6f}",
            f"best_params={_format_compact(best_params)}",
            f"raw_validation_ready={str(bool(tuned_diag.raw_val_predict_ok)).lower()}",
            f"serializable={str(bool(tuned_diag.serializable)).lower()}",
        ]
        self._record_event(action="tune_hyperparameters", model_var=model_var, new_model_var="tuned_model", trials_completed=len(trials))
        observation = self._observation(
            action="tune_hyperparameters",
            stdout=_clip("\n".join(lines), 5000),
            model_var="tuned_model",
            validation_metric=tuned_diag.validation_metric,
        )
        return self._record_observation(observation)

    def _select_candidate_with_predict_proba(self, model_var: str) -> tuple[str, Any] | None:
        names = self._candidate_var_order() if model_var == "auto" else [model_var]
        for name in names:
            try:
                model = self._load_candidate_from_kernel(name)
            except Exception:
                continue
            if callable(getattr(model, "predict_proba", None)):
                return name, model
        return None

    def _looks_like_classification(self) -> bool:
        y = self.state.train[self.state.target_col]
        return y.nunique(dropna=False) <= min(50, max(20, int(len(y) * 0.2)))

    def _inject_model_into_kernel(self, variable_name: str, model: Any) -> None:
        tmp_dir = self.workspace_dir / ".autovibe_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{variable_name}_{uuid.uuid4().hex}.pkl"
        with tmp_path.open("wb") as fh:
            cloudpickle.dump(model, fh)
        source = f"""
import cloudpickle as _autovibe_cloudpickle
from pathlib import Path as _AutovibePath
with open(_AutovibePath({self.kernel.kernel_visible_path(tmp_path)!r}), "rb") as _autovibe_file:
    globals()[{variable_name!r}] = _autovibe_cloudpickle.load(_autovibe_file)
""".strip()
        result = self.kernel.execute_cell(source, store_history=False, timeout=min(self.kernel_timeout, 30))
        try:
            tmp_path.unlink()
        except OSError:
            pass
        if not result.success:
            raise RuntimeError(f"{result.error_name}: {result.error_value or result.stderr}")

    def _finalize_failure_observation(
        self,
        *,
        final_status: str,
        null_reason: str,
        finalize_path: str,
        attempted_vars: list[str],
    ) -> Observation:
        self.private_summary["final_status"] = final_status
        self.private_summary["null_reason"] = null_reason
        self.private_summary["finalize_path"] = finalize_path
        self.private_summary["finalized_by_host"] = True
        self.private_summary["reproducibility_level"] = "none"
        self.private_summary["finalize_attempted_vars"] = attempted_vars
        self.private_summary["valid_submit"] = False
        self.private_summary["final_test_metric"] = None
        self.private_summary["submit_failure_type"] = final_status
        self._record_event(
            action="finalize_failed",
            final_status=final_status,
            null_reason=null_reason,
            finalize_path=finalize_path,
            attempted_vars=attempted_vars,
        )
        observation = self._observation(
            action="finalize",
            stdout="[FINALIZE] No valid candidate was submitted.",
            done=True,
            final_status=final_status,
            null_reason=null_reason,
            finalize_path=finalize_path,
        )
        return self._record_observation(observation)

    def _record_cell_execution(
        self,
        *,
        cell_id: str,
        source: str,
        result: CellExecutionResult,
    ) -> None:
        record = {
            "cell_id": cell_id,
            "source_hash": hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest(),
            "source": source,
            "success": result.success,
            "error_name": result.error_name,
            "error_value": result.error_value,
            "traceback": result.traceback,
            "stdout_full": result.stdout,
            "stderr_full": result.stderr,
            "elapsed_seconds": result.elapsed_seconds,
        }
        self._append_private_jsonl("cell_executions_private.jsonl", record)

    def record_agent_turn(self, record: dict[str, Any]) -> None:
        self._agent_trace_turn += 1
        payload = {"turn": self._agent_trace_turn}
        payload.update(record)
        self._append_private_jsonl("agent_trace_private.jsonl", payload)

    def build_context_pack(self) -> dict[str, Any]:
        recent_model_failures = [
            item
            for item in self.candidate_diagnostics[-10:]
            if item.get("raw_val_predict_ok") is False or item.get("serializable") is False
        ]
        candidate_vars_seen = sorted(
            {
                str(item.get("candidate_var"))
                for item in self.candidate_diagnostics
                if item.get("candidate_var")
            }
        )
        active_blockers = [
            obs.stderr
            for obs in self.state.history[-6:]
            if obs.stderr.strip()
        ]
        return {
            "target_col": self.state.target_col,
            "metric": self.state.metric_name,
            "budget_remaining": self.budget_remaining(),
            "notebook_status": self._notebook_status(),
            "validated_candidates": [self._public_candidate_record(r.to_dict()) for r in self.candidates.all()],
            "candidate_vars_seen": candidate_vars_seen,
            "best_validation_metric": self._best_validation_metric(),
            "active_blockers": active_blockers,
            "recent_errors": active_blockers[-3:],
            "model_check_failures": recent_model_failures,
            "finalization_requirements": (
                "Keep one raw-row-ready, serializable candidate available. "
                "When ready or low on budget, use finalize with model_var='auto'."
            ),
        }

    def _write_private_json(self, filename: str, data: Any) -> None:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.private_dir / filename, data)

    def _append_private_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        path = self.private_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def _write_cleanlab_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.private_dir / "cleanlab_issues_private.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _load_candidate_from_kernel(self, model_var: str) -> Any:
        tmp_dir = self.workspace_dir / ".autovibe_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"candidate_{uuid.uuid4().hex}.pkl"
        self.kernel.dump_variable_to_file(model_var, tmp_path)
        with tmp_path.open("rb") as fh:
            model = cloudpickle.load(fh)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return model

    def _clear_generated_artifacts(self) -> None:
        for path in (
            self.workspace_dir / "data",
            self.workspace_dir / "artifacts",
            self.workspace_dir / ".autovibe_tmp",
        ):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        for filename in (
            "solution.ipynb",
            "final_notebook.ipynb",
            "final_notebook.py",
            "notebook_events.json",
            "feedback_trace.json",
            "validation_trajectory.json",
            "episode_summary.json",
            "scratchpad.json",
        ):
            path = self.workspace_dir / filename
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        if self.private_dir.exists():
            shutil.rmtree(self.private_dir, ignore_errors=True)

    def _notebook_status(self) -> dict[str, Any]:
        return {
            "dirty_since_clean_run": self.dirty_since_clean_run,
            "clean_run_available": self.last_clean_run_id is not None
            and not self.dirty_since_clean_run,
            "validated_candidate_available": self.candidates.latest() is not None,
            "notebook_revision": self.notebook.revision,
            "last_clean_run_id": self.last_clean_run_id,
        }

    def _build_context_prompt(self) -> dict:
        dataset_card = build_dataset_card(
            self.state.train,
            self.state.val,
            self.state.target_col,
            self.state.metric_name,
            max_chars=4500,
        )
        return {
            "task": (
                "You are solving a supervised ML task in a real Jupyter notebook.\n"
                f"Target column: '{self.state.target_col}'\n"
                f"Metric: {self.state.metric_name}\n"
                f"Max turns: {self.state.max_steps}\n\n"
                f"{dataset_card}\n\n"
                "A real .ipynb document is the source of truth for your solution. "
                "You may add, update, delete, move, inspect, and execute cells. "
                "The notebook uses a persistent Jupyter kernel, so variables survive "
                "between executed cells.\n\n"
                "Kernel variables already available: train_df, val_df, target_col, pd, np. "
                "You may import installed ML libraries, create artifacts in the episode "
                "workspace, and display plots or DataFrames.\n\n"
                "Hidden test data is not available in the kernel or workspace. "
                "Interactive state is not enough for final acceptance: run "
                "restart_and_run_all, then validate/finalize, then submit. The final candidate "
                "must be available after the clean run in model or the model_var you name. "
                "Use model_var='auto' if you want the environment to discover a candidate.\n\n"
                "Helpful tools: inspect_data, profile_data, list_candidates, check_candidate, "
                "quick_validate, cleanlab_diagnose, tune_hyperparameters, finalize.\n\n"
                "Return exactly one JSON action per turn."
            )
        }

    def _best_validation_metric(self) -> float | None:
        metrics = [
            record.validation_metric
            for record in self.candidates.all()
            if record.validation_metric is not None
        ]
        if not metrics:
            return None
        return max(metrics)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"


def _prediction_length(preds: Any) -> int | None:
    try:
        return len(preds)
    except TypeError:
        return None


def _predictions_nan_free(preds: Any) -> bool:
    try:
        values = np.asarray(preds, dtype=object).reshape(-1)
        if len(values) == 0:
            return False
        return not bool(pd.isna(pd.Series(values)).any())
    except Exception:
        return False


def _validation_scoring_failed(diagnostic: CandidateDiagnostic) -> bool:
    return bool(
        diagnostic.raw_val_predict_ok
        and diagnostic.prediction_length_ok
        and diagnostic.prediction_nan_free
        and diagnostic.serializable is not False
        and diagnostic.validation_metric is None
        and diagnostic.error_type
    )


def _sample_search_params(search_space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("type", "choice")).lower()
        if kind == "int":
            low = int(spec.get("low", 0))
            high = int(spec.get("high", low))
            params[name] = rng.randint(low, high)
        elif kind == "float":
            low = float(spec.get("low", 0.0))
            high = float(spec.get("high", low))
            if bool(spec.get("log")) and low > 0 and high > 0:
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                params[name] = rng.uniform(low, high)
        elif kind in {"categorical", "choice"}:
            choices = list(spec.get("choices") or [])
            if choices:
                params[name] = rng.choice(choices)
    return params


def _format_compact(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return _clip(text, limit)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return repr(value)
