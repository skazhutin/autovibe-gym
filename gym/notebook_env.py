from __future__ import annotations

import copy
import json
import os
import pickle
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .candidates import CandidateRecord, CandidateRegistry
from .feedback import FeedbackItem, NotebookChecklist
from .jupyter_kernel import (
    CellExecutionResult,
    KernelExecutionBackend,
    LocalJupyterKernelBackend,
)
from .modes import EpisodeMode, resolve_episode_mode
from .notebook import NotebookDocument
from .protocol import Action, Observation, coerce_action


def _default_kernel_backend() -> KernelExecutionBackend:
    mode = os.getenv("AUTOVIBE_KERNEL_BACKEND", "local").lower()
    if mode == "docker":
        from .jupyter_kernel import ContainerJupyterKernelBackend
        return ContainerJupyterKernelBackend()
    return LocalJupyterKernelBackend()


MODEL_INTERFACE_MESSAGE = (
    "[MODEL CHECK] The selected candidate is not submit-ready because it does "
    "not provide a predict(X) method."
)

MODEL_RAW_INPUT_MESSAGE = (
    "[MODEL CHECK] The saved candidate cannot predict raw validation features. "
    "The submitted artifact must reproduce all required preprocessing when "
    "called on new raw rows. If you selected columns, engineered features, "
    "scaled values, or encoded target labels, wrap that logic inside the "
    "candidate so model.predict(raw_features) returns labels in the original "
    "target format."
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


class NotebookGymEnv:
    """Real Jupyter-backed AutoVibe Gym environment."""

    protocol_version = "jupyter-v1"
    checklist_version = "generic-hidden-v1"
    feedback_policy_version = "selective-v1"

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
    ):
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
        self.checklist = NotebookChecklist(target_col=target_col)
        self.candidates = CandidateRegistry()
        self._candidate_objects: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.feedback_trace: list[dict[str, Any]] = []
        self.validation_trajectory: list[dict[str, Any]] = []
        self.private_summary: dict[str, Any] = {}
        self.dirty_since_clean_run = True
        self.last_clean_run_id: str | None = None
        self.notebook_revision_at_clean_run: int | None = None
        self.last_validated_candidate_id: str | None = None
        self.cell_executions_total = 0
        self.kernel_restarts_total = 0
        self.clean_runs_total = 0
        self.validation_calls_total = 0
        self.contract_feedback_count = 0
        self.model_check_failure_count = 0
        self.errors_count = 0
        self._start_time = time.time()

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
        self.checklist = NotebookChecklist(target_col=self.state.target_col)
        self.candidates = CandidateRegistry()
        self._candidate_objects = {}
        self.events = []
        self.feedback_trace = []
        self.validation_trajectory = []
        self.private_summary = {}
        self.dirty_since_clean_run = True
        self.last_clean_run_id = None
        self.notebook_revision_at_clean_run = None
        self.last_validated_candidate_id = None
        self.cell_executions_total = 0
        self.kernel_restarts_total = 0
        self.clean_runs_total = 0
        self.validation_calls_total = 0
        self.contract_feedback_count = 0
        self.model_check_failure_count = 0
        self.errors_count = 0
        self._start_time = time.time()
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
        parsed = coerce_action(action)
        if self.state.step >= self.state.max_steps and parsed.type != "submit":
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
            parsed = Action.add_cell_action(parsed.code, cell_type="code", execute=True)
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
        raise RuntimeError(f"Unsupported action type: {parsed.type}")

    def budget_remaining(self) -> int:
        return max(self.state.max_steps - self.state.step, 0)

    def _consume_step(self, action: Action) -> None:
        if action.type != "submit":
            self.state.step += 1

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

    def validate_candidate(self, model_var: str = "model") -> Observation:
        self.validation_calls_total += 1
        blocker = self._clean_state_blocker()
        if blocker is not None:
            observation = self._observation(
                action="validate",
                feedback_items=[blocker],
                model_var=model_var,
            )
            return self._record_observation(observation)

        try:
            model = self._load_candidate_from_kernel(model_var)
        except Exception as exc:
            feedback = self._contract_feedback(
                "candidate_missing",
                f"Candidate variable '{model_var}' is not available after the clean run.",
                severity="blocker",
            )
            self._record_event(
                action="validate",
                model_var=model_var,
                error=f"{type(exc).__name__}: {exc}",
            )
            observation = self._observation(
                action="validate",
                feedback_items=[feedback],
                model_var=model_var,
            )
            return self._record_observation(observation)

        if not hasattr(model, "predict"):
            self.model_check_failure_count += 1
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

        X_val = self.state.val.drop(columns=[self.state.target_col])
        y_val = self.state.val[self.state.target_col]
        try:
            preds = model.predict(X_val)
            validation_metric = float(self.metric_fn(y_val, preds))
        except Exception:
            self.model_check_failure_count += 1
            observation = self._observation(
                action="validate",
                feedback_items=[
                    self._contract_feedback(
                        "raw_validation_readiness",
                        MODEL_RAW_INPUT_MESSAGE,
                        severity="blocker",
                    )
                ],
                model_var=model_var,
            )
            return self._record_observation(observation)

        candidate_id = str(uuid.uuid4())
        artifact_path = self.private_dir / "artifacts" / f"{candidate_id}.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as fh:
            pickle.dump(model, fh)
        record = CandidateRecord(
            candidate_id=candidate_id,
            model_var=model_var,
            notebook_revision=self.notebook.revision,
            clean_run_id=self.last_clean_run_id or "",
            validation_metric=validation_metric,
            validation_success=True,
            raw_inference_ready=True,
            artifact_path=str(artifact_path),
        )
        self.candidates.add(record)
        self._candidate_objects[candidate_id] = model
        self.last_validated_candidate_id = candidate_id
        self.validation_trajectory.append(record.to_dict())
        self.checklist.record_structural(
            "baseline_candidate_created",
            reason="candidate exists after clean run",
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
        self._record_event(action="validate", model_var=model_var, candidate=record.to_dict())
        observation = self._observation(
            action="validate",
            stdout=f"validation_{self.state.metric_name}={validation_metric:.6f}",
            model_var=model_var,
            validation_metric=validation_metric,
        )
        return self._record_observation(observation)

    def submit_by_name(self, model_var: str = "model") -> Observation:
        blocker = self._clean_state_blocker()
        if blocker is not None:
            observation = self._observation(
                action="submit",
                feedback_items=[blocker],
                model_var=model_var,
            )
            return self._record_observation(observation)

        candidate = self.candidates.latest()
        if (
            candidate is None
            or not candidate.validation_success
            or candidate.model_var != model_var
            or candidate.notebook_revision != self.notebook.revision
            or candidate.clean_run_id != self.last_clean_run_id
        ):
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
                model = pickle.load(fh)

        try:
            X_test = self.state.test.drop(columns=[self.state.target_col])
            y_test = self.state.test[self.state.target_col]
            preds = model.predict(X_test)
            score = float(self.metric_fn(y_test, preds))
            candidate.submitted = True
            self.state.submitted = True
            self.private_summary["final_test_metric"] = score
            self.private_summary["valid_submit"] = True
            self.private_summary["submit_failure_type"] = None
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

    def get_summary(self) -> dict:
        error_count = sum(1 for observation in self.state.history if observation.stderr.strip())
        submit_failure_type = self.private_summary.get("submit_failure_type")
        summary = {
            "steps_used": self.state.step,
            "checklist_coverage": self.checklist.coverage(),
            "private_checklist_coverage": self.checklist.coverage(),
            "error_count": error_count,
            "errors_count": error_count,
            "submitted": self.state.submitted,
            "valid_submit": bool(self.private_summary.get("valid_submit", False)),
            "test_metric": self.private_summary.get("final_test_metric"),
            "final_test_metric": self.private_summary.get("final_test_metric"),
            "submit_failure_type": submit_failure_type,
            "finalization_status": self._finalization_status(submit_failure_type),
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            "notebook_cells_final": len(self.notebook.notebook.cells),
            "notebook_revisions_total": self.notebook.revision,
            "cell_executions_total": self.cell_executions_total,
            "kernel_restarts_total": self.kernel_restarts_total,
            "clean_runs_total": self.clean_runs_total,
            "successful_clean_run": int(not self.dirty_since_clean_run and self.last_clean_run_id is not None),
            "validation_calls_total": self.validation_calls_total,
            "best_validation_metric": self._best_validation_metric(),
            "contract_feedback_count": self.contract_feedback_count,
            "model_check_failure_count": self.model_check_failure_count,
            "checklist_hints_shown_total": self.checklist.hints_shown_total,
            "protocol_version": self.protocol_version,
            "notebook_backend": "jupyter",
            "checklist_version": self.checklist_version,
            "feedback_policy_version": self.feedback_policy_version,
            "episode_workspace": str(self.workspace_dir),
            "private_episode_dir": str(self.private_dir),
        }
        summary.update(self.private_summary)
        return summary

    def _add_cell(self, action: Action) -> Observation:
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
            if self.mode.checklist_feedback_enabled:
                checklist_items = self.checklist.record_execution(
                    source=source,
                    stdout=stdout,
                    cell_id=cell_id,
                    step=self.state.step,
                    execution_success=result.success,
                    has_runtime_error=not result.success,
                    has_contract_blocker=False,
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
    ) -> Observation:
        feedback_items = feedback_items or []
        stderr = "\n".join(
            item.message
            for item in feedback_items
            if item.visible_to_agent and item.channel == "contract"
        )
        return Observation(
            action=action,
            step=self.state.step,
            budget_remaining=self.budget_remaining(),
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
        )

    def _record_observation(self, observation: Observation) -> Observation:
        self.state.history.append(observation)
        self.feedback_trace.append(self._visible_observation_dict(observation))
        self._save_artifacts()
        return observation

    def _record_event(self, **event: Any) -> None:
        event_record = {
            "event_id": str(uuid.uuid4()),
            "step": self.state.step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notebook_revision": self.notebook.revision,
        }
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
        _write_json(self.private_dir / "notebook_events_private.json", self.events)
        _write_json(
            self.private_dir / "feedback_trace_private.json",
            [observation.to_private_dict() for observation in self.state.history],
        )
        _write_json(
            self.private_dir / "validation_trajectory_private.json",
            self.validation_trajectory,
        )
        _write_json(self.private_dir / "episode_summary.json", self.get_summary())

    def _visible_observation_dict(self, observation: Observation) -> dict[str, Any]:
        data = observation.to_dict()
        data.pop("test_metric", None)
        data.pop("checklist_coverage", None)
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

    def _load_candidate_from_kernel(self, model_var: str) -> Any:
        tmp_dir = self.workspace_dir / ".autovibe_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"candidate_{uuid.uuid4().hex}.pkl"
        self.kernel.dump_variable_to_file(model_var, tmp_path)
        with tmp_path.open("rb") as fh:
            model = pickle.load(fh)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return model

    def candidate_variable_names(self) -> list[str]:
        """Return clean-kernel globals that look like submit candidates.

        This is intentionally based only on public kernel state after a clean
        run. It does not inspect hidden test data or score anything; it just
        helps the host recover when an LLM creates a submit-ready object under
        a name other than the documented `model` / `best_model`.
        """
        source = """
import inspect as _autovibe_inspect
import json as _autovibe_json
import types as _autovibe_types

_autovibe_skip = {
    "train_df", "val_df", "target_col", "pd", "np",
}
_autovibe_names = []
for _autovibe_name, _autovibe_value in list(globals().items()):
    if _autovibe_name.startswith("_") or _autovibe_name in _autovibe_skip:
        continue
    if _autovibe_inspect.isclass(_autovibe_value):
        continue
    if isinstance(_autovibe_value, _autovibe_types.ModuleType):
        continue
    _autovibe_predict = getattr(_autovibe_value, "predict", None)
    if callable(_autovibe_predict):
        _autovibe_names.append(_autovibe_name)
print(_autovibe_json.dumps(sorted(set(_autovibe_names))))
""".strip()
        result = self.kernel.execute_cell(
            source,
            timeout=min(self.kernel_timeout, 10),
            store_history=False,
        )
        if not result.success:
            return []
        try:
            parsed = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(name) for name in parsed if str(name).strip()]

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
        return {
            "task": (
                "You are solving a supervised ML task in a real Jupyter notebook.\n"
                f"Target column: '{self.state.target_col}'\n"
                f"Metric: {self.state.metric_name}\n"
                f"Max turns: {self.state.max_steps}\n\n"
                "A real .ipynb document is the source of truth for your solution. "
                "You may add, update, delete, move, inspect, and execute cells. "
                "The notebook uses a persistent Jupyter kernel, so variables survive "
                "between executed cells.\n\n"
                "Kernel variables already available: train_df, val_df, target_col, pd, np. "
                "You may import installed ML libraries, create artifacts in the episode "
                "workspace, and display plots or DataFrames.\n\n"
                "Hidden test data is not available in the kernel or workspace. "
                "Interactive state is not enough for final acceptance: run "
                "restart_and_run_all, then validate, then submit. The final candidate "
                "must be assigned after the clean run to a top-level variable named "
                "exactly model; if preprocessing or label decoding is needed, wrap it "
                "inside that model object so model.predict(raw_features) works.\n\n"
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

    def _finalization_status(self, submit_failure_type: str | None = None) -> str:
        if self.private_summary.get("valid_submit", False):
            return "valid_submit"
        if self.state.submitted and submit_failure_type:
            return f"hidden_submit_failed:{submit_failure_type}"
        if self.state.submitted:
            return "submitted_without_metric"
        latest = self.candidates.latest()
        if latest is not None and latest.validation_success:
            return "validated_not_submitted"
        if not self.dirty_since_clean_run and self.last_clean_run_id is not None:
            return "clean_run_not_validated"
        return "no_clean_run"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return repr(value)
