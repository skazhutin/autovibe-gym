from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from gym.candidate_store import CandidateBundleStore
from gym.candidates import CandidateRecord, CandidateRegistry, MetricDirection
from gym.finalization import ProtectedFinalizationStore
from gym.notebook import NotebookDocument
from gym.scoring import score_with_coercion
from gym.terminal_contract import SymmetricTerminalController, TerminalContractResult
from research.submission import HiddenEvaluationGate, validate_submission_candidate


REPEATED_SINGLE_SHOT_PROTOCOL_VERSION = "repeated-single-shot-producing-revision-v1"


class RepeatedTerminalCandidateError(RuntimeError):
    pass


class RepeatedSingleShotTerminalAdapter:
    """M5 adapter from independent attempts to the common terminal contract.

    The runner remains responsible for generating and initially executing each
    independent attempt. This adapter admits only validation-ready models,
    stores each attempt as an immutable one-cell notebook revision, and later
    clean-replays the host-selected incumbent through ``SymmetricTerminalController``.
    """

    def __init__(
        self,
        *,
        private_dir: str | Path,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        hidden_features: pd.DataFrame,
        hidden_target: pd.Series,
        metric_fn: Callable,
        metric_name: str,
        metric_direction: MetricDirection,
        score_tolerance: float,
        prediction_backend: str | None,
        metric_normalizer: Callable[[float], float] | None = None,
        hidden_gate: HiddenEvaluationGate | None = None,
    ) -> None:
        self.private_dir = Path(private_dir).resolve()
        self.snapshots_dir = self.private_dir / "attempt_snapshots"
        self.candidate_store = CandidateBundleStore(
            self.private_dir / "candidates"
        )
        self.finalization_store = ProtectedFinalizationStore(
            self.private_dir / "finalization"
        )
        self.registry = CandidateRegistry(
            metric_name=metric_name,
            metric_direction=metric_direction,
            score_tolerance=score_tolerance,
        )
        self.validation_features = validation_features
        self.validation_target = validation_target
        self.hidden_features = hidden_features
        self.hidden_target = hidden_target
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.prediction_backend = prediction_backend
        self.metric_normalizer = metric_normalizer or float
        self.hidden_gate = hidden_gate or HiddenEvaluationGate(
            prediction_backend=prediction_backend
        )

    def validate_and_register(
        self,
        *,
        model: Any,
        source_code: str,
        attempt_index: int,
        model_var: str = "model",
        created_step: int | None = None,
        before_validation: Callable[[], None] | None = None,
        resource_snapshot: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        if type(attempt_index) is not int or attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if not isinstance(source_code, str) or not source_code.strip():
            raise RepeatedTerminalCandidateError("Attempt source code is empty")
        if before_validation is not None:
            before_validation()
        validation = validate_submission_candidate(
            model,
            self.validation_features,
            prediction_backend=self.prediction_backend,
        )
        if not validation.valid:
            raise RepeatedTerminalCandidateError(
                f"{validation.error_type or 'SubmissionValidationError'}: "
                f"{validation.error_message or 'candidate is not submit-ready'}"
            )
        try:
            raw_metric = score_with_coercion(
                self.metric_fn,
                self.validation_target,
                validation.predictions,
            )
            validation_metric = float(self.metric_normalizer(float(raw_metric)))
        except Exception as exc:
            raise RepeatedTerminalCandidateError(
                f"Validation scoring failed: {exc}"
            ) from exc
        if not math.isfinite(validation_metric):
            raise RepeatedTerminalCandidateError(
                "Validation scoring returned a non-finite value"
            )

        candidate_id = f"rss-{attempt_index:04d}-{uuid.uuid4().hex}"
        notebook = NotebookDocument.create(
            self.snapshots_dir / f"{candidate_id}.ipynb"
        )
        notebook.add_code_cell(
            source_code,
            metadata={
                "autovibe": {
                    "arm": "repeated_single_shot",
                    "attempt_index": attempt_index,
                }
            },
        )
        record = CandidateRecord(
            candidate_id=candidate_id,
            model_var=model_var,
            notebook_revision=notebook.revision,
            clean_run_id=f"repeated-attempt-{attempt_index}",
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            validation_metric=validation_metric,
            validation_success=True,
            raw_inference_ready=True,
            created_step=attempt_index if created_step is None else created_step,
            registration_index=len(self.registry.all()),
        )
        bundle = self.candidate_store.write_bundle(
            record,
            model,
            notebook_path=notebook.path,
            protocol_version=REPEATED_SINGLE_SHOT_PROTOCOL_VERSION,
            resource_snapshot={
                "arm": "repeated_single_shot",
                "attempt_index": attempt_index,
                **dict(resource_snapshot or {}),
            },
        )
        return self.registry.add(bundle.record)

    def finalize(
        self,
        *,
        execute_code: Callable[[str, dict[str, Any]], tuple[str, str, dict[str, Any]]],
        namespace_factory: Callable[[], dict[str, Any]],
        before_replay_execution: Callable[[], None] | None = None,
        before_preflight: Callable[[], None] | None = None,
        before_artifact_write: Callable[[], None] | None = None,
        before_hidden_evaluation: Callable[[], None] | None = None,
        resource_snapshot: Callable[[], dict[str, Any]] | None = None,
    ) -> TerminalContractResult:
        def replay(candidate: CandidateRecord, snapshot_path: Path) -> Any:
            notebook = NotebookDocument.load(snapshot_path)
            code_cells = [
                str(cell.source)
                for cell in notebook.notebook.cells
                if cell.cell_type == "code"
            ]
            if len(code_cells) != 1:
                raise RepeatedTerminalCandidateError(
                    "Repeated single-shot snapshot must contain exactly one code cell"
                )
            if before_replay_execution is not None:
                before_replay_execution()
            _stdout, stderr, namespace = execute_code(
                code_cells[0],
                namespace_factory(),
            )
            if stderr.strip():
                raise RepeatedTerminalCandidateError(
                    f"Producing revision replay failed: {stderr.strip()}"
                )
            model = namespace.get(candidate.model_var)
            if model is None:
                raise RepeatedTerminalCandidateError(
                    f"Producing revision did not recreate {candidate.model_var!r}"
                )
            return model

        controller = SymmetricTerminalController(
            registry=self.registry,
            candidate_store=self.candidate_store,
            finalization_store=self.finalization_store,
            hidden_gate=self.hidden_gate,
            validation_features=self.validation_features,
            validation_target=self.validation_target,
            hidden_features=self.hidden_features,
            hidden_target=self.hidden_target,
            metric_fn=self.metric_fn,
            protocol_version=REPEATED_SINGLE_SHOT_PROTOCOL_VERSION,
            prediction_backend=self.prediction_backend,
            metric_normalizer=self.metric_normalizer,
        )
        return controller.finalize(
            replay_candidate=replay,
            before_preflight=before_preflight,
            before_artifact_write=before_artifact_write,
            before_hidden_evaluation=before_hidden_evaluation,
            resource_snapshot=resource_snapshot,
        )
