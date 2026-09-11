from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from research.budget import BudgetExhausted
from research.submission import (
    HiddenEvaluationGate,
    SubmissionValidation,
    validate_submission_candidate,
)

from .candidate_store import CandidateBundleError, CandidateBundleStore
from .candidates import CandidateRecord, CandidateRegistry
from .finalization import (
    ProtectedFinalizationArtifact,
    ProtectedFinalizationError,
    ProtectedFinalizationStore,
)
from .scoring import score_with_coercion


TERMINAL_CONTRACT_VERSION = "symmetric-terminal-v1"


@dataclass(frozen=True)
class TerminalContractResult:
    """One terminal outcome produced by the common host-owned controller."""

    final_status: str
    candidate: CandidateRecord | None = None
    validation_metric: float | None = None
    hidden_metric: float | None = None
    artifact: ProtectedFinalizationArtifact | None = None
    failed_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = field(default=None, repr=False, compare=False)
    validation: SubmissionValidation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def succeeded(self) -> bool:
        return self.final_status == "submitted_protected_replay"


class SymmetricTerminalController:
    """Select, replay, validate, freeze, and evaluate one incumbent.

    Candidate generation is deliberately outside this class. Both experimental
    arms provide a clean producing-revision replayer, while this controller owns
    the terminal selection and every gate after it.
    """

    def __init__(
        self,
        *,
        registry: CandidateRegistry,
        candidate_store: CandidateBundleStore,
        finalization_store: ProtectedFinalizationStore,
        hidden_gate: HiddenEvaluationGate,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
        hidden_features: pd.DataFrame,
        hidden_target: pd.Series,
        metric_fn: Callable,
        protocol_version: str,
        prediction_backend: str | None = None,
        metric_normalizer: Callable[[float], float] | None = None,
    ) -> None:
        if not protocol_version:
            raise ValueError("protocol_version must not be empty")
        self.registry = registry
        self.candidate_store = candidate_store
        self.finalization_store = finalization_store
        self.hidden_gate = hidden_gate
        self.validation_features = validation_features
        self.validation_target = validation_target
        self.hidden_features = hidden_features
        self.hidden_target = hidden_target
        self.metric_fn = metric_fn
        self.protocol_version = protocol_version
        self.prediction_backend = prediction_backend
        self.metric_normalizer = metric_normalizer or float

    def finalize(
        self,
        *,
        replay_candidate: Callable[[CandidateRecord, Path], Any],
        before_preflight: Callable[[], None] | None = None,
        before_artifact_write: Callable[[], None] | None = None,
        before_hidden_evaluation: Callable[[], None] | None = None,
        resource_snapshot: Callable[[], dict[str, Any]] | None = None,
        on_candidate_selected: Callable[[CandidateRecord], None] | None = None,
        on_artifact_written: Callable[[ProtectedFinalizationArtifact], None]
        | None = None,
    ) -> TerminalContractResult:
        candidate = self.registry.incumbent()
        if candidate is None:
            return TerminalContractResult(
                final_status="no_incumbent_for_finalization",
                failed_stage="selection",
                error_type="CandidateMissing",
                error_message="No validated incumbent is available.",
            )
        if on_candidate_selected is not None:
            on_candidate_selected(candidate)

        try:
            source_bundle = self.candidate_store.verify_record(candidate)
        except CandidateBundleError as exc:
            return _failure(
                "invalid_candidate_artifact",
                "bundle_verification",
                candidate,
                exc,
            )
        if source_bundle.protocol_version != self.protocol_version:
            return TerminalContractResult(
                final_status="invalid_candidate_artifact",
                candidate=candidate,
                failed_stage="bundle_verification",
                error_type="CandidateProtocolMismatch",
                error_message=(
                    "Candidate bundle protocol does not match the active terminal "
                    "protocol."
                ),
            )

        try:
            model = replay_candidate(
                candidate,
                source_bundle.bundle_dir / "solution.ipynb",
            )
        except BudgetExhausted as exc:
            return _reserve_failure(candidate, "clean_replay", exc)
        except Exception as exc:
            return _failure(
                "protected_replay_failed",
                "clean_replay",
                candidate,
                exc,
            )

        try:
            if before_preflight is not None:
                before_preflight()
            validation = validate_submission_candidate(
                model,
                self.validation_features,
                prediction_backend=self.prediction_backend,
            )
        except BudgetExhausted as exc:
            return _reserve_failure(candidate, "validation_preflight", exc)
        except Exception as exc:
            return _failure(
                "protected_replay_validation_failed",
                "validation_preflight",
                candidate,
                exc,
            )
        if not validation.valid:
            return TerminalContractResult(
                final_status="protected_replay_validation_failed",
                candidate=candidate,
                failed_stage="validation_preflight",
                error_type=validation.error_type or "SubmissionValidationError",
                error_message=(
                    validation.error_message or "Candidate is not submit-ready."
                ),
                validation=validation,
            )

        try:
            raw_metric = score_with_coercion(
                self.metric_fn,
                self.validation_target,
                validation.predictions,
            )
            replay_metric = float(self.metric_normalizer(float(raw_metric)))
            if not math.isfinite(replay_metric):
                raise ValueError("Replayed validation metric is not finite.")
        except Exception as exc:
            return TerminalContractResult(
                final_status="protected_replay_validation_failed",
                candidate=candidate,
                failed_stage="validation_metric",
                error_type=type(exc).__name__,
                error_message=str(exc),
                validation=validation,
            )

        recorded_metric = float(candidate.validation_metric)
        if abs(replay_metric - recorded_metric) > self.registry.score_tolerance:
            return TerminalContractResult(
                final_status="protected_replay_validation_mismatch",
                candidate=candidate,
                validation_metric=replay_metric,
                failed_stage="validation_match",
                error_type="ValidationMetricMismatch",
                error_message=(
                    f"recorded={recorded_metric!r}, replayed={replay_metric!r}"
                ),
                validation=validation,
            )

        try:
            if before_artifact_write is not None:
                before_artifact_write()
            artifact = self.finalization_store.write_artifact(
                candidate_id=candidate.candidate_id,
                model=model,
                protocol_version=self.protocol_version,
                source_candidate_hashes=source_bundle.hashes,
                resource_snapshot=(
                    resource_snapshot() if resource_snapshot is not None else {}
                ),
            )
            replayed_model = self.finalization_store.load_model(
                candidate.candidate_id
            )
            if on_artifact_written is not None:
                on_artifact_written(artifact)
        except BudgetExhausted as exc:
            return _reserve_failure(candidate, "artifact_write", exc)
        except ProtectedFinalizationError as exc:
            return _failure(
                "protected_finalization_artifact_failed",
                "artifact_write",
                candidate,
                exc,
                validation=validation,
                validation_metric=replay_metric,
            )
        except Exception as exc:
            return _failure(
                "protected_finalization_artifact_failed",
                "artifact_write",
                candidate,
                exc,
                validation=validation,
                validation_metric=replay_metric,
            )

        try:
            if before_hidden_evaluation is not None:
                before_hidden_evaluation()
            hidden_metric = self.hidden_gate.evaluate(
                replayed_model,
                self.hidden_features,
                self.hidden_target,
                self.metric_fn,
            )
        except BudgetExhausted as exc:
            return _reserve_failure(candidate, "hidden_evaluation", exc)
        except Exception as exc:
            return _failure(
                "hidden_submit_failed",
                "hidden_evaluation",
                candidate,
                exc,
                validation=validation,
                validation_metric=replay_metric,
                artifact=artifact,
            )

        self.registry.mark_submitted(candidate.candidate_id)
        return TerminalContractResult(
            final_status="submitted_protected_replay",
            candidate=candidate,
            validation_metric=replay_metric,
            hidden_metric=float(hidden_metric),
            artifact=artifact,
            validation=validation,
        )


def _reserve_failure(
    candidate: CandidateRecord,
    stage: str,
    error: BudgetExhausted,
) -> TerminalContractResult:
    return _failure(
        "finalization_reserve_exhausted",
        stage,
        candidate,
        error,
    )


def _failure(
    status: str,
    stage: str,
    candidate: CandidateRecord,
    error: BaseException,
    *,
    validation: SubmissionValidation | None = None,
    validation_metric: float | None = None,
    artifact: ProtectedFinalizationArtifact | None = None,
) -> TerminalContractResult:
    return TerminalContractResult(
        final_status=status,
        candidate=candidate,
        validation_metric=validation_metric,
        artifact=artifact,
        failed_stage=stage,
        error_type=type(error).__name__,
        error_message=str(error),
        validation=validation,
    )
