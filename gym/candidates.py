from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class CandidateRecord:
    """Immutable evidence that one candidate passed the validation contract."""

    candidate_id: str
    model_var: str
    notebook_revision: int
    clean_run_id: str
    metric_name: str = "score"
    metric_direction: MetricDirection = "higher"
    validation_metric: float | None = None
    validation_success: bool = False
    raw_inference_ready: bool = False
    created_step: int = 0
    registration_index: int = 0
    artifact_path: str | None = None

    def to_dict(self, *, submitted: bool = False) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "model_var": self.model_var,
            "notebook_revision": self.notebook_revision,
            "clean_run_id": self.clean_run_id,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "validation_metric": self.validation_metric,
            "validation_success": self.validation_success,
            "raw_inference_ready": self.raw_inference_ready,
            "created_step": self.created_step,
            "registration_index": self.registration_index,
            "submitted": submitted,
            "artifact_path": self.artifact_path,
        }


class CandidateRegistry:
    """Host-owned validated-candidate history with deterministic selection.

    ``latest()`` deliberately retains its historical meaning for callers: it
    returns only the candidate eligible for the current clean run. Notebook
    mutations invalidate that pointer without deleting historical records or
    the incumbent.
    """

    def __init__(
        self,
        *,
        metric_name: str = "score",
        metric_direction: MetricDirection = "higher",
        score_tolerance: float = 1e-12,
    ) -> None:
        if metric_direction not in {"higher", "lower"}:
            raise ValueError("metric_direction must be 'higher' or 'lower'")
        try:
            normalized_tolerance = float(score_tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("score_tolerance must be finite and non-negative") from exc
        if not math.isfinite(normalized_tolerance) or normalized_tolerance < 0:
            raise ValueError("score_tolerance must be finite and non-negative")
        if not metric_name:
            raise ValueError("metric_name must not be empty")

        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.score_tolerance = normalized_tolerance
        self._records: dict[str, CandidateRecord] = {}
        self._latest_id: str | None = None
        self._current_id: str | None = None
        self._incumbent_id: str | None = None
        self._submitted_ids: set[str] = set()

    def add(self, record: CandidateRecord) -> CandidateRecord:
        self._validate_record(record)
        if record.candidate_id in self._records:
            raise ValueError(f"Duplicate candidate_id: {record.candidate_id}")

        incumbent = self.incumbent()
        self._records[record.candidate_id] = record
        self._latest_id = record.candidate_id
        self._current_id = record.candidate_id
        if incumbent is None or self.is_better_score(
            record.validation_metric,
            incumbent.validation_metric,
        ):
            self._incumbent_id = record.candidate_id
        return record

    def latest(self) -> CandidateRecord | None:
        """Return the candidate eligible for the current clean run."""

        if self._current_id is None:
            return None
        return self._records.get(self._current_id)

    def latest_registered(self) -> CandidateRecord | None:
        if self._latest_id is None:
            return None
        return self._records.get(self._latest_id)

    def incumbent(self) -> CandidateRecord | None:
        if self._incumbent_id is None:
            return None
        return self._records.get(self._incumbent_id)

    def invalidate_current(self) -> CandidateRecord | None:
        current = self.latest()
        self._current_id = None
        return current

    def is_better_score(
        self,
        candidate_score: float | None,
        incumbent_score: float | None,
    ) -> bool:
        if not _finite_metric(candidate_score) or not _finite_metric(incumbent_score):
            raise ValueError("candidate and incumbent scores must be finite")
        if self.metric_direction == "higher":
            improvement = float(candidate_score) - float(incumbent_score)
        else:
            improvement = float(incumbent_score) - float(candidate_score)
        return improvement > self.score_tolerance

    def mark_submitted(self, candidate_id: str) -> None:
        if candidate_id not in self._records:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        self._submitted_ids.add(candidate_id)

    def is_submitted(self, candidate_id: str) -> bool:
        return candidate_id in self._submitted_ids

    def clear(self) -> None:
        self._records.clear()
        self._latest_id = None
        self._current_id = None
        self._incumbent_id = None
        self._submitted_ids.clear()

    def all(self) -> list[CandidateRecord]:
        return list(self._records.values())

    def to_list(self) -> list[dict]:
        return [
            record.to_dict(submitted=self.is_submitted(record.candidate_id))
            for record in self.all()
        ]

    def _validate_record(self, record: CandidateRecord) -> None:
        if not record.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if record.metric_name != self.metric_name:
            raise ValueError(
                f"Candidate metric {record.metric_name!r} does not match "
                f"registry metric {self.metric_name!r}"
            )
        if record.metric_direction != self.metric_direction:
            raise ValueError(
                f"Candidate direction {record.metric_direction!r} does not match "
                f"registry direction {self.metric_direction!r}"
            )
        if not record.validation_success or not record.raw_inference_ready:
            raise ValueError("Registry accepts only validated raw-inference-ready candidates")
        if not _finite_metric(record.validation_metric):
            raise ValueError("validation_metric must be finite")
        if type(record.created_step) is not int or record.created_step < 0:
            raise ValueError("created_step must be non-negative")
        if type(record.registration_index) is not int or record.registration_index < 0:
            raise ValueError("registration_index must be non-negative")


def _finite_metric(value: float | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
