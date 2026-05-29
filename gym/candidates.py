from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateRecord:
    candidate_id: str
    model_var: str
    notebook_revision: int
    clean_run_id: str
    validation_metric: float | None = None
    validation_success: bool = False
    raw_inference_ready: bool = False
    submitted: bool = False
    artifact_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "model_var": self.model_var,
            "notebook_revision": self.notebook_revision,
            "clean_run_id": self.clean_run_id,
            "validation_metric": self.validation_metric,
            "validation_success": self.validation_success,
            "raw_inference_ready": self.raw_inference_ready,
            "submitted": self.submitted,
            "artifact_path": self.artifact_path,
        }


class CandidateRegistry:
    def __init__(self) -> None:
        self._records: dict[str, CandidateRecord] = {}
        self._latest_id: str | None = None

    def add(self, record: CandidateRecord) -> CandidateRecord:
        self._records[record.candidate_id] = record
        self._latest_id = record.candidate_id
        return record

    def latest(self) -> CandidateRecord | None:
        if self._latest_id is None:
            return None
        return self._records.get(self._latest_id)

    def clear(self) -> None:
        self._records.clear()
        self._latest_id = None

    def all(self) -> list[CandidateRecord]:
        return list(self._records.values())

    def to_list(self) -> list[dict]:
        return [record.to_dict() for record in self.all()]
