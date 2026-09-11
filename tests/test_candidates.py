from dataclasses import FrozenInstanceError

import pytest

from gym.candidates import CandidateRecord, CandidateRegistry


def _record(
    candidate_id: str,
    score: float,
    *,
    direction: str = "higher",
    step: int = 0,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        model_var=f"model_{candidate_id}",
        notebook_revision=step + 1,
        clean_run_id=f"clean_{candidate_id}",
        metric_name="accuracy",
        metric_direction=direction,
        validation_metric=score,
        validation_success=True,
        raw_inference_ready=True,
        created_step=step,
        artifact_path=f"private/{candidate_id}.pkl",
    )


def test_higher_is_better_registry_keeps_worse_and_tied_candidates_out_of_incumbent():
    registry = CandidateRegistry(
        metric_name="accuracy",
        metric_direction="higher",
        score_tolerance=1e-6,
    )

    registry.add(_record("first", 0.7, step=1))
    registry.add(_record("worse", 0.6, step=2))
    registry.add(_record("tie", 0.7000005, step=3))

    assert registry.incumbent().candidate_id == "first"
    assert registry.latest().candidate_id == "tie"
    assert registry.latest_registered().candidate_id == "tie"
    assert [record.candidate_id for record in registry.all()] == [
        "first",
        "worse",
        "tie",
    ]


@pytest.mark.parametrize(
    ("direction", "initial", "better", "expected"),
    [
        ("higher", 0.5, 0.8, "better"),
        ("lower", 0.5, 0.2, "better"),
    ],
)
def test_registry_respects_explicit_metric_direction(
    direction,
    initial,
    better,
    expected,
):
    registry = CandidateRegistry(
        metric_name="metric",
        metric_direction=direction,
    )
    registry.add(
        CandidateRecord(
            candidate_id="initial",
            model_var="model_initial",
            notebook_revision=1,
            clean_run_id="clean_initial",
            metric_name="metric",
            metric_direction=direction,
            validation_metric=initial,
            validation_success=True,
            raw_inference_ready=True,
        )
    )
    registry.add(
        CandidateRecord(
            candidate_id="better",
            model_var="model_better",
            notebook_revision=2,
            clean_run_id="clean_better",
            metric_name="metric",
            metric_direction=direction,
            validation_metric=better,
            validation_success=True,
            raw_inference_ready=True,
        )
    )

    assert registry.incumbent().candidate_id == expected


def test_invalidating_current_preserves_history_and_incumbent():
    registry = CandidateRegistry(metric_name="accuracy")
    record = registry.add(_record("stable", 0.75))

    invalidated = registry.invalidate_current()

    assert invalidated == record
    assert registry.latest() is None
    assert registry.latest_registered() == record
    assert registry.incumbent() == record
    assert registry.all() == [record]


def test_candidate_record_is_immutable_and_submission_state_is_external():
    registry = CandidateRegistry(metric_name="accuracy")
    record = registry.add(_record("stable", 0.75))

    with pytest.raises(FrozenInstanceError):
        record.validation_metric = 1.0

    registry.mark_submitted(record.candidate_id)

    assert registry.is_submitted(record.candidate_id)
    assert registry.to_list()[0]["submitted"] is True
    assert record.to_dict()["submitted"] is False


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_registry_rejects_non_finite_validation_metrics(score):
    registry = CandidateRegistry(metric_name="accuracy")

    with pytest.raises(ValueError, match="validation_metric must be finite"):
        registry.add(_record("invalid", score))


def test_registry_rejects_duplicate_ids_and_metric_contract_drift():
    registry = CandidateRegistry(metric_name="accuracy")
    registry.add(_record("same", 0.5))

    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        registry.add(_record("same", 0.6))

    with pytest.raises(ValueError, match="does not match registry metric"):
        registry.add(
            CandidateRecord(
                candidate_id="wrong_metric",
                model_var="model",
                notebook_revision=2,
                clean_run_id="clean",
                metric_name="f1",
                validation_metric=0.8,
                validation_success=True,
                raw_inference_ready=True,
            )
        )
