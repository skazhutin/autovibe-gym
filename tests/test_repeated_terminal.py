from __future__ import annotations

import pandas as pd
import pytest

from experiments.repeated_terminal import (
    RepeatedSingleShotTerminalAdapter,
    RepeatedTerminalCandidateError,
)


class _ConstantModel:
    def __init__(self, value: int):
        self.value = value

    def predict(self, rows):
        return [self.value] * len(rows)


def _accuracy(y_true, y_pred):
    return float((pd.Series(y_true).reset_index(drop=True) == pd.Series(y_pred)).mean())


def _adapter(tmp_path):
    return RepeatedSingleShotTerminalAdapter(
        private_dir=tmp_path,
        validation_features=pd.DataFrame({"x": [0, 1]}),
        validation_target=pd.Series([1, 1]),
        hidden_features=pd.DataFrame({"x": [2, 3]}),
        hidden_target=pd.Series([1, 1]),
        metric_fn=_accuracy,
        metric_name="accuracy",
        metric_direction="higher",
        score_tolerance=1e-12,
        prediction_backend="process",
    )


def _execute(source, namespace):
    try:
        exec(source, namespace)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}", namespace
    return "", "", namespace


def _namespace():
    return {"_ConstantModel": _ConstantModel}


def test_repeated_adapter_retains_earlier_incumbent_and_replays_its_exact_snapshot(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    incumbent = adapter.validate_and_register(
        model=_ConstantModel(1),
        source_code="model = _ConstantModel(1)",
        attempt_index=0,
    )
    later = adapter.validate_and_register(
        model=_ConstantModel(0),
        source_code="model = _ConstantModel(0)",
        attempt_index=1,
    )
    trace = []

    result = adapter.finalize(
        execute_code=_execute,
        namespace_factory=_namespace,
        before_replay_execution=lambda: trace.append("replay"),
        before_preflight=lambda: trace.append("preflight"),
        before_artifact_write=lambda: trace.append("artifact"),
        before_hidden_evaluation=lambda: trace.append("hidden"),
    )

    assert result.succeeded
    assert result.candidate == incumbent
    assert result.candidate != later
    assert result.hidden_metric == 1.0
    assert trace == ["replay", "preflight", "artifact", "hidden"]
    assert adapter.registry.is_submitted(incumbent.candidate_id)
    snapshot = adapter.candidate_store.notebook_snapshot_path(incumbent)
    assert "_ConstantModel(1)" in snapshot.read_text(encoding="utf-8")


def test_repeated_adapter_rejects_non_ready_candidate_before_registration(tmp_path):
    adapter = _adapter(tmp_path)

    with pytest.raises(RepeatedTerminalCandidateError, match="ModelInterfaceError"):
        adapter.validate_and_register(
            model=object(),
            source_code="model = object()",
            attempt_index=0,
        )

    assert adapter.registry.all() == []
    assert not adapter.candidate_store.root.exists()


def test_repeated_adapter_has_no_live_model_fallback_when_snapshot_replay_fails(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    candidate = adapter.validate_and_register(
        model=_ConstantModel(1),
        source_code="raise RuntimeError('producing revision failed')",
        attempt_index=0,
    )

    result = adapter.finalize(
        execute_code=_execute,
        namespace_factory=_namespace,
    )

    assert result.final_status == "protected_replay_failed"
    assert result.failed_stage == "clean_replay"
    assert adapter.hidden_gate.attempted is False
    assert not adapter.registry.is_submitted(candidate.candidate_id)
    assert not adapter.finalization_store.root.exists()


def test_repeated_adapter_requires_replayed_named_model_even_when_live_bundle_has_one(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    candidate = adapter.validate_and_register(
        model=_ConstantModel(1),
        source_code="other_model = _ConstantModel(1)",
        attempt_index=0,
    )

    result = adapter.finalize(
        execute_code=_execute,
        namespace_factory=_namespace,
    )

    assert result.final_status == "protected_replay_failed"
    assert "did not recreate" in (result.error_message or "")
    assert adapter.hidden_gate.attempted is False
    assert not adapter.registry.is_submitted(candidate.candidate_id)
