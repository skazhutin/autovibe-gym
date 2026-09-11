from __future__ import annotations

import pandas as pd

from gym.candidate_store import CandidateBundleStore
from gym.candidates import CandidateRecord, CandidateRegistry
from gym.finalization import ProtectedFinalizationStore
from gym.notebook import NotebookDocument
from gym.terminal_contract import SymmetricTerminalController
from research.submission import HiddenEvaluationGate


class _ConstantModel:
    def __init__(self, value: int):
        self.value = value

    def predict(self, rows):
        return [self.value] * len(rows)


def _accuracy(y_true, y_pred):
    return float((pd.Series(y_true).reset_index(drop=True) == pd.Series(y_pred)).mean())


def _register(
    root,
    registry,
    store,
    *,
    candidate_id,
    score,
    value=1,
    registration_index=0,
):
    notebook = NotebookDocument.create(root / f"{candidate_id}.ipynb")
    notebook.add_code_cell(f"model = ConstantModel({value})")
    record = CandidateRecord(
        candidate_id=candidate_id,
        model_var="model",
        notebook_revision=notebook.revision,
        clean_run_id=f"clean-{candidate_id}",
        metric_name="accuracy",
        validation_metric=score,
        validation_success=True,
        raw_inference_ready=True,
        registration_index=registration_index,
    )
    bundle = store.write_bundle(
        record,
        _ConstantModel(value),
        notebook_path=notebook.path,
        protocol_version="fixture-v1",
    )
    registry.add(bundle.record)
    return bundle.record


def _controller(root, registry, store, *, hidden_gate=None):
    return SymmetricTerminalController(
        registry=registry,
        candidate_store=store,
        finalization_store=ProtectedFinalizationStore(root / "finalization"),
        hidden_gate=hidden_gate or HiddenEvaluationGate(prediction_backend="process"),
        validation_features=pd.DataFrame({"x": [0, 1]}),
        validation_target=pd.Series([1, 1]),
        hidden_features=pd.DataFrame({"x": [2, 3]}),
        hidden_target=pd.Series([1, 1]),
        metric_fn=_accuracy,
        protocol_version="fixture-v1",
        prediction_backend="process",
    )


def test_common_terminal_contract_selects_incumbent_replays_and_opens_one_hidden_gate(
    tmp_path,
):
    registry = CandidateRegistry(metric_name="accuracy")
    store = CandidateBundleStore(tmp_path / "candidates")
    incumbent = _register(
        tmp_path,
        registry,
        store,
        candidate_id="incumbent",
        score=1.0,
        registration_index=0,
    )
    _register(
        tmp_path,
        registry,
        store,
        candidate_id="later-worse",
        score=0.0,
        value=0,
        registration_index=1,
    )
    order = []
    replay_paths = []
    controller = _controller(tmp_path, registry, store)

    result = controller.finalize(
        replay_candidate=lambda record, path: (
            order.append("replay"),
            replay_paths.append(path),
            _ConstantModel(1),
        )[-1],
        before_preflight=lambda: order.append("preflight"),
        before_artifact_write=lambda: order.append("artifact"),
        before_hidden_evaluation=lambda: order.append("hidden"),
    )

    assert result.succeeded
    assert result.candidate == incumbent
    assert result.validation_metric == 1.0
    assert result.hidden_metric == 1.0
    assert order == ["replay", "preflight", "artifact", "hidden"]
    assert replay_paths == [store.notebook_snapshot_path(incumbent)]
    assert registry.is_submitted(incumbent.candidate_id)
    assert controller.hidden_gate.attempted is True


def test_common_terminal_contract_fails_closed_on_metric_drift_before_artifact_and_hidden(
    tmp_path,
):
    registry = CandidateRegistry(metric_name="accuracy")
    store = CandidateBundleStore(tmp_path / "candidates")
    incumbent = _register(
        tmp_path,
        registry,
        store,
        candidate_id="incumbent",
        score=1.0,
    )
    controller = _controller(tmp_path, registry, store)

    result = controller.finalize(
        replay_candidate=lambda _record, _path: _ConstantModel(0)
    )

    assert result.final_status == "protected_replay_validation_mismatch"
    assert result.failed_stage == "validation_match"
    assert result.validation_metric == 0.0
    assert not registry.is_submitted(incumbent.candidate_id)
    assert controller.hidden_gate.attempted is False
    assert not (tmp_path / "finalization").exists()


def test_common_terminal_contract_rejects_bundle_drift_before_replay(tmp_path):
    registry = CandidateRegistry(metric_name="accuracy")
    store = CandidateBundleStore(tmp_path / "candidates")
    incumbent = _register(
        tmp_path,
        registry,
        store,
        candidate_id="incumbent",
        score=1.0,
    )
    with open(incumbent.artifact_path, "ab") as handle:
        handle.write(b"corruption")
    replayed = []
    controller = _controller(tmp_path, registry, store)

    result = controller.finalize(
        replay_candidate=lambda _record, _path: replayed.append(True)
    )

    assert result.final_status == "invalid_candidate_artifact"
    assert result.failed_stage == "bundle_verification"
    assert replayed == []
    assert controller.hidden_gate.attempted is False


def test_common_terminal_contract_has_identical_sequence_for_two_arm_replayers(tmp_path):
    traces = []
    for arm in ("iterative", "repeated"):
        root = tmp_path / arm
        registry = CandidateRegistry(metric_name="accuracy")
        store = CandidateBundleStore(root / "candidates")
        _register(
            root,
            registry,
            store,
            candidate_id=f"{arm}-candidate",
            score=1.0,
        )
        trace = []
        result = _controller(root, registry, store).finalize(
            replay_candidate=lambda _record, _path, trace=trace: (
                trace.append("replay"),
                _ConstantModel(1),
            )[-1],
            before_preflight=lambda trace=trace: trace.append("preflight"),
            before_artifact_write=lambda trace=trace: trace.append("artifact"),
            before_hidden_evaluation=lambda trace=trace: trace.append("hidden"),
        )
        traces.append((result.final_status, trace))

    assert traces == [
        ("submitted_protected_replay", ["replay", "preflight", "artifact", "hidden"]),
        ("submitted_protected_replay", ["replay", "preflight", "artifact", "hidden"]),
    ]
