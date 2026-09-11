import json

import pytest

from gym.finalization import (
    PROTECTED_FINALIZATION_SCHEMA_VERSION,
    ProtectedFinalizationError,
    ProtectedFinalizationStore,
)


class _ConstantModel:
    def __init__(self, value=1):
        self.value = value

    def predict(self, rows):
        return [self.value] * len(rows)


def _write(store, candidate_id="candidate-1"):
    return store.write_artifact(
        candidate_id=candidate_id,
        model=_ConstantModel(7),
        protocol_version="jupyter-v1",
        source_candidate_hashes={
            "candidate.json": "a" * 64,
            "model.pkl": "b" * 64,
            "solution.ipynb": "c" * 64,
        },
        resource_snapshot={"phase": "finalization", "code_executions": 2},
    )


def test_protected_artifact_round_trip_is_atomic_private_and_hash_verified(tmp_path):
    private_root = tmp_path / "private" / "finalization"
    store = ProtectedFinalizationStore(private_root)

    artifact = _write(store)

    assert {path.name for path in artifact.artifact_dir.iterdir()} == {
        "receipt.json",
        "replayed_model.pkl",
        "SHA256SUMS",
    }
    receipt_text = (artifact.artifact_dir / "receipt.json").read_text(
        encoding="utf-8"
    )
    receipt = json.loads(receipt_text)
    assert receipt["schema_version"] == PROTECTED_FINALIZATION_SCHEMA_VERSION
    assert receipt["resource_snapshot"] == {
        "phase": "finalization",
        "code_executions": 2,
    }
    assert str(private_root) not in receipt_text
    assert str(private_root) not in json.dumps(artifact.public_receipt())
    assert store.load_model("candidate-1").predict([1, 2]) == [7, 7]
    assert store.verify_artifact("candidate-1") == artifact


@pytest.mark.parametrize("filename", ["receipt.json", "replayed_model.pkl"])
def test_protected_artifact_rejects_checksum_drift(tmp_path, filename):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    artifact = _write(store)
    with (artifact.artifact_dir / filename).open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ProtectedFinalizationError, match="checksum mismatch"):
        store.verify_artifact("candidate-1")


def test_protected_artifact_never_overwrites_and_cleans_failed_publish(
    tmp_path, monkeypatch
):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    _write(store)
    with pytest.raises(ProtectedFinalizationError, match="already exists"):
        _write(store)

    failing_store = ProtectedFinalizationStore(tmp_path / "failed")

    def fail_replace(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr("gym.finalization.os.replace", fail_replace)
    with pytest.raises(ProtectedFinalizationError, match="atomically store"):
        _write(failing_store)
    assert list(failing_store.root.iterdir()) == []


def test_protected_artifact_rejects_unsafe_ids_partial_and_symlinked_entries(
    tmp_path,
):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    for unsafe_id in ("../escape", ".", "..", ".tmp-ignored"):
        with pytest.raises(ProtectedFinalizationError, match="storage-safe"):
            _write(store, unsafe_id)

    artifact = _write(store)
    model_path = artifact.model_path
    replacement = tmp_path / "replacement.pkl"
    replacement.write_bytes(model_path.read_bytes())
    model_path.unlink()
    try:
        model_path.symlink_to(replacement)
    except OSError:
        pytest.skip("File symlinks are not available on this platform")
    with pytest.raises(ProtectedFinalizationError, match="non-regular"):
        store.verify_artifact("candidate-1")


def test_protected_artifact_root_symlink_is_rejected_when_supported(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are not available on this platform")

    with pytest.raises(ProtectedFinalizationError, match="not a regular directory"):
        _write(ProtectedFinalizationStore(linked_root))
