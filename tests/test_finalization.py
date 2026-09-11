import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

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


def test_direct_verification_rejects_linked_finalization_root(tmp_path, monkeypatch):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    _write(store)
    original_is_symlink = Path.is_symlink

    def report_store_root_as_symlink(path):
        if path == store.root:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_store_root_as_symlink)
    with pytest.raises(
        ProtectedFinalizationError,
        match="root is not a regular directory",
    ):
        store.verify_artifact("candidate-1")


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


def test_only_one_concurrent_writer_can_publish_protected_artifact(tmp_path):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    workers = 8
    start = Barrier(workers)

    def publish():
        start.wait()
        try:
            return _write(store)
        except ProtectedFinalizationError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(lambda _index: publish(), range(workers)))

    artifacts = [
        outcome for outcome in outcomes if not isinstance(outcome, Exception)
    ]
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(artifacts) == 1
    assert len(failures) == workers - 1
    assert all("already exists" in str(failure) for failure in failures)
    assert store.verify_artifact("candidate-1") == artifacts[0]


def test_failed_protected_publish_lock_leaves_no_stale_lock(tmp_path, monkeypatch):
    store = ProtectedFinalizationStore(tmp_path / "finalization")

    def fail_fsync(_descriptor):
        raise OSError("simulated lock durability failure")

    monkeypatch.setattr("gym.finalization.os.fsync", fail_fsync)
    with pytest.raises(ProtectedFinalizationError, match="publication lock"):
        _write(store)

    assert list(store.root.iterdir()) == []


def test_successful_protected_publish_survives_lock_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    store = ProtectedFinalizationStore(tmp_path / "finalization")
    lock_path = store.root / ".tmp-lock-candidate-1"
    original_unlink = Path.unlink

    def fail_lock_cleanup(path, *args, **kwargs):
        if path == lock_path:
            raise OSError("simulated post-publish lock cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_lock_cleanup)
    artifact = _write(store)

    assert artifact.artifact_dir.is_dir()
    assert lock_path.is_file()
    assert store.verify_artifact("candidate-1") == artifact


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
