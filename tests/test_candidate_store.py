import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from gym.candidate_store import (
    CANDIDATE_BUNDLE_SCHEMA_VERSION,
    CandidateBundleError,
    CandidateBundleStore,
)
from gym.candidates import CandidateRecord


class _ConstantModel:
    def __init__(self, value: int = 1) -> None:
        self.value = value

    def predict(self, rows):
        return [self.value] * len(rows)


def _record(candidate_id: str = "candidate-1", *, index: int = 0) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        model_var="model",
        notebook_revision=index + 1,
        clean_run_id=f"clean-{index}",
        metric_name="accuracy",
        metric_direction="higher",
        validation_metric=0.75 + index / 100,
        validation_success=True,
        raw_inference_ready=True,
        created_step=index + 2,
        registration_index=index,
    )


def _notebook(tmp_path: Path, content: bytes = b'{"cells":[]}\n') -> Path:
    path = tmp_path / "source.ipynb"
    path.write_bytes(content)
    return path


def _refresh_checksums(bundle_dir: Path) -> None:
    names = ("candidate.json", "model.pkl", "solution.ipynb")
    text = "".join(
        f"{hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(names)
    )
    (bundle_dir / "SHA256SUMS").write_text(text, encoding="ascii")


def test_bundle_round_trip_is_immutable_private_and_hash_verified(tmp_path):
    private_root = tmp_path / "private" / "candidates"
    store = CandidateBundleStore(private_root)
    source_notebook = _notebook(tmp_path, b'{"cells":[{"source":"x = 1"}]}\n')

    bundle = store.write_bundle(
        _record(),
        _ConstantModel(7),
        notebook_path=source_notebook,
        protocol_version="jupyter-v1",
        resource_snapshot={"step": 2, "validation_calls": 1},
    )

    assert {path.name for path in bundle.bundle_dir.iterdir()} == {
        "candidate.json",
        "model.pkl",
        "solution.ipynb",
        "SHA256SUMS",
    }
    metadata_text = (bundle.bundle_dir / "candidate.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["schema_version"] == CANDIDATE_BUNDLE_SCHEMA_VERSION
    assert metadata["resource_snapshot"] == {"step": 2, "validation_calls": 1}
    assert str(private_root) not in metadata_text
    assert "artifact_path" not in metadata_text
    assert "submitted" not in metadata_text
    assert bundle.record.artifact_path == str(bundle.bundle_dir / "model.pkl")
    assert store.notebook_snapshot_path(bundle.record).read_bytes() == source_notebook.read_bytes()
    assert store.load_model(bundle.record).predict([1, 2]) == [7, 7]
    assert store.load_records() == [bundle.record]
    assert store.load_records(expected_protocol_version="jupyter-v1") == [
        bundle.record
    ]


def test_restore_rejects_candidate_from_foreign_protocol(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="foreign-protocol-v1",
    )

    with pytest.raises(CandidateBundleError, match="active protocol"):
        store.load_records(expected_protocol_version="jupyter-v1")


def test_direct_verification_rejects_linked_store_root(tmp_path, monkeypatch):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )
    original_is_symlink = Path.is_symlink

    def report_store_root_as_symlink(path):
        if path == store.root:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_store_root_as_symlink)
    with pytest.raises(CandidateBundleError, match="root is not a regular directory"):
        store.verify_record(bundle.record)


@pytest.mark.parametrize("filename", ["candidate.json", "model.pkl", "solution.ipynb"])
def test_bundle_rejects_checksum_drift_for_every_hashed_payload(tmp_path, filename):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )
    with (bundle.bundle_dir / filename).open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(CandidateBundleError, match="checksum mismatch"):
        store.load_records()


def test_bundle_rejects_unsupported_schema_even_with_fresh_checksum(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )
    metadata_path = bundle.bundle_dir / "candidate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "candidate-bundle-v999"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_checksums(bundle.bundle_dir)

    with pytest.raises(CandidateBundleError, match="Unsupported"):
        store.load_records()


def test_bundle_rejects_duplicate_json_keys_even_with_fresh_checksum(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )
    metadata_path = bundle.bundle_dir / "candidate.json"
    raw = metadata_path.read_text(encoding="utf-8")
    raw = raw.replace(
        "{",
        f'{{"schema_version":"{CANDIDATE_BUNDLE_SCHEMA_VERSION}",',
        1,
    )
    metadata_path.write_text(raw, encoding="utf-8")
    _refresh_checksums(bundle.bundle_dir)

    with pytest.raises(CandidateBundleError, match="Duplicate JSON key"):
        store.load_records()


def test_load_ignores_atomic_temp_directories_but_rejects_partial_final_bundle(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    store.root.mkdir(parents=True)
    (store.root / ".tmp-interrupted").mkdir()

    assert store.load_records() == []

    (store.root / "candidate-1").mkdir()
    with pytest.raises(CandidateBundleError, match="missing or unexpected"):
        store.load_records()


def test_failed_atomic_publish_leaves_no_final_or_temp_bundle(tmp_path, monkeypatch):
    store = CandidateBundleStore(tmp_path / "candidates")

    def fail_replace(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr("gym.candidate_store.os.replace", fail_replace)
    with pytest.raises(CandidateBundleError, match="atomically store"):
        store.write_bundle(
            _record(),
            _ConstantModel(),
            notebook_path=_notebook(tmp_path),
            protocol_version="jupyter-v1",
        )

    assert list(store.root.iterdir()) == []


def test_candidate_scoped_publish_lock_prevents_racing_writer(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    store.root.mkdir(parents=True)
    lock_path = store.root / ".tmp-lock-candidate-1"
    lock_path.write_text("competing-writer\n", encoding="ascii")

    with pytest.raises(CandidateBundleError, match="being published"):
        store.write_bundle(
            _record(),
            _ConstantModel(),
            notebook_path=_notebook(tmp_path),
            protocol_version="jupyter-v1",
        )

    assert not (store.root / "candidate-1").exists()
    assert store.load_records() == []


def test_only_one_concurrent_writer_can_publish_candidate_bundle(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    notebook = _notebook(tmp_path)
    workers = 8
    start = Barrier(workers)

    def publish():
        start.wait()
        try:
            return store.write_bundle(
                _record(),
                _ConstantModel(),
                notebook_path=notebook,
                protocol_version="jupyter-v1",
            )
        except CandidateBundleError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(lambda _index: publish(), range(workers)))

    bundles = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(bundles) == 1
    assert len(failures) == workers - 1
    assert all("already exists" in str(failure) for failure in failures)
    assert store.load_records() == [bundles[0].record]


def test_failed_publish_lock_acquisition_leaves_no_stale_lock(tmp_path, monkeypatch):
    store = CandidateBundleStore(tmp_path / "candidates")

    def fail_fsync(_descriptor):
        raise OSError("simulated lock durability failure")

    monkeypatch.setattr("gym.candidate_store.os.fsync", fail_fsync)
    with pytest.raises(CandidateBundleError, match="publication lock"):
        store.write_bundle(
            _record(),
            _ConstantModel(),
            notebook_path=_notebook(tmp_path),
            protocol_version="jupyter-v1",
        )

    assert list(store.root.iterdir()) == []


def test_successful_publish_is_not_reported_failed_when_lock_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    store = CandidateBundleStore(tmp_path / "candidates")
    lock_path = store.root / ".tmp-lock-candidate-1"
    original_unlink = Path.unlink

    def fail_lock_cleanup(path, *args, **kwargs):
        if path == lock_path:
            raise OSError("simulated post-publish lock cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_lock_cleanup)
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )

    assert bundle.bundle_dir.is_dir()
    assert lock_path.is_file()
    assert store.load_records() == [bundle.record]


def test_duplicate_bundle_and_duplicate_registration_index_fail_closed(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    notebook = _notebook(tmp_path)
    store.write_bundle(
        _record("first", index=0),
        _ConstantModel(),
        notebook_path=notebook,
        protocol_version="jupyter-v1",
    )
    with pytest.raises(CandidateBundleError, match="already exists"):
        store.write_bundle(
            _record("first", index=1),
            _ConstantModel(),
            notebook_path=notebook,
            protocol_version="jupyter-v1",
        )
    store.write_bundle(
        _record("second", index=0),
        _ConstantModel(),
        notebook_path=notebook,
        protocol_version="jupyter-v1",
    )

    with pytest.raises(CandidateBundleError, match="registration_index"):
        store.load_records()


def test_registration_index_gap_fails_closed(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    store.write_bundle(
        _record("second", index=1),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )

    with pytest.raises(CandidateBundleError, match="sequence has gaps"):
        store.load_records()


def test_record_mismatch_and_unsafe_candidate_id_fail_closed(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )

    with pytest.raises(CandidateBundleError, match="does not match stored metadata"):
        store.verify_record(_record(index=4))
    for unsafe_id in ("../escape", ".", "..", ".tmp-ignored"):
        with pytest.raises(CandidateBundleError, match="storage-safe"):
            store.write_bundle(
                _record(unsafe_id),
                _ConstantModel(),
                notebook_path=_notebook(tmp_path),
                protocol_version="jupyter-v1",
            )
    assert bundle.bundle_dir.is_dir()


def test_symlinked_bundle_file_is_rejected_when_supported(tmp_path):
    store = CandidateBundleStore(tmp_path / "candidates")
    bundle = store.write_bundle(
        _record(),
        _ConstantModel(),
        notebook_path=_notebook(tmp_path),
        protocol_version="jupyter-v1",
    )
    model_path = bundle.bundle_dir / "model.pkl"
    replacement = tmp_path / "replacement.pkl"
    replacement.write_bytes(model_path.read_bytes())
    model_path.unlink()
    try:
        model_path.symlink_to(replacement)
    except OSError:
        pytest.skip("File symlinks are not available on this platform")

    with pytest.raises(CandidateBundleError, match="non-regular"):
        store.load_records()

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are not available on this platform")
    with pytest.raises(CandidateBundleError, match="root is not a regular directory"):
        CandidateBundleStore(linked_root).load_records()
