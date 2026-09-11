from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import cloudpickle

from .candidates import CandidateRecord


CANDIDATE_BUNDLE_SCHEMA_VERSION = "candidate-bundle-v1"
_BUNDLE_FILES = ("candidate.json", "model.pkl", "solution.ipynb")
_BUNDLE_ENTRIES = frozenset((*_BUNDLE_FILES, "SHA256SUMS"))
_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_METADATA_KEYS = frozenset(
    {"schema_version", "candidate", "protocol_version", "resource_snapshot", "files"}
)
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "model_var",
        "notebook_revision",
        "clean_run_id",
        "metric_name",
        "metric_direction",
        "validation_metric",
        "validation_success",
        "raw_inference_ready",
        "created_step",
        "registration_index",
    }
)


class CandidateBundleError(RuntimeError):
    """A private candidate bundle is missing, malformed, or corrupted."""


@dataclass(frozen=True)
class CandidateBundle:
    record: CandidateRecord
    bundle_dir: Path
    hashes: Mapping[str, str]
    protocol_version: str
    resource_snapshot: Mapping[str, Any]

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
            "candidate_id": self.record.candidate_id,
            "hashes": dict(sorted(self.hashes.items())),
        }


class CandidateBundleStore:
    """Atomic, immutable private storage for validated candidate evidence."""

    def __init__(self, root: str | Path) -> None:
        # Normalize to an absolute path without resolving symlinks so the
        # fail-closed root check can still detect a linked store directory.
        self.root = Path(os.path.abspath(os.fspath(root)))

    def write_bundle(
        self,
        record: CandidateRecord,
        model: Any,
        *,
        notebook_path: str | Path,
        protocol_version: str,
        resource_snapshot: Mapping[str, Any] | None = None,
    ) -> CandidateBundle:
        self._validate_candidate_id(record.candidate_id)
        _record_from_payload(_record_payload(record))
        if not isinstance(protocol_version, str) or not protocol_version:
            raise CandidateBundleError("Candidate protocol_version is invalid")
        self._ensure_root()
        final_dir = self.root / record.candidate_id
        if os.path.lexists(final_dir):
            raise CandidateBundleError(
                f"Candidate bundle already exists: {record.candidate_id}"
            )

        source_notebook = Path(notebook_path)
        if source_notebook.is_symlink() or not source_notebook.is_file():
            raise CandidateBundleError("Notebook snapshot source is not a regular file")

        try:
            model_bytes = cloudpickle.dumps(model)
            notebook_bytes = source_notebook.read_bytes()
        except Exception as exc:
            raise CandidateBundleError("Could not serialize candidate bundle") from exc

        snapshot = dict(resource_snapshot or {})
        metadata = {
            "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
            "candidate": _record_payload(record),
            "protocol_version": protocol_version,
            "resource_snapshot": snapshot,
            "files": {
                "model": "model.pkl",
                "notebook": "solution.ipynb",
            },
        }
        try:
            metadata_bytes = (
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CandidateBundleError("Candidate metadata is not valid JSON") from exc

        payloads = {
            "candidate.json": metadata_bytes,
            "model.pkl": model_bytes,
            "solution.ipynb": notebook_bytes,
        }
        hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
        checksum_bytes = "".join(
            f"{hashes[name]}  {name}\n" for name in sorted(hashes)
        ).encode("ascii")

        temp_dir = self.root / f".tmp-{record.candidate_id}-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir()
            for name, payload in payloads.items():
                _write_bytes_durable(temp_dir / name, payload)
            _write_bytes_durable(temp_dir / "SHA256SUMS", checksum_bytes)
            _fsync_directory_best_effort(temp_dir)
            if os.path.lexists(final_dir):
                raise CandidateBundleError(
                    f"Candidate bundle already exists: {record.candidate_id}"
                )
            os.replace(temp_dir, final_dir)
            _fsync_directory_best_effort(self.root)
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(exc, CandidateBundleError):
                raise
            raise CandidateBundleError("Could not atomically store candidate bundle") from exc

        stored_record = replace(record, artifact_path=str(final_dir / "model.pkl"))
        return CandidateBundle(
            record=stored_record,
            bundle_dir=final_dir,
            hashes=hashes,
            protocol_version=protocol_version,
            resource_snapshot=snapshot,
        )

    def load_records(self) -> list[CandidateRecord]:
        if not os.path.lexists(self.root):
            return []
        self._ensure_root()
        bundles: list[CandidateBundle] = []
        try:
            entries = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise CandidateBundleError("Could not enumerate candidate store") from exc
        for entry in entries:
            if entry.name.startswith(".tmp-"):
                continue
            if entry.is_symlink() or not entry.is_dir():
                raise CandidateBundleError(
                    f"Unexpected candidate-store entry: {entry.name}"
                )
            bundles.append(self._load_bundle(entry.name))

        indices = [bundle.record.registration_index for bundle in bundles]
        if len(indices) != len(set(indices)):
            raise CandidateBundleError("Duplicate candidate registration_index")
        if sorted(indices) != list(range(len(indices))):
            raise CandidateBundleError("Candidate registration_index sequence has gaps")
        bundles.sort(
            key=lambda bundle: (
                bundle.record.registration_index,
                bundle.record.candidate_id,
            )
        )
        return [bundle.record for bundle in bundles]

    def verify_record(self, record: CandidateRecord) -> CandidateBundle:
        bundle = self._load_bundle(record.candidate_id)
        if _record_payload(bundle.record) != _record_payload(record):
            raise CandidateBundleError("Candidate record does not match stored metadata")
        expected_artifact = bundle.bundle_dir / "model.pkl"
        if record.artifact_path is not None:
            try:
                actual_artifact = Path(record.artifact_path).resolve(strict=False)
            except OSError as exc:
                raise CandidateBundleError("Candidate artifact path is invalid") from exc
            if actual_artifact != expected_artifact.resolve(strict=False):
                raise CandidateBundleError("Candidate artifact path does not match bundle")
        return bundle

    def load_model(self, record: CandidateRecord) -> Any:
        bundle = self.verify_record(record)
        try:
            return cloudpickle.loads((bundle.bundle_dir / "model.pkl").read_bytes())
        except Exception as exc:
            raise CandidateBundleError("Candidate model could not be deserialized") from exc

    def notebook_snapshot_path(self, record: CandidateRecord) -> Path:
        bundle = self.verify_record(record)
        return bundle.bundle_dir / "solution.ipynb"

    def _load_bundle(self, candidate_id: str) -> CandidateBundle:
        self._validate_candidate_id(candidate_id)
        bundle_dir = self.root / candidate_id
        if bundle_dir.is_symlink() or not bundle_dir.is_dir():
            raise CandidateBundleError(f"Missing candidate bundle: {candidate_id}")

        try:
            entries = {entry.name for entry in bundle_dir.iterdir()}
        except OSError as exc:
            raise CandidateBundleError("Could not enumerate candidate bundle") from exc
        if entries != _BUNDLE_ENTRIES:
            raise CandidateBundleError("Candidate bundle has missing or unexpected files")
        for name in _BUNDLE_ENTRIES:
            path = bundle_dir / name
            if path.is_symlink() or not path.is_file():
                raise CandidateBundleError("Candidate bundle contains a non-regular file")

        try:
            hashes = _parse_checksums((bundle_dir / "SHA256SUMS").read_bytes())
            for name in _BUNDLE_FILES:
                actual = _sha256_file(bundle_dir / name)
                if actual != hashes[name]:
                    raise CandidateBundleError(
                        f"Candidate bundle checksum mismatch: {name}"
                    )
            metadata_raw = (bundle_dir / "candidate.json").read_bytes()
        except CandidateBundleError:
            raise
        except OSError as exc:
            raise CandidateBundleError("Could not read candidate bundle") from exc

        metadata = _load_strict_json(metadata_raw)
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise CandidateBundleError("Candidate metadata has an invalid shape")
        if metadata["schema_version"] != CANDIDATE_BUNDLE_SCHEMA_VERSION:
            raise CandidateBundleError("Unsupported candidate bundle schema")
        if not isinstance(metadata["protocol_version"], str) or not metadata["protocol_version"]:
            raise CandidateBundleError("Candidate protocol_version is invalid")
        if not isinstance(metadata["resource_snapshot"], dict):
            raise CandidateBundleError("Candidate resource_snapshot is invalid")
        if metadata["files"] != {
            "model": "model.pkl",
            "notebook": "solution.ipynb",
        }:
            raise CandidateBundleError("Candidate bundle file map is invalid")

        record = _record_from_payload(metadata["candidate"])
        if record.candidate_id != candidate_id:
            raise CandidateBundleError("Candidate directory and metadata IDs differ")
        record = replace(record, artifact_path=str(bundle_dir / "model.pkl"))
        return CandidateBundle(
            record=record,
            bundle_dir=bundle_dir,
            hashes=hashes,
            protocol_version=metadata["protocol_version"],
            resource_snapshot=metadata["resource_snapshot"],
        )

    def _ensure_root(self) -> None:
        if os.path.lexists(self.root):
            if self.root.is_symlink() or not self.root.is_dir():
                raise CandidateBundleError("Candidate store root is not a regular directory")
            return
        try:
            self.root.mkdir(parents=True)
        except FileExistsError:
            if self.root.is_symlink() or not self.root.is_dir():
                raise CandidateBundleError(
                    "Candidate store root is not a regular directory"
                )
        except OSError as exc:
            raise CandidateBundleError("Could not create candidate store") from exc

    @staticmethod
    def _validate_candidate_id(candidate_id: str) -> None:
        invalid = (
            not isinstance(candidate_id, str)
            or not _CANDIDATE_ID_PATTERN.fullmatch(candidate_id)
            or candidate_id in {".", ".."}
            or candidate_id.startswith(".tmp-")
        )
        if invalid:
            raise CandidateBundleError("Candidate ID is not storage-safe")


def _record_payload(record: CandidateRecord) -> dict[str, Any]:
    return {
        "candidate_id": record.candidate_id,
        "model_var": record.model_var,
        "notebook_revision": record.notebook_revision,
        "clean_run_id": record.clean_run_id,
        "metric_name": record.metric_name,
        "metric_direction": record.metric_direction,
        "validation_metric": record.validation_metric,
        "validation_success": record.validation_success,
        "raw_inference_ready": record.raw_inference_ready,
        "created_step": record.created_step,
        "registration_index": record.registration_index,
    }


def _record_from_payload(payload: Any) -> CandidateRecord:
    if not isinstance(payload, dict) or set(payload) != _CANDIDATE_KEYS:
        raise CandidateBundleError("Candidate record metadata has an invalid shape")

    direction = payload["metric_direction"]
    if direction not in {"higher", "lower"}:
        raise CandidateBundleError("Candidate metric_direction is invalid")
    for key in ("candidate_id", "model_var", "metric_name"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise CandidateBundleError(f"Candidate {key} is invalid")
    if not isinstance(payload["clean_run_id"], str):
        raise CandidateBundleError("Candidate clean_run_id is invalid")
    for key in ("notebook_revision", "created_step", "registration_index"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise CandidateBundleError(f"Candidate {key} is invalid")
    for key in ("validation_success", "raw_inference_ready"):
        if type(payload[key]) is not bool or not payload[key]:
            raise CandidateBundleError(f"Candidate {key} is invalid")
    metric = payload["validation_metric"]
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise CandidateBundleError("Candidate validation_metric is invalid")
    if not math.isfinite(float(metric)):
        raise CandidateBundleError("Candidate validation_metric is not finite")

    return CandidateRecord(
        candidate_id=payload["candidate_id"],
        model_var=payload["model_var"],
        notebook_revision=payload["notebook_revision"],
        clean_run_id=payload["clean_run_id"],
        metric_name=payload["metric_name"],
        metric_direction=direction,  # type: ignore[arg-type]
        validation_metric=float(metric),
        validation_success=payload["validation_success"],
        raw_inference_ready=payload["raw_inference_ready"],
        created_step=payload["created_step"],
        registration_index=payload["registration_index"],
    )


def _load_strict_json(raw: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateBundleError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise CandidateBundleError(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except CandidateBundleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateBundleError("Candidate metadata is not valid JSON") from exc


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CandidateBundleError("Candidate checksums are not ASCII") from exc
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise CandidateBundleError("Candidate checksum line is malformed")
        digest, name = parts
        if not _HEX_DIGEST_PATTERN.fullmatch(digest) or name not in _BUNDLE_FILES:
            raise CandidateBundleError("Candidate checksum entry is invalid")
        if name in result:
            raise CandidateBundleError("Candidate checksum entry is duplicated")
        result[name] = digest
    if set(result) != set(_BUNDLE_FILES):
        raise CandidateBundleError("Candidate checksum manifest is incomplete")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory_best_effort(path: Path) -> None:
    """Persist directory entries where the platform supports directory fsync."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
