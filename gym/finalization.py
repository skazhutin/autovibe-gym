from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cloudpickle


PROTECTED_FINALIZATION_SCHEMA_VERSION = "protected-finalization-v1"
_ARTIFACT_FILES = ("receipt.json", "replayed_model.pkl")
_ARTIFACT_ENTRIES = frozenset((*_ARTIFACT_FILES, "SHA256SUMS"))
_SOURCE_CANDIDATE_FILES = frozenset(
    ("candidate.json", "model.pkl", "solution.ipynb")
)
_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "protocol_version",
        "source_candidate_hashes",
        "resource_snapshot",
        "files",
    }
)


class ProtectedFinalizationError(RuntimeError):
    """A producing-revision artifact is missing, malformed, or corrupted."""


@dataclass(frozen=True)
class ProtectedFinalizationArtifact:
    candidate_id: str
    artifact_dir: Path
    model_path: Path
    hashes: Mapping[str, str]
    protocol_version: str
    source_candidate_hashes: Mapping[str, str]
    resource_snapshot: Mapping[str, Any]

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": PROTECTED_FINALIZATION_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "hashes": dict(sorted(self.hashes.items())),
            "source_candidate_hashes": dict(
                sorted(self.source_candidate_hashes.items())
            ),
        }


class ProtectedFinalizationStore:
    """Atomic, immutable storage for the model produced by protected replay."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))

    def write_artifact(
        self,
        *,
        candidate_id: str,
        model: Any,
        protocol_version: str,
        source_candidate_hashes: Mapping[str, str],
        resource_snapshot: Mapping[str, Any] | None = None,
    ) -> ProtectedFinalizationArtifact:
        self._validate_candidate_id(candidate_id)
        if not isinstance(protocol_version, str) or not protocol_version:
            raise ProtectedFinalizationError("Finalization protocol_version is invalid")
        source_hashes = _validated_hashes(source_candidate_hashes)
        snapshot = dict(resource_snapshot or {})
        self._ensure_root()
        final_dir = self.root / candidate_id
        if os.path.lexists(final_dir):
            raise ProtectedFinalizationError(
                f"Protected finalization artifact already exists: {candidate_id}"
            )

        try:
            model_bytes = cloudpickle.dumps(model)
        except Exception as exc:
            raise ProtectedFinalizationError(
                "Could not serialize protected finalization model"
            ) from exc
        try:
            receipt_bytes = (
                json.dumps(
                    {
                        "schema_version": PROTECTED_FINALIZATION_SCHEMA_VERSION,
                        "candidate_id": candidate_id,
                        "protocol_version": protocol_version,
                        "source_candidate_hashes": source_hashes,
                        "resource_snapshot": snapshot,
                        "files": {"model": "replayed_model.pkl"},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtectedFinalizationError(
                "Protected finalization receipt is not valid JSON"
            ) from exc

        payloads = {
            "receipt.json": receipt_bytes,
            "replayed_model.pkl": model_bytes,
        }
        hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
        checksum_bytes = "".join(
            f"{hashes[name]}  {name}\n" for name in sorted(hashes)
        ).encode("ascii")
        temp_dir = self.root / f".tmp-{candidate_id}-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir()
            for name, payload in payloads.items():
                _write_bytes_durable(temp_dir / name, payload)
            _write_bytes_durable(temp_dir / "SHA256SUMS", checksum_bytes)
            _fsync_directory_best_effort(temp_dir)
            if os.path.lexists(final_dir):
                raise ProtectedFinalizationError(
                    f"Protected finalization artifact already exists: {candidate_id}"
                )
            os.replace(temp_dir, final_dir)
            _fsync_directory_best_effort(self.root)
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(exc, ProtectedFinalizationError):
                raise
            raise ProtectedFinalizationError(
                "Could not atomically store protected finalization artifact"
            ) from exc
        return ProtectedFinalizationArtifact(
            candidate_id=candidate_id,
            artifact_dir=final_dir,
            model_path=final_dir / "replayed_model.pkl",
            hashes=hashes,
            protocol_version=protocol_version,
            source_candidate_hashes=source_hashes,
            resource_snapshot=snapshot,
        )

    def verify_artifact(self, candidate_id: str) -> ProtectedFinalizationArtifact:
        self._validate_candidate_id(candidate_id)
        artifact_dir = self.root / candidate_id
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise ProtectedFinalizationError(
                f"Missing protected finalization artifact: {candidate_id}"
            )
        try:
            entries = {entry.name for entry in artifact_dir.iterdir()}
        except OSError as exc:
            raise ProtectedFinalizationError(
                "Could not enumerate protected finalization artifact"
            ) from exc
        if entries != _ARTIFACT_ENTRIES:
            raise ProtectedFinalizationError(
                "Protected finalization artifact has missing or unexpected files"
            )
        for name in _ARTIFACT_ENTRIES:
            path = artifact_dir / name
            if path.is_symlink() or not path.is_file():
                raise ProtectedFinalizationError(
                    "Protected finalization artifact contains a non-regular file"
                )
        try:
            hashes = _parse_checksums((artifact_dir / "SHA256SUMS").read_bytes())
            for name in _ARTIFACT_FILES:
                if _sha256_file(artifact_dir / name) != hashes[name]:
                    raise ProtectedFinalizationError(
                        f"Protected finalization checksum mismatch: {name}"
                    )
            receipt = _load_strict_json((artifact_dir / "receipt.json").read_bytes())
        except ProtectedFinalizationError:
            raise
        except OSError as exc:
            raise ProtectedFinalizationError(
                "Could not read protected finalization artifact"
            ) from exc
        if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
            raise ProtectedFinalizationError(
                "Protected finalization receipt has an invalid shape"
            )
        if receipt["schema_version"] != PROTECTED_FINALIZATION_SCHEMA_VERSION:
            raise ProtectedFinalizationError(
                "Unsupported protected finalization schema"
            )
        if receipt["candidate_id"] != candidate_id:
            raise ProtectedFinalizationError(
                "Finalization directory and receipt IDs differ"
            )
        if (
            not isinstance(receipt["protocol_version"], str)
            or not receipt["protocol_version"]
        ):
            raise ProtectedFinalizationError(
                "Finalization protocol_version is invalid"
            )
        if receipt["files"] != {"model": "replayed_model.pkl"}:
            raise ProtectedFinalizationError(
                "Protected finalization file map is invalid"
            )
        if not isinstance(receipt["resource_snapshot"], dict):
            raise ProtectedFinalizationError(
                "Protected finalization resource snapshot is invalid"
            )
        source_hashes = _validated_hashes(receipt["source_candidate_hashes"])
        return ProtectedFinalizationArtifact(
            candidate_id=candidate_id,
            artifact_dir=artifact_dir,
            model_path=artifact_dir / "replayed_model.pkl",
            hashes=hashes,
            protocol_version=receipt["protocol_version"],
            source_candidate_hashes=source_hashes,
            resource_snapshot=receipt["resource_snapshot"],
        )

    def load_model(self, candidate_id: str) -> Any:
        artifact = self.verify_artifact(candidate_id)
        try:
            return cloudpickle.loads(artifact.model_path.read_bytes())
        except Exception as exc:
            raise ProtectedFinalizationError(
                "Protected finalization model could not be deserialized"
            ) from exc

    def _ensure_root(self) -> None:
        if os.path.lexists(self.root):
            if self.root.is_symlink() or not self.root.is_dir():
                raise ProtectedFinalizationError(
                    "Protected finalization root is not a regular directory"
                )
            return
        try:
            self.root.mkdir(parents=True)
        except FileExistsError:
            if self.root.is_symlink() or not self.root.is_dir():
                raise ProtectedFinalizationError(
                    "Protected finalization root is not a regular directory"
                )
        except OSError as exc:
            raise ProtectedFinalizationError(
                "Could not create protected finalization store"
            ) from exc

    @staticmethod
    def _validate_candidate_id(candidate_id: str) -> None:
        invalid = (
            not isinstance(candidate_id, str)
            or not _CANDIDATE_ID_PATTERN.fullmatch(candidate_id)
            or candidate_id in {".", ".."}
            or candidate_id.startswith(".tmp-")
        )
        if invalid:
            raise ProtectedFinalizationError("Candidate ID is not storage-safe")


def _validated_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_CANDIDATE_FILES:
        raise ProtectedFinalizationError("Source candidate hashes are invalid")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(digest, str)
            or not _HEX_DIGEST_PATTERN.fullmatch(digest)
        ):
            raise ProtectedFinalizationError("Source candidate hashes are invalid")
        result[name] = digest
    return dict(sorted(result.items()))


def _load_strict_json(raw: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtectedFinalizationError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtectedFinalizationError(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except ProtectedFinalizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedFinalizationError(
            "Protected finalization receipt is not valid JSON"
        ) from exc


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtectedFinalizationError(
            "Protected finalization checksums are not ASCII"
        ) from exc
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ProtectedFinalizationError(
                "Protected finalization checksum line is malformed"
            )
        digest, name = parts
        if not _HEX_DIGEST_PATTERN.fullmatch(digest) or name not in _ARTIFACT_FILES:
            raise ProtectedFinalizationError(
                "Protected finalization checksum entry is invalid"
            )
        if name in result:
            raise ProtectedFinalizationError(
                "Protected finalization checksum entry is duplicated"
            )
        result[name] = digest
    if set(result) != set(_ARTIFACT_FILES):
        raise ProtectedFinalizationError(
            "Protected finalization checksum manifest is incomplete"
        )
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
