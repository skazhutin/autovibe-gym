"""Auditable, behavior-neutral artifacts for Paper V1 experiment runs."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
REQUIRED_CONDITION_FIELDS = (
    "experiment_id",
    "git_commit",
    "dataset_id",
    "dataset_hash",
    "split_id",
    "split_seed",
    "model_provider",
    "model_id",
    "model_version",
    "decoding_config_hash",
    "arm",
    "replicate_index",
    "budget_policy_hash",
    "execution_policy_hash",
    "prompt_template_hash",
)
_SENSITIVE_KEY = re.compile(
    r"(^|_)(api_?key|password|passwd|secret|authorization|cookie|access_token|refresh_token)($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]+|(?:sk|ghp|gho|github_pat)-?[a-z0-9_-]{12,})"
)
_RUN_ID = re.compile(r"^run_[A-Za-z0-9._-]+$")


class FailureCategory(str, Enum):
    SUCCESS = "success"
    INVALID_SUBMISSION = "invalid_submission"
    AGENT_CODE_FAILURE = "agent_code_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXECUTION_TIMEOUT = "execution_timeout"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_CAPACITY = "provider_capacity"
    PROVIDER_OTHER = "provider_other"
    SANDBOX_FAILURE = "sandbox_failure"
    ORCHESTRATOR_FAILURE = "orchestrator_failure"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def sanitize(value: Any, *, omit_private: bool = False) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if omit_private and key == "private":
                continue
            cleaned[key] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(key)
                else sanitize(raw_value, omit_private=omit_private)
            )
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item, omit_private=omit_private) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)[:20_000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)[:20_000]


def validate_condition(condition: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_CONDITION_FIELDS if key not in condition]
    if missing:
        raise ValueError(f"Condition is missing required fields: {', '.join(missing)}")
    if int(condition["replicate_index"]) < 0:
        raise ValueError("replicate_index must be non-negative")
    for key in REQUIRED_CONDITION_FIELDS:
        if key not in {"split_seed", "replicate_index"} and not str(condition[key]).strip():
            raise ValueError(f"Condition field {key!r} must not be empty")


def condition_id(condition: Mapping[str, Any]) -> str:
    validate_condition(condition)
    frozen = {key: condition[key] for key in REQUIRED_CONDITION_FIELDS}
    return f"cond_{canonical_hash(frozen)[:24]}"


def classify_failure(
    *,
    final_status: str | None = None,
    valid_submit: bool | None = None,
    error_type: str | None = None,
    http_status: int | None = None,
    origin: str | None = None,
) -> FailureCategory:
    status = " ".join(filter(None, [final_status, error_type])).lower()
    if valid_submit is True or ("submitted" in status and "fail" not in status):
        return FailureCategory.SUCCESS
    if http_status == 429:
        return FailureCategory.PROVIDER_RATE_LIMIT
    if http_status in {401, 403} or any(word in status for word in ("auth", "api key", "permission")):
        return FailureCategory.PROVIDER_AUTH
    if http_status in {502, 503, 504, 529} or any(
        word in status for word in ("capacity", "unavailable", "overloaded")
    ):
        return FailureCategory.PROVIDER_CAPACITY
    if any(word in status for word in ("budget", "max_steps", "max steps", "token limit")):
        return FailureCategory.BUDGET_EXHAUSTED
    if any(word in status for word in ("timeout", "timed out")):
        return FailureCategory.EXECUTION_TIMEOUT
    if any(
        word in status
        for word in (
            "invalid_submission",
            "no_candidate",
            "submit_blocked",
            "hidden_submit_failed",
        )
    ):
        return FailureCategory.INVALID_SUBMISSION
    if "sandbox" in status or origin == "sandbox":
        return FailureCategory.SANDBOX_FAILURE
    if origin == "provider":
        return FailureCategory.PROVIDER_OTHER
    if origin == "orchestrator":
        return FailureCategory.ORCHESTRATOR_FAILURE
    return FailureCategory.AGENT_CODE_FAILURE


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class JsonlLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_index = 0

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "index": self._next_index,
            "timestamp": utc_now(),
            **sanitize(event),
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_index += 1
        return record

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


@dataclass
class RunRecorder:
    run_dir: Path
    manifest: dict[str, Any]
    usage_ledger: JsonlLedger
    execution_ledger: JsonlLedger
    manifest_events: JsonlLedger
    _finalized: bool = False

    @classmethod
    def create(
        cls,
        root: str | Path,
        condition: Mapping[str, Any],
        *,
        expected_condition_id: str | None = None,
        run_id: str | None = None,
        rerun_of: str | None = None,
        rerun_reason: str | None = None,
        git_dirty: bool | None = None,
    ) -> "RunRecorder":
        clean_condition = sanitize(dict(condition))
        computed_condition_id = condition_id(clean_condition)
        if expected_condition_id and expected_condition_id != computed_condition_id:
            raise ValueError(
                f"Condition ID mismatch: expected {expected_condition_id}, computed {computed_condition_id}"
            )
        if rerun_of and not rerun_reason:
            raise ValueError("rerun_reason is required when rerun_of is set")

        unique_run_id = run_id or f"run_{uuid.uuid4().hex}"
        if not _RUN_ID.fullmatch(unique_run_id):
            raise ValueError("run_id must match ^run_[A-Za-z0-9._-]+$")
        run_dir = Path(root).resolve() / unique_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        started_at = utc_now()
        recorder = cls(
            run_dir=run_dir,
            manifest={
                "schema_version": SCHEMA_VERSION,
                "experiment_id": clean_condition["experiment_id"],
                "condition_id": computed_condition_id,
                "run_id": unique_run_id,
                "rerun_of": rerun_of,
                "rerun_reason": sanitize(rerun_reason),
                "state": "running",
                "started_at": started_at,
                "completed_at": None,
                "condition": clean_condition,
                "provenance": {
                    "git_commit": clean_condition["git_commit"],
                    "git_dirty": git_dirty,
                },
                "failure_category": None,
                "final_status": None,
                "summary": {},
                "ledger_totals": {},
                "artifacts": {
                    "usage_ledger": "usage_ledger.jsonl",
                    "execution_ledger": "execution_ledger.jsonl",
                    "manifest_events": "manifest_events.jsonl",
                },
            },
            usage_ledger=JsonlLedger(run_dir / "usage_ledger.jsonl"),
            execution_ledger=JsonlLedger(run_dir / "execution_ledger.jsonl"),
            manifest_events=JsonlLedger(run_dir / "manifest_events.jsonl"),
        )
        recorder._write_manifest()
        recorder.manifest_events.append({"event": "run_started", "state": "running"})
        atexit.register(recorder._finalize_abandoned)
        return recorder

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def condition_id(self) -> str:
        return str(self.manifest["condition_id"])

    def _write_manifest(self) -> None:
        validate_manifest(self.manifest)
        _atomic_write_json(self.manifest_path, self.manifest)

    def record_usage(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_seconds: float | None = None,
        success: bool = True,
        error_category: FailureCategory | None = None,
        request_id: str | None = None,
        finish_reason: str | None = None,
        retry_index: int = 0,
        event: str = "logical_llm_call",
    ) -> dict[str, Any]:
        return self.usage_ledger.append(
            {
                "event": event,
                "provider": provider,
                "model": model,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "duration_seconds": duration_seconds,
                "success": success,
                "error_category": error_category,
                "request_id": request_id,
                "finish_reason": finish_reason,
                "retry_index": retry_index,
            }
        )

    def record_execution(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return self.execution_ledger.append(event)

    def record_notebook_events(self, events: list[Mapping[str, Any]]) -> None:
        for event in events:
            public_event = sanitize(event, omit_private=True)
            self.record_execution({"event": "notebook_event", "payload": public_event})

    def finalize(
        self,
        summary: Mapping[str, Any],
        *,
        failure_category: FailureCategory | None = None,
    ) -> Path:
        if self._finalized:
            return self.manifest_path
        clean_summary = sanitize(dict(summary), omit_private=True)
        category = failure_category or classify_failure(
            final_status=str(clean_summary.get("final_status") or clean_summary.get("status") or ""),
            valid_submit=clean_summary.get("valid_submit"),
        )
        usage = self.usage_ledger.read()
        executions = self.execution_ledger.read()
        logical_calls = [item for item in usage if item.get("event") == "logical_llm_call"]
        provider_attempts = [
            item for item in usage if item.get("event") == "provider_request_attempt"
        ]
        self.manifest.update(
            {
                "state": "completed",
                "completed_at": utc_now(),
                "failure_category": category.value,
                "final_status": clean_summary.get("final_status") or clean_summary.get("status"),
                "summary": clean_summary,
                "ledger_totals": {
                    "logical_llm_calls": len(logical_calls),
                    "provider_request_attempts": len(provider_attempts),
                    "input_tokens": sum(
                        int(item.get("input_tokens") or 0) for item in logical_calls
                    ),
                    "output_tokens": sum(
                        int(item.get("output_tokens") or 0) for item in logical_calls
                    ),
                    "execution_events": len(executions),
                },
            }
        )
        self._write_manifest()
        self.manifest_events.append(
            {"event": "run_finalized", "state": "completed", "failure_category": category}
        )
        self._finalized = True
        atexit.unregister(self._finalize_abandoned)
        return self.manifest_path

    def fail(self, error: BaseException, *, origin: str = "orchestrator") -> Path:
        category = classify_failure(error_type=type(error).__name__, origin=origin)
        return self.finalize(
            {
                "final_status": "uncaught_exception",
                "valid_submit": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            failure_category=category,
        )

    def _finalize_abandoned(self) -> None:
        if not self._finalized:
            self.finalize(
                {
                    "final_status": "process_exited_before_finalization",
                    "valid_submit": False,
                },
                failure_category=FailureCategory.ORCHESTRATOR_FAILURE,
            )


class RecordingLLMClient:
    """Transparent logical-call recorder; provider retry behavior is untouched."""

    def __init__(self, client: Any, recorder: RunRecorder, *, provider: str, model: str):
        self._client = client
        self._recorder = recorder
        self._provider = provider
        self._model = model
        set_hook = getattr(client, "set_request_attempt_hook", None)
        if callable(set_hook):
            set_hook(self._record_provider_attempt)

    def _record_provider_attempt(self, event: Mapping[str, Any]) -> None:
        error_category = None
        if not event.get("success", False):
            error_category = classify_failure(
                error_type=str(event.get("error_type") or ""),
                http_status=event.get("http_status"),
                origin="provider",
            )
        self._recorder.record_usage(
            provider=self._provider,
            model=self._model,
            input_tokens=int(event.get("input_tokens") or 0),
            output_tokens=int(event.get("output_tokens") or 0),
            duration_seconds=event.get("duration_seconds"),
            success=bool(event.get("success")),
            error_category=error_category,
            request_id=event.get("request_id"),
            finish_reason=event.get("finish_reason"),
            retry_index=int(event.get("retry_index") or 0),
            event="provider_request_attempt",
        )

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = self._client.complete(*args, **kwargs)
        except Exception as error:
            self._recorder.record_usage(
                provider=self._provider,
                model=self._model,
                duration_seconds=time.perf_counter() - started,
                success=False,
                error_category=classify_failure(error_type=type(error).__name__, origin="provider"),
            )
            raise
        self._recorder.record_usage(
            provider=self._provider,
            model=self._model,
            input_tokens=getattr(response, "input_tokens", 0),
            output_tokens=getattr(response, "output_tokens", 0),
            duration_seconds=time.perf_counter() - started,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class RecordingExecutor:
    def __init__(self, executor: Any, recorder: RunRecorder):
        self._executor = executor
        self._recorder = recorder

    def run(self, code: str, namespace: dict | None = None) -> Any:
        started = time.perf_counter()
        try:
            result = self._executor.run(code, namespace)
        except Exception as error:
            self._recorder.record_execution(
                {
                    "event": "code_execution",
                    "status": "exception",
                    "duration_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                }
            )
            raise
        stderr = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
        self._recorder.record_execution(
            {
                "event": "code_execution",
                "status": "error" if stderr else "success",
                "duration_seconds": time.perf_counter() - started,
                "code_hash": canonical_hash(code),
            }
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "condition_id",
        "run_id",
        "state",
        "started_at",
        "condition",
        "provenance",
        "failure_category",
        "summary",
        "ledger_totals",
        "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema version: {manifest['schema_version']}")
    if manifest["state"] not in {"running", "completed"}:
        raise ValueError(f"Invalid manifest state: {manifest['state']}")
    validate_condition(manifest["condition"])
    if manifest["condition_id"] != condition_id(manifest["condition"]):
        raise ValueError("Manifest condition_id does not match condition payload")
