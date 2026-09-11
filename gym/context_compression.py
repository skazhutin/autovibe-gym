from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


CONTEXT_PACK_SCHEMA_VERSION = "context-pack-v1"
CONTEXT_COMPRESSION_POLICY_VERSION = "deterministic-context-v1"
_CONTEXT_PREFIX = f"[CONTEXT PACK {CONTEXT_PACK_SCHEMA_VERSION}]\n"
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "task_contract",
        "tested_hypotheses",
        "incumbent_state",
        "unresolved_errors",
        "remaining_resources",
        "allowed_actions",
        "notebook_state",
        "finalization_contract",
    }
)
_TASK_KEYS = frozenset(
    {
        "protocol_version",
        "target_column",
        "metric_name",
        "metric_direction",
        "max_steps",
        "train_rows",
        "feedback_validation_rows",
        "feature_schema",
        "feature_schema_sha256",
    }
)
_FEATURE_KEYS = frozenset({"name", "dtype"})
_HYPOTHESIS_KEYS = frozenset(
    {
        "action",
        "model_var",
        "notebook_revision",
        "outcome",
        "validation_metric",
        "feedback_keys",
        "candidate_id",
    }
)
_INCUMBENT_KEYS = frozenset(
    {
        "candidate_id",
        "model_var",
        "notebook_revision",
        "metric_name",
        "metric_direction",
        "validation_metric",
        "raw_inference_ready",
        "registration_index",
    }
)
_ERROR_KEYS = frozenset({"action", "feedback_keys", "cell_id"})
_RESOURCE_KEYS = frozenset(
    {
        "steps_remaining",
        "step_limit",
        "episode_budget",
    }
)
_EPISODE_BUDGET_KEYS = frozenset(
    {
        "phase",
        "global_remaining",
        "exploration_remaining",
        "finalization_remaining",
        "exploration_exhausted_reason",
        "validation_query_budget",
    }
)
_RESOURCE_COUNTER_KEYS = frozenset(
    {"total_tokens", "llm_calls", "code_executions", "tool_calls"}
)
_VALIDATION_QUERY_BUDGET_KEYS = frozenset(
    {
        "policy_version",
        "max_queries",
        "finalization_reserve_queries",
        "feedback_numeric_decimals",
        "queries_used",
        "phase_remaining_queries",
        "global_remaining_queries",
    }
)
_NOTEBOOK_KEYS = frozenset(
    {
        "dirty_since_clean_run",
        "clean_run_available",
        "validated_candidate_available",
        "notebook_revision",
        "last_clean_run_id",
    }
)
_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContextCompressionError(RuntimeError):
    """The deterministic context contract is invalid or cannot fit safely."""


@dataclass(frozen=True)
class ContextCompressionPolicy:
    """Explicit M3 bounds; confirmatory values must be selected and frozen later."""

    max_context_bytes: int
    max_recent_messages: int
    max_recent_message_bytes: int
    max_feature_columns: int
    max_tested_hypotheses: int
    max_unresolved_errors: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            if name in {"max_context_bytes", "max_recent_message_bytes"}:
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CompressedContext:
    messages: tuple[Mapping[str, str], ...]
    pack_bytes: bytes
    context_bytes: int
    pack_sha256: str
    context_sha256: str
    source_context_bytes: int
    source_context_sha256: str
    retained_recent_messages: int
    dropped_recent_messages: int
    truncated_recent_messages: int
    truncation: Mapping[str, int]
    policy: Mapping[str, int]

    def as_messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages]

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
            "policy_version": CONTEXT_COMPRESSION_POLICY_VERSION,
            "pack_sha256": self.pack_sha256,
            "context_bytes": self.context_bytes,
            "source_context_bytes": self.source_context_bytes,
            "context_reduction_bytes": self.source_context_bytes
            - self.context_bytes,
            "retained_recent_messages": self.retained_recent_messages,
            "dropped_recent_messages": self.dropped_recent_messages,
            "truncated_recent_messages": self.truncated_recent_messages,
            "truncation": dict(sorted(self.truncation.items())),
            "policy": dict(sorted(self.policy.items())),
        }

    def private_receipt(self) -> dict[str, Any]:
        receipt = self.public_receipt()
        receipt.update(
            {
                "context_sha256": self.context_sha256,
                "source_context_sha256": self.source_context_sha256,
            }
        )
        return receipt


def compress_context(
    *,
    state: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    policy: ContextCompressionPolicy,
) -> CompressedContext:
    """Build one canonical required-state message plus a bounded recent suffix."""

    pack = _validated_state(state)
    truncation = {
        "feature_columns_omitted": 0,
        "tested_hypotheses_omitted": 0,
        "unresolved_errors_omitted": 0,
    }
    _cap_tail(
        pack["tested_hypotheses"],
        policy.max_tested_hypotheses,
        truncation,
        "tested_hypotheses_omitted",
    )
    _cap_tail(
        pack["unresolved_errors"],
        policy.max_unresolved_errors,
        truncation,
        "unresolved_errors_omitted",
    )
    feature_schema = pack["task_contract"]["feature_schema"]
    if len(feature_schema) > policy.max_feature_columns:
        truncation["feature_columns_omitted"] += (
            len(feature_schema) - policy.max_feature_columns
        )
        del feature_schema[policy.max_feature_columns :]

    pack["compression"] = {
        "policy_version": CONTEXT_COMPRESSION_POLICY_VERSION,
        "policy": policy.to_dict(),
        "truncation": truncation,
    }
    pack_message, pack_bytes = _fit_required_pack(pack, policy, truncation)

    normalized_messages = [_normalized_message(message) for message in messages]
    source_context_payload = _canonical_json_bytes(normalized_messages)
    history = normalized_messages[1:]
    if policy.max_recent_messages == 0:
        selected: list[tuple[dict[str, str], bool]] = []
    else:
        selected = [
            _truncate_message(message, policy.max_recent_message_bytes)
            for message in history[-policy.max_recent_messages :]
        ]

    packed_messages = [pack_message] + [message for message, _ in selected]
    while selected and len(_canonical_json_bytes(packed_messages)) > policy.max_context_bytes:
        selected.pop(0)
        packed_messages = [pack_message] + [message for message, _ in selected]
    context_payload = _canonical_json_bytes(packed_messages)
    if len(context_payload) > policy.max_context_bytes:
        raise ContextCompressionError(
            "Required context pack exceeds max_context_bytes"
        )

    retained = len(selected)
    return CompressedContext(
        messages=tuple(dict(message) for message in packed_messages),
        pack_bytes=pack_bytes,
        context_bytes=len(context_payload),
        pack_sha256=hashlib.sha256(pack_bytes).hexdigest(),
        context_sha256=hashlib.sha256(context_payload).hexdigest(),
        source_context_bytes=len(source_context_payload),
        source_context_sha256=hashlib.sha256(source_context_payload).hexdigest(),
        retained_recent_messages=retained,
        dropped_recent_messages=max(0, len(history) - retained),
        truncated_recent_messages=sum(1 for _, truncated in selected if truncated),
        truncation=dict(truncation),
        policy=policy.to_dict(),
    )


def _fit_required_pack(
    pack: dict[str, Any],
    policy: ContextCompressionPolicy,
    truncation: dict[str, int],
) -> tuple[dict[str, str], bytes]:
    while True:
        pack["compression"]["truncation"] = dict(truncation)
        pack_bytes = _canonical_json_bytes(pack)
        message = {"role": "user", "content": _CONTEXT_PREFIX + pack_bytes.decode("utf-8")}
        if len(_canonical_json_bytes([message])) <= policy.max_context_bytes:
            return message, pack_bytes
        if pack["unresolved_errors"]:
            pack["unresolved_errors"].pop(0)
            truncation["unresolved_errors_omitted"] += 1
            continue
        if pack["tested_hypotheses"]:
            pack["tested_hypotheses"].pop(0)
            truncation["tested_hypotheses_omitted"] += 1
            continue
        feature_schema = pack["task_contract"]["feature_schema"]
        if feature_schema:
            feature_schema.pop()
            truncation["feature_columns_omitted"] += 1
            continue
        raise ContextCompressionError(
            "Required context pack exceeds max_context_bytes"
        )


def _validated_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ContextCompressionError("Context pack has an invalid top-level shape")
    try:
        pack = json.loads(_canonical_json_bytes(state).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContextCompressionError("Context pack is not canonical JSON data") from exc
    if pack["schema_version"] != CONTEXT_PACK_SCHEMA_VERSION:
        raise ContextCompressionError("Unsupported context-pack schema")
    task = pack["task_contract"]
    _require_exact_mapping(task, _TASK_KEYS, "task contract")
    for key in ("protocol_version", "target_column", "metric_name"):
        if not isinstance(task[key], str) or not task[key]:
            raise ContextCompressionError(f"Task {key} must be a non-empty string")
    if task["metric_direction"] not in {"higher", "lower"}:
        raise ContextCompressionError("Task metric direction is invalid")
    for key in ("max_steps", "train_rows", "feedback_validation_rows"):
        if type(task[key]) is not int or task[key] < 0:
            raise ContextCompressionError(f"Task {key} must be non-negative")
    if not isinstance(task["feature_schema_sha256"], str) or not _HEX_DIGEST_PATTERN.fullmatch(
        task["feature_schema_sha256"]
    ):
        raise ContextCompressionError("Feature schema hash is invalid")
    for feature in _require_list(task["feature_schema"], "feature schema"):
        _require_exact_mapping(feature, _FEATURE_KEYS, "feature schema entry")
        if not all(isinstance(feature[key], str) for key in _FEATURE_KEYS):
            raise ContextCompressionError("Feature schema values must be strings")
    for hypothesis in _require_list(pack["tested_hypotheses"], "tested hypotheses"):
        _validate_hypothesis(hypothesis)
    incumbent = pack["incumbent_state"]
    if incumbent is not None:
        _validate_incumbent(incumbent)
    for error in _require_list(pack["unresolved_errors"], "unresolved errors"):
        _require_exact_mapping(error, _ERROR_KEYS, "unresolved error")
        if not isinstance(error["action"], str):
            raise ContextCompressionError("Unresolved error action must be a string")
        if error["cell_id"] is not None and not isinstance(error["cell_id"], str):
            raise ContextCompressionError("Unresolved error cell_id is invalid")
        keys = _require_list(error["feedback_keys"], "feedback keys")
        if not all(isinstance(key, str) for key in keys):
            raise ContextCompressionError("Feedback keys must be strings")
    _validate_resources(pack["remaining_resources"])
    actions = _require_list(pack["allowed_actions"], "allowed actions")
    if not all(isinstance(action, str) for action in actions):
        raise ContextCompressionError("Allowed actions must be strings")
    notebook = pack["notebook_state"]
    _require_exact_mapping(notebook, _NOTEBOOK_KEYS, "notebook state")
    for key in (
        "dirty_since_clean_run",
        "clean_run_available",
        "validated_candidate_available",
    ):
        if type(notebook[key]) is not bool:
            raise ContextCompressionError(f"Notebook {key} must be boolean")
    if type(notebook["notebook_revision"]) is not int or notebook["notebook_revision"] < 0:
        raise ContextCompressionError("Notebook revision must be non-negative")
    if notebook["last_clean_run_id"] is not None and not isinstance(
        notebook["last_clean_run_id"], str
    ):
        raise ContextCompressionError("Notebook clean-run ID is invalid")
    if not isinstance(pack["finalization_contract"], str):
        raise ContextCompressionError("Finalization contract must be a string")
    return pack


def _validate_hypothesis(value: Any) -> None:
    _require_exact_mapping(value, _HYPOTHESIS_KEYS, "tested hypothesis")
    if value["action"] not in {"validate", "quick_validate", "check_candidate"}:
        raise ContextCompressionError("Hypothesis action is invalid")
    if value["model_var"] is not None and not isinstance(value["model_var"], str):
        raise ContextCompressionError("Hypothesis model_var is invalid")
    if type(value["notebook_revision"]) is not int or value["notebook_revision"] < 0:
        raise ContextCompressionError("Hypothesis notebook revision is invalid")
    if value["outcome"] not in {"validated", "blocked", "checked"}:
        raise ContextCompressionError("Hypothesis outcome is invalid")
    metric = value["validation_metric"]
    if metric is not None and (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
    ):
        raise ContextCompressionError("Hypothesis validation metric is invalid")
    keys = _require_list(value["feedback_keys"], "hypothesis feedback keys")
    if not all(isinstance(key, str) for key in keys):
        raise ContextCompressionError("Hypothesis feedback keys must be strings")
    if value["candidate_id"] is not None and not isinstance(
        value["candidate_id"], str
    ):
        raise ContextCompressionError("Hypothesis candidate_id is invalid")


def _validate_incumbent(value: Any) -> None:
    _require_exact_mapping(value, _INCUMBENT_KEYS, "incumbent state")
    for key in ("candidate_id", "model_var", "metric_name"):
        if not isinstance(value[key], str) or not value[key]:
            raise ContextCompressionError(f"Hypothesis {key} is invalid")
    if value["metric_direction"] not in {"higher", "lower"}:
        raise ContextCompressionError("Incumbent metric direction is invalid")
    for key in ("notebook_revision", "registration_index"):
        if type(value[key]) is not int or value[key] < 0:
            raise ContextCompressionError(f"Incumbent {key} is invalid")
    metric = value["validation_metric"]
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
    ):
        raise ContextCompressionError("Incumbent validation metric is invalid")
    if type(value["raw_inference_ready"]) is not bool:
        raise ContextCompressionError("Incumbent readiness must be boolean")


def _validate_resources(value: Any) -> None:
    _require_exact_mapping(value, _RESOURCE_KEYS, "remaining resources")
    for key in ("steps_remaining", "step_limit"):
        if type(value[key]) is not int or value[key] < 0:
            raise ContextCompressionError(f"Resource {key} must be non-negative")
    budget = value["episode_budget"]
    if budget is None:
        return
    _require_exact_mapping(budget, _EPISODE_BUDGET_KEYS, "episode budget")
    if budget["phase"] not in {"exploration", "finalization"}:
        raise ContextCompressionError("Episode budget phase is invalid")
    reason = budget["exploration_exhausted_reason"]
    if reason is not None and not isinstance(reason, str):
        raise ContextCompressionError("Exploration exhaustion reason is invalid")
    for key in (
        "global_remaining",
        "exploration_remaining",
        "finalization_remaining",
    ):
        counters = budget[key]
        if counters is None:
            continue
        _require_exact_mapping(counters, _RESOURCE_COUNTER_KEYS, key)
        if not all(type(item) is int and item >= 0 for item in counters.values()):
            raise ContextCompressionError(f"{key} counters must be non-negative integers")
    query_budget = budget["validation_query_budget"]
    if query_budget is None:
        return
    _require_exact_mapping(
        query_budget,
        _VALIDATION_QUERY_BUDGET_KEYS,
        "validation query budget",
    )
    if (
        not isinstance(query_budget["policy_version"], str)
        or not query_budget["policy_version"]
    ):
        raise ContextCompressionError("Validation-query policy version is invalid")
    for key in _VALIDATION_QUERY_BUDGET_KEYS - {"policy_version"}:
        if type(query_budget[key]) is not int or query_budget[key] < 0:
            raise ContextCompressionError(
                f"Validation-query resource {key} must be non-negative"
            )
    maximum = query_budget["max_queries"]
    reserve = query_budget["finalization_reserve_queries"]
    used = query_budget["queries_used"]
    if maximum <= 0 or reserve > maximum or used > maximum:
        raise ContextCompressionError("Validation-query limits are inconsistent")
    if query_budget["feedback_numeric_decimals"] > 15:
        raise ContextCompressionError("Validation feedback precision is invalid")
    if query_budget["global_remaining_queries"] != maximum - used:
        raise ContextCompressionError(
            "Validation-query global remainder is inconsistent"
        )
    if query_budget["phase_remaining_queries"] > maximum:
        raise ContextCompressionError(
            "Validation-query phase remainder is inconsistent"
        )


def _normalized_message(message: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(message, Mapping):
        raise ContextCompressionError("Message history entry must be an object")
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant", "system"} or not isinstance(content, str):
        raise ContextCompressionError("Message history entry is invalid")
    return {"role": role, "content": content}


def _truncate_message(
    message: dict[str, str], max_content_bytes: int
) -> tuple[dict[str, str], bool]:
    content = message["content"]
    encoded = content.encode("utf-8")
    if len(encoded) <= max_content_bytes:
        return message, False
    marker = "...[truncated]"
    marker_bytes = marker.encode("ascii")
    if max_content_bytes < len(marker_bytes):
        truncated = marker_bytes[:max_content_bytes].decode("ascii")
    else:
        prefix = encoded[: max_content_bytes - len(marker_bytes)]
        while True:
            try:
                decoded = prefix.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                prefix = prefix[: exc.start]
        truncated = decoded + marker
    return {"role": message["role"], "content": truncated}, True


def _cap_tail(
    values: list[Any],
    limit: int,
    truncation: dict[str, int],
    counter: str,
) -> None:
    if len(values) <= limit:
        return
    omitted = len(values) - limit
    truncation[counter] += omitted
    if limit == 0:
        values.clear()
    else:
        del values[:omitted]


def _require_exact_mapping(value: Any, keys: frozenset[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContextCompressionError(f"Context {label} has an invalid shape")


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContextCompressionError(f"Context {label} must be a list")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
