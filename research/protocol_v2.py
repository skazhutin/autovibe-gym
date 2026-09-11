"""Fail-closed admission contract for a future Paper V2 protocol freeze.

This module binds the separately implemented M1-M5 and VQ1 mechanisms into one
canonical, result-blind configuration.  It deliberately does not activate any
runner, choose any policy value, write a freeze artifact, or inspect outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gym.candidate_store import CANDIDATE_BUNDLE_SCHEMA_VERSION
from gym.context_compression import (
    CONTEXT_COMPRESSION_POLICY_VERSION,
    ContextCompressionPolicy,
)
from gym.finalization import PROTECTED_FINALIZATION_SCHEMA_VERSION
from gym.terminal_contract import TERMINAL_CONTRACT_VERSION
from research.budget import (
    EpisodeBudgetPolicy,
    FinalizationReservePolicy,
    ValidationQueryPolicy,
)
from research.run_artifacts import canonical_hash
from research.stopping import FrozenStoppingPolicy


PROTOCOL_V2_BUNDLE_VERSION = "paper-v2-protocol-bundle-v1"
SELECTION_SCOPE = "excluded-development-only"

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "execution_git_commit",
        "selection_evidence",
        "study_contract",
        "experiment_contract",
        "policies",
        "terminal_contract",
        "execution_contract",
        "arms",
    }
)
_STUDY_KEYS = frozenset(
    {
        "protocol_document_hash",
        "hypotheses_hash",
        "analysis_plan_hash",
        "failure_policy_hash",
    }
)
_EXPERIMENT_KEYS = frozenset(
    {
        "dataset_manifest_hash",
        "model_manifest_hash",
        "prompt_manifest_hash",
        "decoding_config_hash",
        "matrix_specification_hash",
        "randomization_plan_hash",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "selection_scope",
        "confirmatory_outcomes_visible",
        "excluded_task_manifest_hash",
        "policy_selection_plan_hash",
        "preregistration_hash",
        "independent_review_hash",
    }
)
_POLICIES_KEYS = frozenset(
    {
        "episode_budget",
        "finalization_reserve",
        "validation_query",
        "context_compression",
        "stopping",
    }
)
_EPISODE_KEYS = frozenset(
    {
        "total_token_limit",
        "max_output_tokens_per_call",
        "max_llm_calls",
        "max_code_executions",
        "max_tool_calls",
        "wall_clock_limit_seconds",
    }
)
_RESERVE_KEYS = frozenset(
    {
        "total_token_limit",
        "max_llm_calls",
        "max_code_executions",
        "max_tool_calls",
        "wall_clock_limit_seconds",
    }
)
_VALIDATION_QUERY_KEYS = frozenset(
    {
        "policy_version",
        "max_queries",
        "finalization_reserve_queries",
        "feedback_numeric_decimals",
    }
)
_CONTEXT_KEYS = frozenset(
    {
        "policy_version",
        "max_context_bytes",
        "max_recent_messages",
        "max_recent_message_bytes",
        "max_feature_columns",
        "max_tested_hypotheses",
        "max_unresolved_errors",
    }
)
_TERMINAL_KEYS = frozenset(
    {
        "terminal_contract_version",
        "candidate_bundle_version",
        "protected_finalization_version",
        "metric_direction",
        "score_tolerance",
        "hidden_evaluations_max",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "repeated_executor_backend",
        "iterative_kernel_backend",
        "candidate_prediction_backend",
        "candidate_prediction_network",
        "docker_image_digest",
        "sandbox_timeout_seconds",
    }
)
_ARM_KEYS = frozenset(
    {
        "product_mode",
        "runner",
        "terminal_adapter",
        "context_policy",
    }
)
_EXPECTED_ARMS = {
    "A": {
        "product_mode": "repeated_single_shot",
        "runner": "experiments.run_multishot",
        "terminal_adapter": "RepeatedSingleShotTerminalAdapter",
        "context_policy": "not_applicable",
    },
    "B": {
        "product_mode": "iterative_no_checklist",
        "runner": "experiments.run_gym",
        "terminal_adapter": "NotebookGymEnv",
        "context_policy": CONTEXT_COMPRESSION_POLICY_VERSION,
    },
    "C": {
        "product_mode": "gym_with_checklist",
        "runner": "experiments.run_gym",
        "terminal_adapter": "NotebookGymEnv",
        "context_policy": CONTEXT_COMPRESSION_POLICY_VERSION,
    },
}


class ProtocolV2Error(ValueError):
    """A candidate bundle cannot be admitted for Paper V2 review/freeze."""


@dataclass(frozen=True)
class PaperV2ProtocolBundle:
    canonical_payload: bytes
    episode_budget: EpisodeBudgetPolicy
    finalization_reserve: FinalizationReservePolicy
    validation_query: ValidationQueryPolicy
    context_compression: ContextCompressionPolicy
    stopping: FrozenStoppingPolicy

    @property
    def bundle_hash(self) -> str:
        return hashlib.sha256(self.canonical_payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))

    def policy_hashes(self) -> dict[str, str]:
        return {
            "episode_budget": canonical_hash(self.episode_budget.to_dict()),
            "finalization_reserve": canonical_hash(
                self.finalization_reserve.to_dict()
            ),
            "validation_query": canonical_hash(self.validation_query.to_dict()),
            "context_compression": canonical_hash(
                {
                    "policy_version": CONTEXT_COMPRESSION_POLICY_VERSION,
                    **self.context_compression.to_dict(),
                }
            ),
            "stopping": self.stopping.policy_hash,
        }

    def public_receipt(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "schema_version": PROTOCOL_V2_BUNDLE_VERSION,
            "experiment_id": payload["experiment_id"],
            "execution_git_commit": payload["execution_git_commit"],
            "bundle_hash": self.bundle_hash,
            "policy_hashes": self.policy_hashes(),
            "terminal_contract_hash": canonical_hash(
                payload["terminal_contract"]
            ),
            "execution_contract_hash": canonical_hash(
                payload["execution_contract"]
            ),
            "arm_bindings_hash": canonical_hash(payload["arms"]),
            "selection_evidence_hash": canonical_hash(
                payload["selection_evidence"]
            ),
            "study_contract_hash": canonical_hash(payload["study_contract"]),
            "experiment_contract_hash": canonical_hash(
                payload["experiment_contract"]
            ),
        }


def parse_protocol_v2_bundle(payload: Mapping[str, Any]) -> PaperV2ProtocolBundle:
    root = _exact_mapping(payload, _TOP_LEVEL_KEYS, "protocol bundle")
    if root["schema_version"] != PROTOCOL_V2_BUNDLE_VERSION:
        raise ProtocolV2Error("Unsupported Paper V2 protocol-bundle schema")
    _nonempty_string(root["experiment_id"], "experiment_id")
    _require_pattern(
        root["execution_git_commit"], _HEX_40, "execution_git_commit"
    )

    _validate_selection_evidence(root["selection_evidence"])
    _validate_hash_contract(root["study_contract"], _STUDY_KEYS, "study_contract")
    _validate_hash_contract(
        root["experiment_contract"],
        _EXPERIMENT_KEYS,
        "experiment_contract",
    )
    policies = _exact_mapping(root["policies"], _POLICIES_KEYS, "policies")
    episode_budget = _parse_episode_budget(policies["episode_budget"])
    finalization_reserve = _parse_finalization_reserve(
        policies["finalization_reserve"], episode_budget
    )
    validation_query = _parse_validation_query(policies["validation_query"])
    context_compression = _parse_context_compression(
        policies["context_compression"]
    )
    stopping = _parse_stopping(policies["stopping"])

    _validate_terminal_contract(root["terminal_contract"])
    _validate_execution_contract(root["execution_contract"])
    _validate_arm_bindings(root["arms"])
    _validate_cross_policy_contract(
        episode_budget=episode_budget,
        finalization_reserve=finalization_reserve,
        validation_query=validation_query,
    )

    normalized_payload = _normalized_payload(
        root=root,
        episode_budget=episode_budget,
        finalization_reserve=finalization_reserve,
        validation_query=validation_query,
        context_compression=context_compression,
        stopping=stopping,
    )
    canonical_payload = _canonical_json_bytes(normalized_payload)
    return PaperV2ProtocolBundle(
        canonical_payload=canonical_payload,
        episode_budget=episode_budget,
        finalization_reserve=finalization_reserve,
        validation_query=validation_query,
        context_compression=context_compression,
        stopping=stopping,
    )


def load_protocol_v2_bundle(path: str | Path) -> PaperV2ProtocolBundle:
    bundle_path = Path(path)
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ProtocolV2Error("Protocol bundle path must be a regular file")
    try:
        raw = bundle_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolV2Error("Could not read protocol bundle") from exc
    return parse_protocol_v2_bundle(_strict_json(raw))


def _validate_selection_evidence(value: Any) -> None:
    evidence = _exact_mapping(value, _SELECTION_KEYS, "selection_evidence")
    if evidence["selection_scope"] != SELECTION_SCOPE:
        raise ProtocolV2Error(
            "Policy values must be selected on excluded development tasks"
        )
    if evidence["confirmatory_outcomes_visible"] is not False:
        raise ProtocolV2Error(
            "Protocol admission requires confirmatory outcomes to remain unseen"
        )
    for key in (
        "excluded_task_manifest_hash",
        "policy_selection_plan_hash",
        "preregistration_hash",
        "independent_review_hash",
    ):
        _require_pattern(evidence[key], _HEX_64, f"selection_evidence.{key}")


def _validate_hash_contract(
    value: Any,
    expected_keys: frozenset[str],
    label: str,
) -> None:
    contract = _exact_mapping(value, expected_keys, label)
    for key in expected_keys:
        _require_pattern(contract[key], _HEX_64, f"{label}.{key}")


def _parse_episode_budget(value: Any) -> EpisodeBudgetPolicy:
    raw = _exact_mapping(value, _EPISODE_KEYS, "policies.episode_budget")
    for key in _EPISODE_KEYS - {"wall_clock_limit_seconds"}:
        _positive_integer(raw[key], f"policies.episode_budget.{key}")
    _positive_number(
        raw["wall_clock_limit_seconds"],
        "policies.episode_budget.wall_clock_limit_seconds",
    )
    return EpisodeBudgetPolicy(**raw)


def _parse_finalization_reserve(
    value: Any,
    episode_budget: EpisodeBudgetPolicy,
) -> FinalizationReservePolicy:
    raw = _exact_mapping(
        value, _RESERVE_KEYS, "policies.finalization_reserve"
    )
    for key in _RESERVE_KEYS - {"wall_clock_limit_seconds"}:
        _nonnegative_integer(
            raw[key], f"policies.finalization_reserve.{key}"
        )
    _positive_number(
        raw["wall_clock_limit_seconds"],
        "policies.finalization_reserve.wall_clock_limit_seconds",
    )
    try:
        policy = FinalizationReservePolicy(**raw)
        policy.validate_against(episode_budget)
    except ValueError as exc:
        raise ProtocolV2Error(str(exc)) from exc
    return policy


def _parse_validation_query(value: Any) -> ValidationQueryPolicy:
    raw = _exact_mapping(
        value, _VALIDATION_QUERY_KEYS, "policies.validation_query"
    )
    if raw["policy_version"] != "validation-query-v1":
        raise ProtocolV2Error("Unsupported validation-query policy version")
    try:
        return ValidationQueryPolicy(
            max_queries=raw["max_queries"],
            finalization_reserve_queries=raw[
                "finalization_reserve_queries"
            ],
            feedback_numeric_decimals=raw["feedback_numeric_decimals"],
        )
    except ValueError as exc:
        raise ProtocolV2Error(str(exc)) from exc


def _parse_context_compression(value: Any) -> ContextCompressionPolicy:
    raw = _exact_mapping(
        value, _CONTEXT_KEYS, "policies.context_compression"
    )
    if raw["policy_version"] != CONTEXT_COMPRESSION_POLICY_VERSION:
        raise ProtocolV2Error("Unsupported context-compression policy version")
    values = {key: raw[key] for key in _CONTEXT_KEYS if key != "policy_version"}
    try:
        return ContextCompressionPolicy(**values)
    except ValueError as exc:
        raise ProtocolV2Error(str(exc)) from exc


def _parse_stopping(value: Any) -> FrozenStoppingPolicy:
    raw = _exact_mapping(
        value,
        frozenset(
            {
                "schema_version",
                "reserve_boundary_resources",
                "no_improvement_patience",
                "stop_on_exploration_exhausted",
                "stop_on_agent_finalize_request",
                "stop_on_unrecoverable_failure",
            }
        ),
        "policies.stopping",
    )
    try:
        return FrozenStoppingPolicy.from_dict(raw)
    except ValueError as exc:
        raise ProtocolV2Error(str(exc)) from exc


def _validate_terminal_contract(value: Any) -> None:
    terminal = _exact_mapping(value, _TERMINAL_KEYS, "terminal_contract")
    expected_versions = {
        "terminal_contract_version": TERMINAL_CONTRACT_VERSION,
        "candidate_bundle_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
        "protected_finalization_version": PROTECTED_FINALIZATION_SCHEMA_VERSION,
    }
    for key, expected in expected_versions.items():
        if terminal[key] != expected:
            raise ProtocolV2Error(f"terminal_contract.{key} is incompatible")
    if terminal["metric_direction"] not in {"higher", "lower"}:
        raise ProtocolV2Error("terminal_contract.metric_direction is invalid")
    _nonnegative_number(
        terminal["score_tolerance"], "terminal_contract.score_tolerance"
    )
    if terminal["hidden_evaluations_max"] != 1:
        raise ProtocolV2Error(
            "terminal_contract.hidden_evaluations_max must equal one"
        )


def _validate_execution_contract(value: Any) -> None:
    execution = _exact_mapping(value, _EXECUTION_KEYS, "execution_contract")
    expected = {
        "repeated_executor_backend": "docker",
        "iterative_kernel_backend": "docker",
        "candidate_prediction_backend": "docker",
        "candidate_prediction_network": "none",
    }
    for key, expected_value in expected.items():
        if execution[key] != expected_value:
            raise ProtocolV2Error(
                f"execution_contract.{key} must be {expected_value!r}"
            )
    _require_pattern(
        execution["docker_image_digest"],
        _DOCKER_DIGEST,
        "execution_contract.docker_image_digest",
    )
    _positive_integer(
        execution["sandbox_timeout_seconds"],
        "execution_contract.sandbox_timeout_seconds",
    )


def _validate_arm_bindings(value: Any) -> None:
    arms = _exact_mapping(value, frozenset(_EXPECTED_ARMS), "arms")
    for arm_id, expected in _EXPECTED_ARMS.items():
        arm = _exact_mapping(arms[arm_id], _ARM_KEYS, f"arms.{arm_id}")
        if arm != expected:
            raise ProtocolV2Error(
                f"arms.{arm_id} does not match the frozen adapter mapping"
            )


def _validate_cross_policy_contract(
    *,
    episode_budget: EpisodeBudgetPolicy,
    finalization_reserve: FinalizationReservePolicy,
    validation_query: ValidationQueryPolicy,
) -> None:
    if finalization_reserve.max_code_executions < 1:
        raise ProtocolV2Error(
            "M5 protected replay requires at least one reserved code execution"
        )
    if finalization_reserve.max_tool_calls < 3:
        raise ProtocolV2Error(
            "M5 terminal sequence requires at least three reserved tool calls"
        )
    if (
        finalization_reserve.wall_clock_limit_seconds
        >= episode_budget.wall_clock_limit_seconds
    ):
        raise ProtocolV2Error(
            "Finalization wall-clock reserve must leave positive exploration time"
        )
    if validation_query.finalization_reserve_queries < 1:
        raise ProtocolV2Error(
            "M5 protected replay requires a reserved validation query"
        )
    if (
        validation_query.finalization_reserve_queries
        >= validation_query.max_queries
    ):
        raise ProtocolV2Error(
            "Validation-query reserve must leave positive exploration capacity"
        )


def _normalized_payload(
    *,
    root: Mapping[str, Any],
    episode_budget: EpisodeBudgetPolicy,
    finalization_reserve: FinalizationReservePolicy,
    validation_query: ValidationQueryPolicy,
    context_compression: ContextCompressionPolicy,
    stopping: FrozenStoppingPolicy,
) -> dict[str, Any]:
    episode_payload = episode_budget.to_dict()
    episode_payload["wall_clock_limit_seconds"] = float(
        episode_budget.wall_clock_limit_seconds
    )
    reserve_payload = finalization_reserve.to_dict()
    reserve_payload["wall_clock_limit_seconds"] = float(
        finalization_reserve.wall_clock_limit_seconds
    )
    terminal = dict(root["terminal_contract"])
    terminal["score_tolerance"] = float(terminal["score_tolerance"])
    return {
        "schema_version": PROTOCOL_V2_BUNDLE_VERSION,
        "experiment_id": root["experiment_id"],
        "execution_git_commit": root["execution_git_commit"],
        "selection_evidence": dict(root["selection_evidence"]),
        "study_contract": dict(root["study_contract"]),
        "experiment_contract": dict(root["experiment_contract"]),
        "policies": {
            "episode_budget": episode_payload,
            "finalization_reserve": reserve_payload,
            "validation_query": validation_query.to_dict(),
            "context_compression": {
                "policy_version": CONTEXT_COMPRESSION_POLICY_VERSION,
                **context_compression.to_dict(),
            },
            "stopping": stopping.to_dict(),
        },
        "terminal_contract": terminal,
        "execution_contract": dict(root["execution_contract"]),
        "arms": {
            arm_id: dict(binding)
            for arm_id, binding in _EXPECTED_ARMS.items()
        },
    }


def _exact_mapping(
    value: Any,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolV2Error(f"{label} must be an object")
    raw = dict(value)
    if any(type(key) is not str for key in raw):
        raise ProtocolV2Error(f"{label} keys must be strings")
    if set(raw) != expected_keys:
        raise ProtocolV2Error(
            f"{label} fields differ: expected {sorted(expected_keys)}, "
            f"got {sorted(raw)}"
        )
    return raw


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolV2Error(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ProtocolV2Error(f"{label} must not have surrounding whitespace")
    return value


def _require_pattern(value: Any, pattern: re.Pattern[str], label: str) -> str:
    text = _nonempty_string(value, label)
    if pattern.fullmatch(text) is None:
        raise ProtocolV2Error(f"{label} has an invalid format")
    return text


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProtocolV2Error(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolV2Error(f"{label} must be a non-negative integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    numeric = _finite_number(value, label)
    if numeric <= 0:
        raise ProtocolV2Error(f"{label} must be positive")
    return numeric


def _nonnegative_number(value: Any, label: str) -> float:
    numeric = _finite_number(value, label)
    if numeric < 0:
        raise ProtocolV2Error(f"{label} must be non-negative")
    return numeric


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolV2Error(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ProtocolV2Error(f"{label} must be finite")
    return numeric


def _strict_json(raw: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolV2Error(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolV2Error(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except ProtocolV2Error:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolV2Error("Protocol bundle is not valid JSON") from exc


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolV2Error(
            "Protocol bundle is not canonical-JSON serializable"
        ) from exc
