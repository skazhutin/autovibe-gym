"""Result-blind semantic contract for the Paper V2 preregistration draft.

The M6 protocol bundle binds document hashes but intentionally does not decide
whether those documents contain an admissible scientific design.  This module
adds that semantic boundary.  It validates a fixed hypothesis hierarchy,
estimands, failure handling, and claim limits while leaving numerical research
decisions unresolved until excluded-development evidence and human review exist.

Nothing in this module activates a runner, selects policy values, writes a
freeze, or inspects experiment outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from research.protocol_v2 import PaperV2ProtocolBundle
from research.run_artifacts import canonical_hash


PREREGISTRATION_V2_VERSION = "paper-v2-preregistration-v1"
PREREGISTRATION_DRAFT_STATUS = "draft"
PREREGISTRATION_FREEZE_CANDIDATE_STATUS = "freeze_candidate"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "study_id",
        "status",
        "confirmatory_outcomes_visible",
        "research_questions",
        "hypotheses",
        "analysis_plan",
        "failure_policy",
        "mechanistic_plan",
        "exploratory_plan",
        "claim_rules",
        "freeze_decisions",
    }
)

_EXPECTED_QUESTIONS = {
    "primary": (
        "Under matched global resources and symmetric terminal handling, is "
        "iterative feedback non-inferior in reliability and superior in "
        "failure-adjusted utility to repeated independent attempts?"
    ),
    "secondary": (
        "Does checklist feedback change failure-adjusted utility beyond the "
        "same iterative runtime, validation, and terminal feedback?"
    ),
}

_EXPECTED_HYPOTHESES = (
    {
        "id": "P1",
        "role": "primary_confirmatory",
        "contrast": "B-A",
        "endpoint": "valid_terminal_outcome",
        "estimand": "equal_stratum_mean_paired_risk_difference",
        "alternative": "noninferiority",
        "decision_rule": "lower_confidence_bound_above_negative_margin",
        "gate": None,
    },
    {
        "id": "P2",
        "role": "gated_confirmatory",
        "contrast": "B-A",
        "endpoint": "failure_adjusted_normalized_utility",
        "estimand": "equal_stratum_mean_paired_difference",
        "alternative": "greater",
        "decision_rule": "adjusted_one_sided_superiority",
        "gate": "P1",
    },
    {
        "id": "S1",
        "role": "key_secondary_confirmatory",
        "contrast": "C-B",
        "endpoint": "failure_adjusted_normalized_utility",
        "estimand": "equal_stratum_mean_paired_difference",
        "alternative": "two_sided",
        "decision_rule": "multiplicity_adjusted_two_sided_difference",
        "gate": None,
    },
)

_ANALYSIS_PLAN = {
    "analysis_population": "all_terminal_agent_outcomes",
    "pairing_keys": ["dataset_id", "model_id", "replicate_index"],
    "stratum_keys": ["dataset_id", "model_id"],
    "stratum_weighting": "equal",
    "primary_sequence": ["P1", "P2"],
    "secondary_family": ["S1"],
    "validity_interval": "paired_stratified_resampling",
    "utility_test": "paired_stratified_randomization",
    "utility_interval": "paired_stratified_resampling",
    "successful_only_analysis": "descriptive_selection_biased",
    "hidden_evaluations_max": 1,
    "unconditional_utility_required": True,
}

_FAILURE_POLICY = {
    "agent_invalid": "retain_as_invalid_and_assign_dummy_utility",
    "infrastructure_failure": "censor_attempt_and_replace_same_condition",
    "protocol_violation": "stop_series_without_outcome_deletion",
    "complete_case_primary_analysis": False,
    "retain_all_attempts": True,
    "replacement_may_change_condition": False,
}

_MECHANISTIC_PLAN = {
    "evidence_class": "separate_excluded_development_study",
    "selection_scope": "excluded-development-only",
    "pool_with_confirmatory_abc": False,
    "design": "nested_incremental_ablation",
    "claim_limit": "incremental_fixed_order_effects_only",
    "arms": [
        {
            "id": "D0",
            "adds": "legacy_iterative_reference",
            "predecessor": None,
        },
        {
            "id": "D1",
            "adds": "immutable_incumbent_preservation",
            "predecessor": "D0",
        },
        {
            "id": "D2",
            "adds": "protected_finalization_reserve",
            "predecessor": "D1",
        },
        {
            "id": "D3",
            "adds": "deterministic_context_compression",
            "predecessor": "D2",
        },
    ],
}

_EXPLORATORY_PLAN = {
    "confirmatory_reclassification_allowed": False,
    "analyses": [
        "dataset_and_model_stratified_effects",
        "task_characteristic_interactions",
        "trajectory_failure_taxonomy",
        "validation_to_hidden_generalization_gap",
        "resource_quality_pareto_frontiers",
    ],
}

_CLAIM_RULES = {
    "system_superiority_requires_confirmatory_abc": True,
    "mechanism_claim_requires_separate_ablation": True,
    "successful_only_score_is_primary": False,
    "checklist_coverage_proves_quality": False,
    "exploratory_results_may_rewrite_confirmatory_claims": False,
}

_REQUIRED_DECISIONS = {
    "noninferiority_margin": "noninferiority_margin",
    "familywise_alpha": "familywise_alpha",
    "target_power": "target_power",
    "replicates_per_stratum": "positive_integer",
    "resampling_draws": "positive_integer",
    "randomization_draws": "positive_integer",
    "analysis_seed": "nonnegative_integer",
    "secondary_multiplicity_strategy": "multiplicity_strategy",
    "dataset_model_scope": "artifact_hash",
    "fanu_references": "artifact_hash",
    "statistical_power_analysis": "artifact_hash",
    "independent_statistical_review": "artifact_hash",
}

_DECISION_KEYS = frozenset({"status", "value", "rationale", "evidence_hash"})
_HYPOTHESIS_KEYS = frozenset(_EXPECTED_HYPOTHESES[0])
_MECHANISTIC_ARM_KEYS = frozenset({"id", "adds", "predecessor"})


class PreregistrationV2Error(ValueError):
    """The Paper V2 preregistration candidate is malformed or not admissible."""


@dataclass(frozen=True)
class PaperV2Preregistration:
    canonical_payload: bytes

    @property
    def preregistration_hash(self) -> str:
        return hashlib.sha256(self.canonical_payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))

    @property
    def status(self) -> str:
        return self.to_dict()["status"]

    def unresolved_decisions(self) -> tuple[str, ...]:
        decisions = self.to_dict()["freeze_decisions"]
        return tuple(
            decision_id
            for decision_id in _REQUIRED_DECISIONS
            if decisions[decision_id]["status"] != "resolved"
        )

    @property
    def freeze_ready(self) -> bool:
        return (
            self.status == PREREGISTRATION_FREEZE_CANDIDATE_STATUS
            and not self.unresolved_decisions()
        )

    def assert_freeze_ready(self) -> None:
        if self.status != PREREGISTRATION_FREEZE_CANDIDATE_STATUS:
            raise PreregistrationV2Error(
                "Preregistration status is not freeze_candidate"
            )
        unresolved = self.unresolved_decisions()
        if unresolved:
            raise PreregistrationV2Error(
                "Freeze-blocking decisions remain unresolved: " + ", ".join(unresolved)
            )

    def component_hashes(self) -> dict[str, str]:
        payload = self.to_dict()
        return {
            "protocol_document_hash": canonical_hash(payload),
            "hypotheses_hash": canonical_hash(payload["hypotheses"]),
            "analysis_plan_hash": canonical_hash(payload["analysis_plan"]),
            "failure_policy_hash": canonical_hash(payload["failure_policy"]),
        }

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": PREREGISTRATION_V2_VERSION,
            "study_id": self.to_dict()["study_id"],
            "status": self.status,
            "preregistration_hash": self.preregistration_hash,
            "component_hashes": self.component_hashes(),
            "freeze_ready": self.freeze_ready,
            "unresolved_decisions": list(self.unresolved_decisions()),
        }

    def verify_protocol_references(
        self, protocol_bundle: PaperV2ProtocolBundle
    ) -> None:
        if not self.freeze_ready:
            raise PreregistrationV2Error(
                "A draft preregistration cannot bind a freeze candidate"
            )
        protocol = protocol_bundle.to_dict()
        if (
            protocol["selection_evidence"]["preregistration_hash"]
            != self.preregistration_hash
        ):
            raise PreregistrationV2Error("Protocol preregistration hash does not match")
        expected = self.component_hashes()
        for key, expected_hash in expected.items():
            if protocol["study_contract"][key] != expected_hash:
                raise PreregistrationV2Error(
                    f"Protocol study-contract hash does not match {key}"
                )


def parse_preregistration_v2(
    payload: Mapping[str, Any],
) -> PaperV2Preregistration:
    root = _exact_mapping(payload, _TOP_LEVEL_KEYS, "preregistration")
    if root["schema_version"] != PREREGISTRATION_V2_VERSION:
        raise PreregistrationV2Error("Unsupported preregistration schema")
    _nonempty_string(root["study_id"], "study_id")
    if root["status"] not in {
        PREREGISTRATION_DRAFT_STATUS,
        PREREGISTRATION_FREEZE_CANDIDATE_STATUS,
    }:
        raise PreregistrationV2Error("Unsupported preregistration status")
    if root["confirmatory_outcomes_visible"] is not False:
        raise PreregistrationV2Error(
            "Preregistration requires confirmatory outcomes to remain unseen"
        )

    _require_exact_value(
        root["research_questions"], _EXPECTED_QUESTIONS, "research_questions"
    )
    _validate_hypotheses(root["hypotheses"])
    _require_exact_value(root["analysis_plan"], _ANALYSIS_PLAN, "analysis_plan")
    _require_exact_value(root["failure_policy"], _FAILURE_POLICY, "failure_policy")
    _validate_mechanistic_plan(root["mechanistic_plan"])
    _require_exact_value(
        root["exploratory_plan"], _EXPLORATORY_PLAN, "exploratory_plan"
    )
    _require_exact_value(root["claim_rules"], _CLAIM_RULES, "claim_rules")
    _validate_freeze_decisions(root["freeze_decisions"])

    canonical_payload = _canonical_json_bytes(root)
    preregistration = PaperV2Preregistration(canonical_payload)
    if (
        preregistration.status == PREREGISTRATION_FREEZE_CANDIDATE_STATUS
        and preregistration.unresolved_decisions()
    ):
        raise PreregistrationV2Error(
            "freeze_candidate status requires every decision to be resolved"
        )
    return preregistration


def load_preregistration_v2(path: str | Path) -> PaperV2Preregistration:
    preregistration_path = Path(path)
    if preregistration_path.is_symlink() or not preregistration_path.is_file():
        raise PreregistrationV2Error("Preregistration path must be a regular file")
    try:
        raw = preregistration_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PreregistrationV2Error("Could not read preregistration") from exc
    return parse_preregistration_v2(_strict_json(raw))


def _validate_hypotheses(value: Any) -> None:
    if not isinstance(value, list) or len(value) != len(_EXPECTED_HYPOTHESES):
        raise PreregistrationV2Error(
            "hypotheses must contain the exact P1, P2, S1 hierarchy"
        )
    normalized = []
    for index, item in enumerate(value):
        normalized.append(
            _exact_mapping(item, _HYPOTHESIS_KEYS, f"hypotheses[{index}]")
        )
    if normalized != list(_EXPECTED_HYPOTHESES):
        raise PreregistrationV2Error(
            "hypotheses differ from the admitted P1, P2, S1 hierarchy"
        )


def _validate_mechanistic_plan(value: Any) -> None:
    _require_exact_value(value, _MECHANISTIC_PLAN, "mechanistic_plan")
    plan = dict(value)
    arms = plan["arms"]
    if any(
        set(_exact_mapping(arm, _MECHANISTIC_ARM_KEYS, "mechanistic arm"))
        != _MECHANISTIC_ARM_KEYS
        for arm in arms
    ):
        raise PreregistrationV2Error("Mechanistic arm fields differ")


def _validate_freeze_decisions(value: Any) -> None:
    decisions = _exact_mapping(
        value, frozenset(_REQUIRED_DECISIONS), "freeze_decisions"
    )
    for decision_id, value_type in _REQUIRED_DECISIONS.items():
        decision = _exact_mapping(
            decisions[decision_id],
            _DECISION_KEYS,
            f"freeze_decisions.{decision_id}",
        )
        status = decision["status"]
        if status == "unresolved":
            if any(
                decision[key] is not None
                for key in ("value", "rationale", "evidence_hash")
            ):
                raise PreregistrationV2Error(
                    f"Unresolved decision {decision_id} must not contain a value"
                )
            continue
        if status != "resolved":
            raise PreregistrationV2Error(
                f"Decision {decision_id} has an unsupported status"
            )
        _validate_decision_value(decision_id, decision["value"], value_type)
        _nonempty_string(
            decision["rationale"], f"freeze_decisions.{decision_id}.rationale"
        )
        _sha256(
            decision["evidence_hash"],
            f"freeze_decisions.{decision_id}.evidence_hash",
        )


def _validate_decision_value(decision_id: str, value: Any, value_type: str) -> None:
    label = f"freeze_decisions.{decision_id}.value"
    if value_type == "noninferiority_margin":
        numeric = _finite_number(value, label)
        if not 0 < numeric < 0.5:
            raise PreregistrationV2Error(f"{label} must be in (0, 0.5)")
        return
    if value_type == "familywise_alpha":
        numeric = _finite_number(value, label)
        if not 0 < numeric <= 0.1:
            raise PreregistrationV2Error(f"{label} must be in (0, 0.1]")
        return
    if value_type == "target_power":
        numeric = _finite_number(value, label)
        if not 0.8 <= numeric < 1:
            raise PreregistrationV2Error(f"{label} must be in [0.8, 1)")
        return
    if value_type == "positive_integer":
        if type(value) is not int or value <= 0:
            raise PreregistrationV2Error(f"{label} must be a positive integer")
        return
    if value_type == "nonnegative_integer":
        if type(value) is not int or value < 0:
            raise PreregistrationV2Error(f"{label} must be a non-negative integer")
        return
    if value_type == "multiplicity_strategy":
        if value not in {
            "hierarchical_gatekeeping_with_secondary_holm",
            "closed_testing",
            "alpha_split_with_gatekeeping",
        }:
            raise PreregistrationV2Error(
                f"{label} is not an admitted multiplicity strategy"
            )
        return
    if value_type == "artifact_hash":
        _sha256(value, label)
        return
    raise AssertionError(f"Unknown decision type: {value_type}")


def _require_exact_value(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise PreregistrationV2Error(f"{label} differs from the admitted design")


def _exact_mapping(
    value: Any, expected_keys: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreregistrationV2Error(f"{label} must be an object")
    raw = dict(value)
    if any(type(key) is not str for key in raw):
        raise PreregistrationV2Error(f"{label} keys must be strings")
    if set(raw) != expected_keys:
        raise PreregistrationV2Error(
            f"{label} fields differ: expected {sorted(expected_keys)}, "
            f"got {sorted(raw)}"
        )
    return raw


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise PreregistrationV2Error(f"{label} must be a non-empty string")
    if value != value.strip():
        raise PreregistrationV2Error(f"{label} must not have surrounding whitespace")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PreregistrationV2Error(f"{label} must be a lowercase SHA-256")
    return text


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreregistrationV2Error(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PreregistrationV2Error(f"{label} must be finite")
    return numeric


def _strict_json(raw: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreregistrationV2Error(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise PreregistrationV2Error(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except PreregistrationV2Error:
        raise
    except json.JSONDecodeError as exc:
        raise PreregistrationV2Error("Preregistration is not valid JSON") from exc


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
        raise PreregistrationV2Error(
            "Preregistration is not canonical-JSON serializable"
        ) from exc
