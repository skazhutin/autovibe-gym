"""Result-blind evidence provenance for AutoVibe Paper V2 planning.

M9 defines two boundaries that precede any real pilot or power scenario:

* excluded development datasets must be content/lineage-disjoint from the
  future confirmatory scope; and
* every M8 assumption must have an admissible source, declared use, immutable
  artifact hash, and complete coverage across a predeclared scenario set.

This module validates metadata only.  It does not read datasets, run models,
select values, inspect outcomes, or create a freeze.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.power_v2 import PaperV2PowerScenario
from research.preregistration_v2 import PaperV2Preregistration
from research.run_artifacts import canonical_hash


EXCLUDED_DEVELOPMENT_MANIFEST_VERSION = "paper-v2-excluded-development-manifest-v1"
CONFIRMATORY_DATASET_SCOPE_VERSION = "paper-v2-confirmatory-dataset-scope-v1"
ASSUMPTION_EVIDENCE_VERSION = "paper-v2-assumption-evidence-v1"
ASSUMPTION_EVIDENCE_CLASS = "result-blind-planning-evidence"
SELECTION_SCOPE = "excluded-development-only"
SCENARIO_SELECTION_RULE = "minimum_replicates_meeting_target_in_all_required_scenarios"

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_DEVELOPMENT_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "selection_scope",
        "confirmatory_outcomes_visible",
        "tasks",
        "independent_review_hash",
    }
)
_DEVELOPMENT_TASK_KEYS = frozenset(
    {
        "task_id",
        "dataset_identity_hash",
        "dataset_content_hash",
        "source_lineage_hash",
        "split_definition_hash",
        "target_definition_hash",
        "metric_definition_hash",
        "dataset_card_hash",
    }
)
_CONFIRMATORY_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "confirmatory_outcomes_visible",
        "datasets",
        "independent_review_hash",
    }
)
_CONFIRMATORY_DATASET_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_identity_hash",
        "dataset_content_hash",
        "source_lineage_hash",
        "dataset_card_hash",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "package_id",
        "evidence_class",
        "confirmatory_outcomes_visible",
        "development_manifest_hash",
        "scenario_policy",
        "assumptions",
        "independent_review_hash",
    }
)
_SCENARIO_POLICY_KEYS = frozenset(
    {
        "selection_rule",
        "favorable_row_selection_allowed",
        "scenario_grid_plan_hash",
        "complete_grid_review_hash",
        "required_scenarios",
    }
)
_SCENARIO_REFERENCE_KEYS = frozenset({"scenario_id", "scenario_hash"})
_ASSUMPTION_KEYS = frozenset(
    {
        "scenario_id",
        "scenario_hash",
        "assumption_key",
        "assumption_value_hash",
        "use_class",
        "source_class",
        "source_artifact_hash",
        "development_manifest_hash",
    }
)

_SOURCE_CLASSES = frozenset(
    {
        "independent_methodological_justification",
        "excluded_development_pilot",
        "historical_paper_v1",
        "external_primary_source",
        "deterministic_design_choice",
        "power_grid_design",
    }
)
_USE_CLASSES = frozenset(
    {
        "scientific_acceptability_threshold",
        "statistical_design_choice",
        "reproducibility_choice",
        "candidate_sample_size",
        "nuisance_parameter",
    }
)

_TESTING_USE = {
    "/testing/multiplicity_strategy": "statistical_design_choice",
    "/testing/familywise_alpha": "statistical_design_choice",
    "/testing/primary_sequence_alpha": "statistical_design_choice",
    "/testing/secondary_alpha": "statistical_design_choice",
    "/testing/noninferiority_margin": "scientific_acceptability_threshold",
    "/testing/target_power": "statistical_design_choice",
}
_SIMULATION_USE = {
    "/simulation/draws": "reproducibility_choice",
    "/simulation/seed": "reproducibility_choice",
    "/simulation/rng_engine": "reproducibility_choice",
}
_ALLOWED_SOURCES = {
    "scientific_acceptability_threshold": frozenset(
        {
            "independent_methodological_justification",
            "external_primary_source",
        }
    ),
    "statistical_design_choice": frozenset(
        {
            "independent_methodological_justification",
            "external_primary_source",
        }
    ),
    "reproducibility_choice": frozenset(
        {
            "deterministic_design_choice",
            "independent_methodological_justification",
        }
    ),
    "candidate_sample_size": frozenset({"power_grid_design"}),
    "nuisance_parameter": frozenset(
        {
            "excluded_development_pilot",
            "historical_paper_v1",
            "external_primary_source",
        }
    ),
}


class EvidenceV2Error(ValueError):
    """An M9 metadata artifact violates the result-blind evidence contract."""


@dataclass(frozen=True)
class ExcludedDevelopmentManifest:
    canonical_payload: bytes

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))

    def public_receipt(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "schema_version": EXCLUDED_DEVELOPMENT_MANIFEST_VERSION,
            "manifest_id": payload["manifest_id"],
            "manifest_hash": self.manifest_hash,
            "task_count": len(payload["tasks"]),
            "task_set_hash": canonical_hash(payload["tasks"]),
            "independent_review_hash": payload["independent_review_hash"],
            "selection_scope": SELECTION_SCOPE,
        }

    def assert_disjoint(self, confirmatory: ConfirmatoryDatasetScope) -> None:
        development = self.to_dict()["tasks"]
        frozen = confirmatory.to_dict()["datasets"]
        for key, label in (
            ("dataset_identity_hash", "dataset identity"),
            ("dataset_content_hash", "dataset content"),
            ("source_lineage_hash", "source lineage"),
        ):
            development_values = {item[key] for item in development}
            confirmatory_values = {item[key] for item in frozen}
            if development_values & confirmatory_values:
                raise EvidenceV2Error(
                    f"Excluded-development and confirmatory {label} overlap"
                )


@dataclass(frozen=True)
class ConfirmatoryDatasetScope:
    canonical_payload: bytes

    @property
    def scope_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))


@dataclass(frozen=True)
class AssumptionEvidencePackage:
    canonical_payload: bytes

    @property
    def package_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))

    def public_receipt(self) -> dict[str, Any]:
        payload = self.to_dict()
        policy = payload["scenario_policy"]
        return {
            "schema_version": ASSUMPTION_EVIDENCE_VERSION,
            "package_id": payload["package_id"],
            "package_hash": self.package_hash,
            "development_manifest_hash": payload["development_manifest_hash"],
            "scenario_policy_hash": canonical_hash(policy),
            "assumption_registry_hash": canonical_hash(payload["assumptions"]),
            "scenario_count": len(policy["required_scenarios"]),
            "assumption_count": len(payload["assumptions"]),
            "independent_review_hash": payload["independent_review_hash"],
            "evidence_class": ASSUMPTION_EVIDENCE_CLASS,
        }

    def verify(
        self,
        *,
        development_manifest: ExcludedDevelopmentManifest,
        preregistration: PaperV2Preregistration,
        scenarios: Sequence[PaperV2PowerScenario],
    ) -> None:
        payload = self.to_dict()
        if payload["development_manifest_hash"] != development_manifest.manifest_hash:
            raise EvidenceV2Error(
                "Assumption evidence does not match the development manifest"
            )
        if not scenarios:
            raise EvidenceV2Error("At least one required power scenario is needed")

        actual_scenarios: dict[str, str] = {}
        expected_assumptions: dict[tuple[str, str], tuple[str, str]] = {}
        for scenario in scenarios:
            scenario.verify_preregistration(preregistration)
            scenario_payload = scenario.to_dict()
            scenario_id = scenario_payload["scenario_id"]
            if scenario_id in actual_scenarios:
                raise EvidenceV2Error(f"Duplicate supplied scenario_id: {scenario_id}")
            actual_scenarios[scenario_id] = scenario.scenario_hash
            for assumption_key, value in _scenario_assumption_values(scenario).items():
                expected_assumptions[(scenario_id, assumption_key)] = (
                    scenario.scenario_hash,
                    canonical_hash(value),
                )

        required = {
            item["scenario_id"]: item["scenario_hash"]
            for item in payload["scenario_policy"]["required_scenarios"]
        }
        if required != actual_scenarios:
            raise EvidenceV2Error(
                "Supplied scenarios differ from the predeclared required set"
            )

        records = {
            (item["scenario_id"], item["assumption_key"]): item
            for item in payload["assumptions"]
        }
        if set(records) != set(expected_assumptions):
            missing = sorted(set(expected_assumptions) - set(records))
            extra = sorted(set(records) - set(expected_assumptions))
            raise EvidenceV2Error(
                f"Assumption provenance coverage differs: missing={missing}, extra={extra}"
            )
        for (scenario_id, assumption_key), record in records.items():
            expected_scenario_hash, expected_value_hash = expected_assumptions[
                (scenario_id, assumption_key)
            ]
            if record["scenario_hash"] != expected_scenario_hash:
                raise EvidenceV2Error(
                    f"Assumption {scenario_id}:{assumption_key} has scenario hash drift"
                )
            if record["assumption_value_hash"] != expected_value_hash:
                raise EvidenceV2Error(
                    f"Assumption {scenario_id}:{assumption_key} has value hash drift"
                )
            expected_use = _assumption_use_class(assumption_key)
            if record["use_class"] != expected_use:
                raise EvidenceV2Error(
                    f"Assumption {scenario_id}:{assumption_key} has the wrong use class"
                )
            if record["source_class"] not in _ALLOWED_SOURCES[expected_use]:
                raise EvidenceV2Error(
                    f"Assumption {scenario_id}:{assumption_key} has an "
                    "inadmissible source class"
                )


def parse_excluded_development_manifest(
    payload: Mapping[str, Any],
) -> ExcludedDevelopmentManifest:
    root = _exact_mapping(payload, _DEVELOPMENT_KEYS, "development manifest")
    if root["schema_version"] != EXCLUDED_DEVELOPMENT_MANIFEST_VERSION:
        raise EvidenceV2Error("Unsupported development-manifest schema")
    _identifier(root["manifest_id"], "manifest_id")
    if root["selection_scope"] != SELECTION_SCOPE:
        raise EvidenceV2Error("Development manifest must remain excluded-only")
    if root["confirmatory_outcomes_visible"] is not False:
        raise EvidenceV2Error(
            "Development selection requires confirmatory outcomes to remain unseen"
        )
    tasks = _bounded_list(root["tasks"], "tasks", minimum=1, maximum=1_000)
    seen_ids: set[str] = set()
    seen_definitions: set[tuple[str, ...]] = set()
    for index, value in enumerate(tasks):
        task = _exact_mapping(value, _DEVELOPMENT_TASK_KEYS, f"tasks[{index}]")
        task_id = _identifier(task["task_id"], f"tasks[{index}].task_id")
        if task_id in seen_ids:
            raise EvidenceV2Error(f"Duplicate development task_id: {task_id}")
        seen_ids.add(task_id)
        for key in _DEVELOPMENT_TASK_KEYS - {"task_id"}:
            _sha256(task[key], f"tasks[{index}].{key}")
        definition = tuple(
            task[key]
            for key in (
                "dataset_content_hash",
                "split_definition_hash",
                "target_definition_hash",
                "metric_definition_hash",
            )
        )
        if definition in seen_definitions:
            raise EvidenceV2Error("Duplicate development task definition")
        seen_definitions.add(definition)
    _sha256(root["independent_review_hash"], "independent_review_hash")
    return ExcludedDevelopmentManifest(_canonical_json_bytes(root))


def parse_confirmatory_dataset_scope(
    payload: Mapping[str, Any],
) -> ConfirmatoryDatasetScope:
    root = _exact_mapping(payload, _CONFIRMATORY_KEYS, "confirmatory scope")
    if root["schema_version"] != CONFIRMATORY_DATASET_SCOPE_VERSION:
        raise EvidenceV2Error("Unsupported confirmatory-scope schema")
    _identifier(root["manifest_id"], "manifest_id")
    if root["confirmatory_outcomes_visible"] is not False:
        raise EvidenceV2Error(
            "Confirmatory scope must be defined before outcomes are visible"
        )
    datasets = _bounded_list(root["datasets"], "datasets", minimum=1, maximum=1_000)
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for index, value in enumerate(datasets):
        dataset = _exact_mapping(
            value, _CONFIRMATORY_DATASET_KEYS, f"datasets[{index}]"
        )
        dataset_id = _identifier(dataset["dataset_id"], f"datasets[{index}].dataset_id")
        if dataset_id in seen_ids:
            raise EvidenceV2Error(f"Duplicate confirmatory dataset_id: {dataset_id}")
        seen_ids.add(dataset_id)
        for key in _CONFIRMATORY_DATASET_KEYS - {"dataset_id"}:
            _sha256(dataset[key], f"datasets[{index}].{key}")
        if dataset["dataset_content_hash"] in seen_content:
            raise EvidenceV2Error("Duplicate confirmatory dataset content")
        seen_content.add(dataset["dataset_content_hash"])
    _sha256(root["independent_review_hash"], "independent_review_hash")
    return ConfirmatoryDatasetScope(_canonical_json_bytes(root))


def parse_assumption_evidence_package(
    payload: Mapping[str, Any],
) -> AssumptionEvidencePackage:
    root = _exact_mapping(payload, _EVIDENCE_KEYS, "assumption evidence")
    if root["schema_version"] != ASSUMPTION_EVIDENCE_VERSION:
        raise EvidenceV2Error("Unsupported assumption-evidence schema")
    _identifier(root["package_id"], "package_id")
    if root["evidence_class"] != ASSUMPTION_EVIDENCE_CLASS:
        raise EvidenceV2Error("Assumption package has the wrong evidence class")
    if root["confirmatory_outcomes_visible"] is not False:
        raise EvidenceV2Error(
            "Assumption selection requires confirmatory outcomes to remain unseen"
        )
    development_hash = _sha256(
        root["development_manifest_hash"], "development_manifest_hash"
    )
    policy = _exact_mapping(
        root["scenario_policy"], _SCENARIO_POLICY_KEYS, "scenario_policy"
    )
    if policy["selection_rule"] != SCENARIO_SELECTION_RULE:
        raise EvidenceV2Error("Scenario selection rule is not robust-all-scenarios")
    if policy["favorable_row_selection_allowed"] is not False:
        raise EvidenceV2Error("Favorable-row scenario selection is forbidden")
    _sha256(policy["scenario_grid_plan_hash"], "scenario_grid_plan_hash")
    _sha256(policy["complete_grid_review_hash"], "complete_grid_review_hash")
    references = _bounded_list(
        policy["required_scenarios"],
        "required_scenarios",
        minimum=1,
        maximum=10_000,
    )
    seen_scenarios: set[str] = set()
    for index, value in enumerate(references):
        reference = _exact_mapping(
            value, _SCENARIO_REFERENCE_KEYS, f"required_scenarios[{index}]"
        )
        scenario_id = _identifier(
            reference["scenario_id"], f"required_scenarios[{index}].scenario_id"
        )
        if scenario_id in seen_scenarios:
            raise EvidenceV2Error(f"Duplicate required scenario_id: {scenario_id}")
        seen_scenarios.add(scenario_id)
        _sha256(
            reference["scenario_hash"],
            f"required_scenarios[{index}].scenario_hash",
        )

    assumptions = _bounded_list(
        root["assumptions"], "assumptions", minimum=1, maximum=1_000_000
    )
    seen_assumptions: set[tuple[str, str]] = set()
    for index, value in enumerate(assumptions):
        record = _exact_mapping(value, _ASSUMPTION_KEYS, f"assumptions[{index}]")
        scenario_id = _identifier(
            record["scenario_id"], f"assumptions[{index}].scenario_id"
        )
        _sha256(record["scenario_hash"], f"assumptions[{index}].scenario_hash")
        assumption_key = _assumption_key(
            record["assumption_key"], f"assumptions[{index}].assumption_key"
        )
        assumption_identity = (scenario_id, assumption_key)
        if assumption_identity in seen_assumptions:
            raise EvidenceV2Error(
                f"Duplicate scenario assumption: {scenario_id}:{assumption_key}"
            )
        seen_assumptions.add(assumption_identity)
        _sha256(
            record["assumption_value_hash"],
            f"assumptions[{index}].assumption_value_hash",
        )
        if record["use_class"] not in _USE_CLASSES:
            raise EvidenceV2Error(f"Unknown use class for {assumption_key}")
        if record["source_class"] not in _SOURCE_CLASSES:
            raise EvidenceV2Error(f"Unknown source class for {assumption_key}")
        _sha256(
            record["source_artifact_hash"],
            f"assumptions[{index}].source_artifact_hash",
        )
        if record["source_class"] == "excluded_development_pilot":
            if record["development_manifest_hash"] != development_hash:
                raise EvidenceV2Error(
                    f"Excluded-pilot assumption {assumption_key} is not bound "
                    "to the development manifest"
                )
        elif record["development_manifest_hash"] is not None:
            raise EvidenceV2Error(
                f"Non-pilot assumption {assumption_key} must not claim a "
                "development manifest"
            )
    _sha256(root["independent_review_hash"], "independent_review_hash")
    return AssumptionEvidencePackage(_canonical_json_bytes(root))


def load_excluded_development_manifest(
    path: str | Path,
) -> ExcludedDevelopmentManifest:
    return parse_excluded_development_manifest(
        _load_strict_json_file(path, "Development manifest")
    )


def load_confirmatory_dataset_scope(path: str | Path) -> ConfirmatoryDatasetScope:
    return parse_confirmatory_dataset_scope(
        _load_strict_json_file(path, "Confirmatory scope")
    )


def load_assumption_evidence_package(path: str | Path) -> AssumptionEvidencePackage:
    return parse_assumption_evidence_package(
        _load_strict_json_file(path, "Assumption evidence")
    )


def _scenario_assumption_values(scenario: PaperV2PowerScenario) -> dict[str, Any]:
    payload = scenario.to_dict()
    assumptions = {
        key: payload[section][key.rsplit("/", 1)[-1]]
        for section, keys in (
            ("testing", _TESTING_USE),
            ("simulation", _SIMULATION_USE),
        )
        for key in keys
    }
    for stratum in payload["strata"]:
        stratum_id = _json_pointer_escape(stratum["stratum_id"])
        prefix = f"/strata/by-id/{stratum_id}"
        assumptions[f"{prefix}/replicates"] = stratum["replicates"]
        for key in (
            "a_rate",
            "b_given_a_valid",
            "b_given_a_invalid",
            "c_given_b_valid",
            "c_given_b_invalid",
        ):
            assumptions[f"{prefix}/validity/{key}"] = stratum["validity"][key]
        for arm in ("A", "B", "C"):
            assumptions[f"{prefix}/successful_fanu/means/{arm}"] = stratum[
                "successful_fanu"
            ]["means"][arm]
            assumptions[f"{prefix}/successful_fanu/standard_deviations/{arm}"] = (
                stratum["successful_fanu"]["standard_deviations"][arm]
            )
        assumptions[f"{prefix}/successful_fanu/shared_correlation"] = stratum[
            "successful_fanu"
        ]["shared_correlation"]
    return assumptions


def _assumption_use_class(assumption_key: str) -> str:
    if assumption_key in _TESTING_USE:
        return _TESTING_USE[assumption_key]
    if assumption_key in _SIMULATION_USE:
        return _SIMULATION_USE[assumption_key]
    if re.fullmatch(r"/strata/by-id/[^/]+/replicates", assumption_key):
        return "candidate_sample_size"
    if re.fullmatch(
        r"/strata/by-id/[^/]+/(validity/[^/]+|successful_fanu/[^/]+(?:/[^/]+)?)",
        assumption_key,
    ):
        return "nuisance_parameter"
    raise EvidenceV2Error(f"Unknown scenario assumption key: {assumption_key}")


def _load_strict_json_file(path: str | Path, label: str) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise EvidenceV2Error(f"{label} path must be a regular file")
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceV2Error(f"Could not read {label.lower()}") from exc
    return _strict_json(raw)


def _bounded_list(value: Any, label: str, *, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise EvidenceV2Error(
            f"{label} must contain between {minimum} and {maximum} items"
        )
    return value


def _exact_mapping(
    value: Any, expected_keys: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceV2Error(f"{label} must be an object")
    raw = dict(value)
    if any(type(key) is not str for key in raw):
        raise EvidenceV2Error(f"{label} keys must be strings")
    if set(raw) != expected_keys:
        raise EvidenceV2Error(
            f"{label} fields differ: expected {sorted(expected_keys)}, "
            f"got {sorted(raw)}"
        )
    return raw


def _identifier(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if _ID.fullmatch(text) is None:
        raise EvidenceV2Error(f"{label} is not a portable identifier")
    return text


def _assumption_key(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if not text.startswith("/") or "//" in text:
        raise EvidenceV2Error(f"{label} must be a canonical assumption pointer")
    return text


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise EvidenceV2Error(f"{label} must be a non-empty string")
    if value != value.strip():
        raise EvidenceV2Error(f"{label} must not have surrounding whitespace")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise EvidenceV2Error(f"{label} must be a lowercase SHA-256")
    return text


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _strict_json(raw: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceV2Error(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise EvidenceV2Error(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except EvidenceV2Error:
        raise
    except json.JSONDecodeError as exc:
        raise EvidenceV2Error("Evidence artifact is not valid JSON") from exc


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
        raise EvidenceV2Error(
            "Evidence artifact is not canonical-JSON serializable"
        ) from exc
