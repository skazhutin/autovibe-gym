"""Planning-only simulation power analysis for AutoVibe Paper V2.

The calculator is intentionally detached from experiment runners and consumes
only explicit, result-blind assumptions.  It approximates the planned paired,
equal-stratum analysis with normal critical bounds so that sample-size and
assumption sensitivity can be reviewed before a confirmatory freeze.  It is not
the confirmatory analysis implementation and cannot establish a scientific
effect from observed outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from research.preregistration_v2 import PaperV2Preregistration, load_preregistration_v2
from research.run_artifacts import canonical_hash


POWER_SCENARIO_VERSION = "paper-v2-power-scenario-v1"
POWER_RESULT_VERSION = "paper-v2-power-result-v1"
POWER_EVIDENCE_CLASS = "planning_simulation_only"
POWER_RNG_ENGINE = "numpy-pcg64-v1"
MULTIPLICITY_STRATEGY = "alpha_split_with_gatekeeping"

_SCENARIO_KEYS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "evidence_class",
        "confirmatory_outcomes_visible",
        "study_contract",
        "testing",
        "simulation",
        "strata",
    }
)
_STUDY_KEYS = frozenset(
    {"hypotheses_hash", "analysis_plan_hash", "failure_policy_hash"}
)
_TESTING_KEYS = frozenset(
    {
        "multiplicity_strategy",
        "familywise_alpha",
        "primary_sequence_alpha",
        "secondary_alpha",
        "noninferiority_margin",
        "target_power",
    }
)
_SIMULATION_KEYS = frozenset({"draws", "seed", "rng_engine"})
_STRATUM_KEYS = frozenset({"stratum_id", "replicates", "validity", "successful_fanu"})
_VALIDITY_KEYS = frozenset(
    {
        "a_rate",
        "b_given_a_valid",
        "b_given_a_invalid",
        "c_given_b_valid",
        "c_given_b_invalid",
    }
)
_FANU_KEYS = frozenset({"means", "standard_deviations", "shared_correlation"})
_ARM_KEYS = frozenset({"A", "B", "C"})
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "scenario_hash",
        "evidence_class",
        "confirmatory_outcomes_visible",
        "study_contract",
        "testing",
        "simulation",
        "runtime_versions",
        "strata",
        "decision_approximation",
        "hypothesis_power",
        "claim_limits",
        "result_hash",
    }
)
_DECISION_APPROXIMATION = {
    "P1": "normal_lower_bound_for_equal_stratum_paired_risk_difference",
    "P2": "gated_normal_lower_bound_for_equal_stratum_paired_fanu_difference",
    "S1": "two_sided_normal_bound_for_equal_stratum_paired_fanu_difference",
}
_CLAIM_LIMITS = [
    "planning_simulation_not_confirmatory_evidence",
    "normal_bound_approximation_not_final_resampling_analysis",
    "power_depends_on_explicit_unverified_assumptions",
    "historical_and_confirmatory_outcomes_not_loaded",
]
_POWER_KEYS = frozenset(
    {
        "successes",
        "draws",
        "estimated_power",
        "monte_carlo_standard_error",
        "monte_carlo_interval_95",
        "target_power",
        "meets_target",
    }
)
_DESCRIPTIVE_POWER_KEYS = frozenset(
    {"eligible_draws", "successes", "estimated_probability", "target_applies"}
)
_RUNTIME_VERSION_KEYS = frozenset({"python", "numpy"})


class PowerV2Error(ValueError):
    """A power scenario or result violates the planning-only contract."""


@dataclass(frozen=True)
class PaperV2PowerScenario:
    canonical_payload: bytes

    @property
    def scenario_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload.decode("utf-8"))

    def public_receipt(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "schema_version": POWER_SCENARIO_VERSION,
            "scenario_id": payload["scenario_id"],
            "scenario_hash": self.scenario_hash,
            "study_contract_hash": canonical_hash(payload["study_contract"]),
            "testing_contract_hash": canonical_hash(payload["testing"]),
            "simulation_contract_hash": canonical_hash(payload["simulation"]),
            "strata_hash": canonical_hash(payload["strata"]),
            "evidence_class": POWER_EVIDENCE_CLASS,
        }

    def verify_preregistration(self, preregistration: PaperV2Preregistration) -> None:
        expected = preregistration.component_hashes()
        actual = self.to_dict()["study_contract"]
        for key in _STUDY_KEYS:
            if actual[key] != expected[key]:
                raise PowerV2Error(
                    f"Power scenario study contract does not match {key}"
                )


def parse_power_scenario(payload: Mapping[str, Any]) -> PaperV2PowerScenario:
    root = _exact_mapping(payload, _SCENARIO_KEYS, "power scenario")
    if root["schema_version"] != POWER_SCENARIO_VERSION:
        raise PowerV2Error("Unsupported power-scenario schema")
    _nonempty_string(root["scenario_id"], "scenario_id")
    if root["evidence_class"] != POWER_EVIDENCE_CLASS:
        raise PowerV2Error("Power scenario must remain planning evidence")
    if root["confirmatory_outcomes_visible"] is not False:
        raise PowerV2Error(
            "Power planning requires confirmatory outcomes to remain unseen"
        )

    study = _exact_mapping(root["study_contract"], _STUDY_KEYS, "study_contract")
    for key in _STUDY_KEYS:
        _sha256(study[key], f"study_contract.{key}")
    _validate_testing(root["testing"])
    _validate_simulation(root["simulation"])
    _validate_strata(root["strata"])
    return PaperV2PowerScenario(_canonical_json_bytes(root))


def load_power_scenario(path: str | Path) -> PaperV2PowerScenario:
    scenario_path = Path(path)
    if scenario_path.is_symlink() or not scenario_path.is_file():
        raise PowerV2Error("Power scenario path must be a regular file")
    try:
        raw = scenario_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PowerV2Error("Could not read power scenario") from exc
    return parse_power_scenario(_strict_json(raw))


def run_power_analysis(
    scenario: PaperV2PowerScenario,
    preregistration: PaperV2Preregistration,
) -> dict[str, Any]:
    """Simulate planning power from explicit assumptions.

    The normal-bound decision approximation is intentionally reported as a
    claim limit.  The eventual confirmatory analysis must remain the separately
    preregistered resampling/randomization pipeline.
    """

    scenario.verify_preregistration(preregistration)
    payload = scenario.to_dict()
    testing = payload["testing"]
    simulation = payload["simulation"]
    rng = np.random.Generator(np.random.PCG64(simulation["seed"]))
    primary_alpha = float(testing["primary_sequence_alpha"])
    secondary_alpha = float(testing["secondary_alpha"])
    noninferiority_margin = float(testing["noninferiority_margin"])
    z_primary = NormalDist().inv_cdf(1.0 - primary_alpha)
    z_secondary = NormalDist().inv_cdf(1.0 - secondary_alpha / 2.0)

    counts = {
        "P1": 0,
        "P2_gated": 0,
        "S1": 0,
        "all_confirmatory": 0,
    }
    p1_eligible = 0
    p2_conditional_successes = 0
    draws = int(simulation["draws"])
    for _ in range(draws):
        validity_differences: list[np.ndarray] = []
        ba_utility_differences: list[np.ndarray] = []
        cb_utility_differences: list[np.ndarray] = []
        for stratum in payload["strata"]:
            a_valid, b_valid, c_valid = _simulate_validity(rng, stratum)
            utilities = _simulate_fanu(
                rng,
                stratum,
                a_valid=a_valid,
                b_valid=b_valid,
                c_valid=c_valid,
            )
            validity_differences.append(b_valid.astype(float) - a_valid.astype(float))
            ba_utility_differences.append(utilities["B"] - utilities["A"])
            cb_utility_differences.append(utilities["C"] - utilities["B"])

        validity_effect, validity_se = _equal_stratum_effect_and_se(
            validity_differences
        )
        ba_effect, ba_se = _equal_stratum_effect_and_se(ba_utility_differences)
        cb_effect, cb_se = _equal_stratum_effect_and_se(cb_utility_differences)

        p1 = validity_effect - z_primary * validity_se > -noninferiority_margin
        p2_raw = ba_effect - z_primary * ba_se > 0.0
        s1 = abs(cb_effect) - z_secondary * cb_se > 0.0
        if p1:
            counts["P1"] += 1
            p1_eligible += 1
            if p2_raw:
                p2_conditional_successes += 1
        p2 = p1 and p2_raw
        if p2:
            counts["P2_gated"] += 1
        if s1:
            counts["S1"] += 1
        if p2 and s1:
            counts["all_confirmatory"] += 1

    target_power = float(testing["target_power"])
    hypothesis_power = {
        hypothesis_id: _power_summary(count, draws, target_power)
        for hypothesis_id, count in counts.items()
    }
    hypothesis_power["P2_given_P1_descriptive"] = {
        "eligible_draws": p1_eligible,
        "successes": p2_conditional_successes,
        "estimated_probability": (
            p2_conditional_successes / p1_eligible if p1_eligible else None
        ),
        "target_applies": False,
    }

    result: dict[str, Any] = {
        "schema_version": POWER_RESULT_VERSION,
        "scenario_id": payload["scenario_id"],
        "scenario_hash": scenario.scenario_hash,
        "evidence_class": POWER_EVIDENCE_CLASS,
        "confirmatory_outcomes_visible": False,
        "study_contract": payload["study_contract"],
        "testing": testing,
        "simulation": simulation,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "strata": payload["strata"],
        "decision_approximation": dict(_DECISION_APPROXIMATION),
        "hypothesis_power": hypothesis_power,
        "claim_limits": list(_CLAIM_LIMITS),
    }
    result["result_hash"] = canonical_hash(result)
    validate_power_result(result)
    return result


def write_power_result(result: Mapping[str, Any], path: str | Path) -> bool:
    """Publish one planning result; identical reruns are idempotent."""

    output = Path(path)
    validate_power_result(result)
    encoded = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
            return True
        except FileExistsError:
            if output.is_symlink() or not output.is_file():
                raise PowerV2Error("Existing power-result path is not a regular file")
            if output.read_bytes() == encoded:
                return False
            raise FileExistsError(f"Refusing to overwrite power result: {output}")
    finally:
        if temporary.exists():
            temporary.unlink()


def _simulate_validity(
    rng: np.random.Generator, stratum: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    replicates = int(stratum["replicates"])
    assumptions = stratum["validity"]
    a_valid = rng.random(replicates) < float(assumptions["a_rate"])
    b_probabilities = np.where(
        a_valid,
        float(assumptions["b_given_a_valid"]),
        float(assumptions["b_given_a_invalid"]),
    )
    b_valid = rng.random(replicates) < b_probabilities
    c_probabilities = np.where(
        b_valid,
        float(assumptions["c_given_b_valid"]),
        float(assumptions["c_given_b_invalid"]),
    )
    c_valid = rng.random(replicates) < c_probabilities
    return a_valid, b_valid, c_valid


def _simulate_fanu(
    rng: np.random.Generator,
    stratum: Mapping[str, Any],
    *,
    a_valid: np.ndarray,
    b_valid: np.ndarray,
    c_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    assumptions = stratum["successful_fanu"]
    means = assumptions["means"]
    deviations = assumptions["standard_deviations"]
    correlation = float(assumptions["shared_correlation"])
    common_scale = math.sqrt(correlation)
    independent_scale = math.sqrt(1.0 - correlation)
    replicates = int(stratum["replicates"])
    common = rng.normal(size=replicates)
    validity = {"A": a_valid, "B": b_valid, "C": c_valid}
    result = {}
    for arm in ("A", "B", "C"):
        successful = float(means[arm]) + float(deviations[arm]) * (
            common_scale * common + independent_scale * rng.normal(size=replicates)
        )
        result[arm] = np.where(validity[arm], successful, 0.0)
    return result


def _equal_stratum_effect_and_se(
    differences: list[np.ndarray],
) -> tuple[float, float]:
    means = np.asarray([float(np.mean(values)) for values in differences])
    variances = np.asarray(
        [float(np.var(values, ddof=1)) / len(values) for values in differences]
    )
    effect = float(np.mean(means))
    standard_error = math.sqrt(float(np.sum(variances))) / len(differences)
    return effect, standard_error


def _power_summary(count: int, draws: int, target_power: float) -> dict[str, Any]:
    estimate = count / draws
    standard_error = math.sqrt(estimate * (1.0 - estimate) / draws)
    radius = NormalDist().inv_cdf(0.975) * standard_error
    return {
        "successes": count,
        "draws": draws,
        "estimated_power": estimate,
        "monte_carlo_standard_error": standard_error,
        "monte_carlo_interval_95": [
            max(0.0, estimate - radius),
            min(1.0, estimate + radius),
        ],
        "target_power": target_power,
        "meets_target": estimate >= target_power,
    }


def _validate_testing(value: Any) -> None:
    testing = _exact_mapping(value, _TESTING_KEYS, "testing")
    if testing["multiplicity_strategy"] != MULTIPLICITY_STRATEGY:
        raise PowerV2Error("Unsupported power multiplicity strategy")
    familywise = _bounded_number(
        testing["familywise_alpha"],
        "testing.familywise_alpha",
        lower=0.0,
        upper=0.1,
        lower_open=True,
    )
    primary = _bounded_number(
        testing["primary_sequence_alpha"],
        "testing.primary_sequence_alpha",
        lower=0.0,
        upper=familywise,
        lower_open=True,
    )
    secondary = _bounded_number(
        testing["secondary_alpha"],
        "testing.secondary_alpha",
        lower=0.0,
        upper=familywise,
        lower_open=True,
    )
    if primary + secondary > familywise + 1e-15:
        raise PowerV2Error(
            "Primary and secondary alpha allocation exceeds family-wise alpha"
        )
    _bounded_number(
        testing["noninferiority_margin"],
        "testing.noninferiority_margin",
        lower=0.0,
        upper=0.5,
        lower_open=True,
        upper_open=True,
    )
    _bounded_number(
        testing["target_power"],
        "testing.target_power",
        lower=0.8,
        upper=1.0,
        upper_open=True,
    )


def _validate_simulation(value: Any) -> None:
    simulation = _exact_mapping(value, _SIMULATION_KEYS, "simulation")
    _bounded_integer(
        simulation["draws"], "simulation.draws", minimum=100, maximum=1_000_000
    )
    _bounded_integer(
        simulation["seed"],
        "simulation.seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    if simulation["rng_engine"] != POWER_RNG_ENGINE:
        raise PowerV2Error("Unsupported power-simulation RNG engine")


def _validate_strata(value: Any) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 1_000:
        raise PowerV2Error("strata must contain between 1 and 1000 entries")
    seen: set[str] = set()
    for index, item in enumerate(value):
        stratum = _exact_mapping(item, _STRATUM_KEYS, f"strata[{index}]")
        stratum_id = _nonempty_string(
            stratum["stratum_id"], f"strata[{index}].stratum_id"
        )
        if stratum_id in seen:
            raise PowerV2Error(f"Duplicate stratum_id: {stratum_id}")
        seen.add(stratum_id)
        _bounded_integer(
            stratum["replicates"],
            f"strata[{index}].replicates",
            minimum=2,
            maximum=100_000,
        )
        validity = _exact_mapping(
            stratum["validity"], _VALIDITY_KEYS, f"strata[{index}].validity"
        )
        for key in _VALIDITY_KEYS:
            _bounded_number(
                validity[key],
                f"strata[{index}].validity.{key}",
                lower=0.0,
                upper=1.0,
            )
        fanu = _exact_mapping(
            stratum["successful_fanu"],
            _FANU_KEYS,
            f"strata[{index}].successful_fanu",
        )
        means = _exact_mapping(
            fanu["means"], _ARM_KEYS, f"strata[{index}].successful_fanu.means"
        )
        deviations = _exact_mapping(
            fanu["standard_deviations"],
            _ARM_KEYS,
            f"strata[{index}].successful_fanu.standard_deviations",
        )
        for arm in _ARM_KEYS:
            _finite_number(means[arm], f"strata[{index}].means.{arm}")
            _bounded_number(
                deviations[arm],
                f"strata[{index}].standard_deviations.{arm}",
                lower=0.0,
                upper=None,
                lower_open=True,
            )
        _bounded_number(
            fanu["shared_correlation"],
            f"strata[{index}].successful_fanu.shared_correlation",
            lower=0.0,
            upper=1.0,
            upper_open=True,
        )


def _validate_result_hash(result: Mapping[str, Any]) -> None:
    raw = dict(result)
    supplied = raw.pop("result_hash", None)
    if supplied != canonical_hash(raw):
        raise PowerV2Error("Power result hash does not match its payload")


def validate_power_result(result: Mapping[str, Any]) -> None:
    root = _exact_mapping(result, _RESULT_KEYS, "power result")
    _validate_result_hash(root)
    if root["schema_version"] != POWER_RESULT_VERSION:
        raise PowerV2Error("Unsupported power-result schema")
    scenario_id = _nonempty_string(root["scenario_id"], "scenario_id")
    _sha256(root["scenario_hash"], "scenario_hash")
    if root["evidence_class"] != POWER_EVIDENCE_CLASS:
        raise PowerV2Error("Power result must remain planning evidence")
    if root["confirmatory_outcomes_visible"] is not False:
        raise PowerV2Error(
            "Power result requires confirmatory outcomes to remain unseen"
        )
    runtime_versions = _exact_mapping(
        root["runtime_versions"], _RUNTIME_VERSION_KEYS, "runtime_versions"
    )
    for key in _RUNTIME_VERSION_KEYS:
        _nonempty_string(runtime_versions[key], f"runtime_versions.{key}")
    scenario_payload = {
        "schema_version": POWER_SCENARIO_VERSION,
        "scenario_id": scenario_id,
        "evidence_class": POWER_EVIDENCE_CLASS,
        "confirmatory_outcomes_visible": False,
        "study_contract": root["study_contract"],
        "testing": root["testing"],
        "simulation": root["simulation"],
        "strata": root["strata"],
    }
    scenario = parse_power_scenario(scenario_payload)
    if scenario.scenario_hash != root["scenario_hash"]:
        raise PowerV2Error("Power result scenario hash does not match assumptions")
    if root["decision_approximation"] != _DECISION_APPROXIMATION:
        raise PowerV2Error("Power result decision approximation differs")
    if root["claim_limits"] != _CLAIM_LIMITS:
        raise PowerV2Error("Power result claim limits differ")

    powers = _exact_mapping(
        root["hypothesis_power"],
        frozenset(
            {
                "P1",
                "P2_gated",
                "S1",
                "all_confirmatory",
                "P2_given_P1_descriptive",
            }
        ),
        "hypothesis_power",
    )
    draws = int(root["simulation"]["draws"])
    target_power = float(root["testing"]["target_power"])
    for hypothesis_id in ("P1", "P2_gated", "S1", "all_confirmatory"):
        _validate_power_summary(
            powers[hypothesis_id],
            f"hypothesis_power.{hypothesis_id}",
            draws=draws,
            target_power=target_power,
        )
    descriptive = _exact_mapping(
        powers["P2_given_P1_descriptive"],
        _DESCRIPTIVE_POWER_KEYS,
        "hypothesis_power.P2_given_P1_descriptive",
    )
    eligible = _bounded_integer(
        descriptive["eligible_draws"],
        "hypothesis_power.P2_given_P1_descriptive.eligible_draws",
        minimum=0,
        maximum=draws,
    )
    successes = _bounded_integer(
        descriptive["successes"],
        "hypothesis_power.P2_given_P1_descriptive.successes",
        minimum=0,
        maximum=eligible,
    )
    expected_probability = successes / eligible if eligible else None
    if descriptive["estimated_probability"] != expected_probability:
        raise PowerV2Error("Descriptive P2 conditional probability is inconsistent")
    if descriptive["target_applies"] is not False:
        raise PowerV2Error("Target power must not apply to descriptive P2 probability")


def _validate_power_summary(
    value: Any,
    label: str,
    *,
    draws: int,
    target_power: float,
) -> None:
    summary = _exact_mapping(value, _POWER_KEYS, label)
    successes = _bounded_integer(
        summary["successes"], f"{label}.successes", minimum=0, maximum=draws
    )
    if summary["draws"] != draws:
        raise PowerV2Error(f"{label}.draws differs from simulation draws")
    expected_power = successes / draws
    if summary["estimated_power"] != expected_power:
        raise PowerV2Error(f"{label}.estimated_power is inconsistent")
    expected_se = math.sqrt(expected_power * (1.0 - expected_power) / draws)
    if not math.isclose(
        _finite_number(summary["monte_carlo_standard_error"], f"{label}.se"),
        expected_se,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise PowerV2Error(f"{label}.monte_carlo_standard_error is inconsistent")
    interval = summary["monte_carlo_interval_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise PowerV2Error(f"{label}.monte_carlo_interval_95 must have two bounds")
    lower = _bounded_number(
        interval[0], f"{label}.interval.lower", lower=0.0, upper=1.0
    )
    upper = _bounded_number(
        interval[1], f"{label}.interval.upper", lower=0.0, upper=1.0
    )
    if lower > upper or not lower <= expected_power <= upper:
        raise PowerV2Error(f"{label}.monte_carlo_interval_95 is inconsistent")
    if summary["target_power"] != target_power:
        raise PowerV2Error(f"{label}.target_power differs from testing target")
    if summary["meets_target"] is not (expected_power >= target_power):
        raise PowerV2Error(f"{label}.meets_target is inconsistent")


def _exact_mapping(
    value: Any, expected_keys: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PowerV2Error(f"{label} must be an object")
    raw = dict(value)
    if any(type(key) is not str for key in raw):
        raise PowerV2Error(f"{label} keys must be strings")
    if set(raw) != expected_keys:
        raise PowerV2Error(
            f"{label} fields differ: expected {sorted(expected_keys)}, "
            f"got {sorted(raw)}"
        )
    return raw


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise PowerV2Error(f"{label} must be a non-empty string")
    if value != value.strip():
        raise PowerV2Error(f"{label} must not have surrounding whitespace")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PowerV2Error(f"{label} must be a lowercase SHA-256")
    return text


def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PowerV2Error(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_number(
    value: Any,
    label: str,
    *,
    lower: float,
    upper: float | None,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    numeric = _finite_number(value, label)
    lower_valid = numeric > lower if lower_open else numeric >= lower
    upper_valid = True
    if upper is not None:
        upper_valid = numeric < upper if upper_open else numeric <= upper
    if not lower_valid or not upper_valid:
        bounds = (
            ("(" if lower_open else "[")
            + str(lower)
            + ", "
            + (str(upper) if upper is not None else "infinity")
            + (")" if upper_open or upper is None else "]")
        )
        raise PowerV2Error(f"{label} must be in {bounds}")
    return numeric


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PowerV2Error(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PowerV2Error(f"{label} must be finite")
    return numeric


def _strict_json(raw: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PowerV2Error(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise PowerV2Error(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except PowerV2Error:
        raise
    except json.JSONDecodeError as exc:
        raise PowerV2Error("Power scenario is not valid JSON") from exc


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
        raise PowerV2Error("Power scenario is not canonical-JSON serializable") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper V2 result-blind planning power simulation"
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        scenario = load_power_scenario(args.scenario)
        preregistration = load_preregistration_v2(args.preregistration)
        result = run_power_analysis(scenario, preregistration)
        created = write_power_result(result, args.output)
    except (PowerV2Error, ValueError, OSError) as exc:
        print(
            json.dumps(
                {"error": "power_analysis_failed", "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "result_hash": result["result_hash"],
                "created": created,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
