from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.power_v2 import (
    MULTIPLICITY_STRATEGY,
    POWER_EVIDENCE_CLASS,
    POWER_RNG_ENGINE,
    POWER_SCENARIO_VERSION,
    PowerV2Error,
    load_power_scenario,
    parse_power_scenario,
    run_power_analysis,
    write_power_result,
)
from research.preregistration_v2 import load_preregistration_v2
from research.run_artifacts import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "research/protocols/paper_v2/preregistration.draft.json"


def _preregistration():
    return load_preregistration_v2(PREREGISTRATION_PATH)


def _scenario_payload():
    hashes = _preregistration().component_hashes()
    return {
        "schema_version": POWER_SCENARIO_VERSION,
        "scenario_id": "synthetic-power-fixture",
        "evidence_class": POWER_EVIDENCE_CLASS,
        "confirmatory_outcomes_visible": False,
        "study_contract": {
            "hypotheses_hash": hashes["hypotheses_hash"],
            "analysis_plan_hash": hashes["analysis_plan_hash"],
            "failure_policy_hash": hashes["failure_policy_hash"],
        },
        "testing": {
            "multiplicity_strategy": MULTIPLICITY_STRATEGY,
            "familywise_alpha": 0.05,
            "primary_sequence_alpha": 0.04,
            "secondary_alpha": 0.01,
            "noninferiority_margin": 0.1,
            "target_power": 0.8,
        },
        "simulation": {
            "draws": 100,
            "seed": 17,
            "rng_engine": POWER_RNG_ENGINE,
        },
        "strata": [
            {
                "stratum_id": "synthetic-dataset-model",
                "replicates": 20,
                "validity": {
                    "a_rate": 0.7,
                    "b_given_a_valid": 0.9,
                    "b_given_a_invalid": 0.5,
                    "c_given_b_valid": 0.9,
                    "c_given_b_invalid": 0.5,
                },
                "successful_fanu": {
                    "means": {"A": 0.5, "B": 0.7, "C": 0.75},
                    "standard_deviations": {"A": 0.2, "B": 0.2, "C": 0.2},
                    "shared_correlation": 0.5,
                },
            }
        ],
    }


def _extreme_scenario_payload(*, favorable: bool):
    payload = _scenario_payload()
    validity = payload["strata"][0]["validity"]
    means = payload["strata"][0]["successful_fanu"]["means"]
    deviations = payload["strata"][0]["successful_fanu"]["standard_deviations"]
    if favorable:
        validity.update(
            {
                "a_rate": 0.0,
                "b_given_a_valid": 1.0,
                "b_given_a_invalid": 1.0,
                "c_given_b_valid": 1.0,
                "c_given_b_invalid": 1.0,
            }
        )
        means.update({"A": 0.0, "B": 2.0, "C": 4.0})
    else:
        validity.update(
            {
                "a_rate": 1.0,
                "b_given_a_valid": 0.0,
                "b_given_a_invalid": 0.0,
                "c_given_b_valid": 0.0,
                "c_given_b_invalid": 0.0,
            }
        )
        means.update({"A": 2.0, "B": 0.0, "C": 0.0})
    deviations.update({"A": 0.001, "B": 0.001, "C": 0.001})
    return payload


def test_power_scenario_is_canonical_hash_stable_and_receipted():
    payload = _scenario_payload()
    reordered = dict(reversed(list(payload.items())))

    first = parse_power_scenario(payload)
    second = parse_power_scenario(reordered)

    assert first.canonical_payload == second.canonical_payload
    assert first.scenario_hash == second.scenario_hash
    receipt = first.public_receipt()
    assert receipt["scenario_hash"] == first.scenario_hash
    assert receipt["evidence_class"] == POWER_EVIDENCE_CLASS
    assert len(receipt["study_contract_hash"]) == 64
    first.verify_preregistration(_preregistration())


def test_power_scenario_rejects_preregistration_hash_drift():
    payload = _scenario_payload()
    payload["study_contract"]["analysis_plan_hash"] = "f" * 64
    scenario = parse_power_scenario(payload)

    with pytest.raises(PowerV2Error, match="analysis_plan_hash"):
        scenario.verify_preregistration(_preregistration())


def test_power_analysis_is_deterministic_and_hash_bound():
    scenario = parse_power_scenario(_scenario_payload())

    first = run_power_analysis(scenario, _preregistration())
    second = run_power_analysis(scenario, _preregistration())

    assert first == second
    assert first["scenario_hash"] == scenario.scenario_hash
    assert first["evidence_class"] == POWER_EVIDENCE_CLASS
    assert first["confirmatory_outcomes_visible"] is False
    assert set(first["runtime_versions"]) == {"python", "numpy"}
    result_hash = first["result_hash"]
    unhashed = dict(first)
    unhashed.pop("result_hash")
    assert result_hash == canonical_hash(unhashed)
    assert first["claim_limits"] == [
        "planning_simulation_not_confirmatory_evidence",
        "normal_bound_approximation_not_final_resampling_analysis",
        "power_depends_on_explicit_unverified_assumptions",
        "historical_and_confirmatory_outcomes_not_loaded",
    ]


def test_extreme_synthetic_scenarios_exercise_gate_and_secondary_test():
    favorable = run_power_analysis(
        parse_power_scenario(_extreme_scenario_payload(favorable=True)),
        _preregistration(),
    )
    assert favorable["hypothesis_power"]["P1"]["estimated_power"] == 1.0
    assert favorable["hypothesis_power"]["P2_gated"]["estimated_power"] == 1.0
    assert favorable["hypothesis_power"]["S1"]["estimated_power"] == 1.0
    assert favorable["hypothesis_power"]["all_confirmatory"]["estimated_power"] == 1.0

    unfavorable = run_power_analysis(
        parse_power_scenario(_extreme_scenario_payload(favorable=False)),
        _preregistration(),
    )
    assert unfavorable["hypothesis_power"]["P1"]["estimated_power"] == 0.0
    assert unfavorable["hypothesis_power"]["P2_gated"]["estimated_power"] == 0.0
    assert (
        unfavorable["hypothesis_power"]["P2_given_P1_descriptive"][
            "estimated_probability"
        ]
        is None
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"confirmatory_outcomes_visible": True}),
            "outcomes to remain unseen",
        ),
        (
            lambda payload: payload.update({"evidence_class": "confirmatory"}),
            "planning evidence",
        ),
        (
            lambda payload: payload["testing"].update(
                {"primary_sequence_alpha": 0.05, "secondary_alpha": 0.01}
            ),
            "exceeds family-wise alpha",
        ),
        (
            lambda payload: payload["testing"].update(
                {"multiplicity_strategy": "uncorrected"}
            ),
            "multiplicity strategy",
        ),
        (
            lambda payload: payload["simulation"].update({"draws": 99}),
            "simulation.draws",
        ),
        (
            lambda payload: payload["simulation"].update({"seed": True}),
            "simulation.seed",
        ),
        (
            lambda payload: payload["strata"][0]["validity"].update({"a_rate": 1.1}),
            "a_rate",
        ),
        (
            lambda payload: payload["strata"][0]["successful_fanu"].update(
                {"shared_correlation": 1.0}
            ),
            "shared_correlation",
        ),
        (
            lambda payload: payload["strata"][0].update({"replicates": 1}),
            "replicates",
        ),
        (
            lambda payload: payload["strata"].append(
                copy.deepcopy(payload["strata"][0])
            ),
            "Duplicate stratum_id",
        ),
    ],
)
def test_power_scenario_rejects_visibility_multiplicity_and_assumption_drift(
    mutate, message
):
    payload = _scenario_payload()
    mutate(payload)

    with pytest.raises(PowerV2Error, match=message):
        parse_power_scenario(payload)


def test_power_scenario_loader_rejects_duplicate_and_unknown_fields(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(_scenario_payload()), encoding="utf-8")
    assert load_power_scenario(path).to_dict() == _scenario_payload()

    raw = json.dumps(_scenario_payload())
    path.write_text(
        raw.replace(
            '"schema_version": "paper-v2-power-scenario-v1",',
            '"schema_version": "paper-v2-power-scenario-v1", '
            '"schema_version": "paper-v2-power-scenario-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PowerV2Error, match="Duplicate JSON key"):
        load_power_scenario(path)

    unknown = _scenario_payload()
    unknown["selected_after_results"] = True
    with pytest.raises(PowerV2Error, match="fields differ"):
        parse_power_scenario(unknown)


def test_power_result_write_is_no_overwrite_and_hash_checked(tmp_path):
    scenario = parse_power_scenario(_scenario_payload())
    result = run_power_analysis(scenario, _preregistration())
    path = tmp_path / "power-result.json"

    assert write_power_result(result, path) is True
    assert write_power_result(result, path) is False

    drifted = copy.deepcopy(result)
    drifted["hypothesis_power"]["P1"]["estimated_power"] = 0.123
    with pytest.raises(PowerV2Error, match="hash does not match"):
        write_power_result(drifted, tmp_path / "drifted.json")

    internally_inconsistent = copy.deepcopy(result)
    internally_inconsistent["hypothesis_power"]["P1"]["estimated_power"] = 0.123
    unhashed = dict(internally_inconsistent)
    unhashed.pop("result_hash")
    internally_inconsistent["result_hash"] = canonical_hash(unhashed)
    with pytest.raises(PowerV2Error, match="estimated_power is inconsistent"):
        write_power_result(internally_inconsistent, tmp_path / "inconsistent.json")

    different_scenario = _scenario_payload()
    different_scenario["scenario_id"] = "different-synthetic-scenario"
    different = run_power_analysis(
        parse_power_scenario(different_scenario), _preregistration()
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_power_result(different, path)


def test_power_module_is_not_wired_to_experiment_runners():
    for relative in (
        "experiments/run_gym.py",
        "experiments/run_multishot.py",
        "experiments/run_baseline.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "power_v2" not in source
        assert "run_power_analysis" not in source


def test_power_scenario_json_schema_matches_synthetic_fixture():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "research/schemas/paper_v2_power_scenario.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    assert list(validator.iter_errors(_scenario_payload())) == []
    invalid = _scenario_payload()
    invalid["confirmatory_outcomes_visible"] = True
    assert list(validator.iter_errors(invalid))
