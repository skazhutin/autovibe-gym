from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.evidence_v2 import (
    ASSUMPTION_EVIDENCE_CLASS,
    ASSUMPTION_EVIDENCE_VERSION,
    CONFIRMATORY_DATASET_SCOPE_VERSION,
    EXCLUDED_DEVELOPMENT_MANIFEST_VERSION,
    SCENARIO_SELECTION_RULE,
    SELECTION_SCOPE,
    EvidenceV2Error,
    _assumption_use_class,
    _scenario_assumption_values,
    load_assumption_evidence_package,
    load_confirmatory_dataset_scope,
    load_excluded_development_manifest,
    parse_assumption_evidence_package,
    parse_confirmatory_dataset_scope,
    parse_excluded_development_manifest,
)
from research.power_v2 import (
    MULTIPLICITY_STRATEGY,
    POWER_EVIDENCE_CLASS,
    POWER_RNG_ENGINE,
    POWER_SCENARIO_VERSION,
    parse_power_scenario,
)
from research.preregistration_v2 import load_preregistration_v2
from research.run_artifacts import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "research/protocols/paper_v2/preregistration.draft.json"


def _hash(label: str) -> str:
    return canonical_hash({"synthetic_fixture": label})


def _preregistration():
    return load_preregistration_v2(PREREGISTRATION_PATH)


def _development_payload():
    return {
        "schema_version": EXCLUDED_DEVELOPMENT_MANIFEST_VERSION,
        "manifest_id": "synthetic-excluded-development",
        "selection_scope": SELECTION_SCOPE,
        "confirmatory_outcomes_visible": False,
        "tasks": [
            {
                "task_id": "synthetic-dev-task",
                "dataset_identity_hash": _hash("dev-identity"),
                "dataset_content_hash": _hash("dev-content"),
                "source_lineage_hash": _hash("dev-lineage"),
                "split_definition_hash": _hash("dev-split"),
                "target_definition_hash": _hash("dev-target"),
                "metric_definition_hash": _hash("dev-metric"),
                "dataset_card_hash": _hash("dev-card"),
            }
        ],
        "independent_review_hash": _hash("dev-review"),
    }


def _confirmatory_payload():
    return {
        "schema_version": CONFIRMATORY_DATASET_SCOPE_VERSION,
        "manifest_id": "synthetic-confirmatory-scope",
        "confirmatory_outcomes_visible": False,
        "datasets": [
            {
                "dataset_id": "synthetic-confirmatory-dataset",
                "dataset_identity_hash": _hash("confirmatory-identity"),
                "dataset_content_hash": _hash("confirmatory-content"),
                "source_lineage_hash": _hash("confirmatory-lineage"),
                "dataset_card_hash": _hash("confirmatory-card"),
            }
        ],
        "independent_review_hash": _hash("confirmatory-review"),
    }


def _scenario_payload(*, scenario_id: str = "synthetic-evidence-scenario"):
    hashes = _preregistration().component_hashes()
    return {
        "schema_version": POWER_SCENARIO_VERSION,
        "scenario_id": scenario_id,
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
            "seed": 23,
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


def _source_for(use_class: str) -> str:
    return {
        "scientific_acceptability_threshold": (
            "independent_methodological_justification"
        ),
        "statistical_design_choice": "external_primary_source",
        "reproducibility_choice": "deterministic_design_choice",
        "candidate_sample_size": "power_grid_design",
        "nuisance_parameter": "historical_paper_v1",
    }[use_class]


def _evidence_payload(development_manifest, scenarios):
    assumptions = []
    references = []
    for scenario in scenarios:
        scenario_payload = scenario.to_dict()
        scenario_id = scenario_payload["scenario_id"]
        references.append(
            {"scenario_id": scenario_id, "scenario_hash": scenario.scenario_hash}
        )
        for key, value in sorted(_scenario_assumption_values(scenario).items()):
            use_class = _assumption_use_class(key)
            assumptions.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_hash": scenario.scenario_hash,
                    "assumption_key": key,
                    "assumption_value_hash": canonical_hash(value),
                    "use_class": use_class,
                    "source_class": _source_for(use_class),
                    "source_artifact_hash": _hash(f"source:{scenario_id}:{key}"),
                    "development_manifest_hash": None,
                }
            )
    return {
        "schema_version": ASSUMPTION_EVIDENCE_VERSION,
        "package_id": "synthetic-assumption-evidence",
        "evidence_class": ASSUMPTION_EVIDENCE_CLASS,
        "confirmatory_outcomes_visible": False,
        "development_manifest_hash": development_manifest.manifest_hash,
        "scenario_policy": {
            "selection_rule": SCENARIO_SELECTION_RULE,
            "favorable_row_selection_allowed": False,
            "scenario_grid_plan_hash": _hash("scenario-grid-plan"),
            "complete_grid_review_hash": _hash("complete-grid-review"),
            "required_scenarios": references,
        },
        "assumptions": assumptions,
        "independent_review_hash": _hash("assumption-review"),
    }


def _fixtures():
    development = parse_excluded_development_manifest(_development_payload())
    scenario = parse_power_scenario(_scenario_payload())
    evidence_payload = _evidence_payload(development, [scenario])
    return development, scenario, evidence_payload


def _record(payload, key: str):
    return next(
        item for item in payload["assumptions"] if item["assumption_key"] == key
    )


def test_development_and_evidence_receipts_are_canonical_and_hash_stable():
    development_payload = _development_payload()
    development = parse_excluded_development_manifest(development_payload)
    reordered_development = parse_excluded_development_manifest(
        dict(reversed(list(development_payload.items())))
    )
    assert development.manifest_hash == reordered_development.manifest_hash
    assert development.public_receipt()["task_count"] == 1
    assert development.public_receipt()["selection_scope"] == SELECTION_SCOPE

    _, scenario, evidence_payload = _fixtures()
    evidence = parse_assumption_evidence_package(evidence_payload)
    reordered_evidence = parse_assumption_evidence_package(
        dict(reversed(list(evidence_payload.items())))
    )
    assert evidence.package_hash == reordered_evidence.package_hash
    assert evidence.public_receipt()["scenario_count"] == 1
    assert evidence.public_receipt()["assumption_count"] == len(
        _scenario_assumption_values(scenario)
    )


def test_excluded_development_is_disjoint_from_confirmatory_scope():
    development = parse_excluded_development_manifest(_development_payload())
    confirmatory = parse_confirmatory_dataset_scope(_confirmatory_payload())

    development.assert_disjoint(confirmatory)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("dataset_identity_hash", "dataset identity"),
        ("dataset_content_hash", "dataset content"),
        ("source_lineage_hash", "source lineage"),
    ],
)
def test_excluded_development_rejects_confirmatory_overlap(key, message):
    development_payload = _development_payload()
    confirmatory_payload = _confirmatory_payload()
    confirmatory_payload["datasets"][0][key] = development_payload["tasks"][0][key]

    development = parse_excluded_development_manifest(development_payload)
    confirmatory = parse_confirmatory_dataset_scope(confirmatory_payload)
    with pytest.raises(EvidenceV2Error, match=message):
        development.assert_disjoint(confirmatory)


def test_dataset_manifests_reject_duplicate_semantics_and_outcome_visibility():
    development_payload = _development_payload()
    development_payload["tasks"].append(copy.deepcopy(development_payload["tasks"][0]))
    development_payload["tasks"][1]["task_id"] = "renamed-task"
    with pytest.raises(EvidenceV2Error, match="Duplicate development task definition"):
        parse_excluded_development_manifest(development_payload)

    confirmatory_payload = _confirmatory_payload()
    confirmatory_payload["confirmatory_outcomes_visible"] = True
    with pytest.raises(EvidenceV2Error, match="before outcomes are visible"):
        parse_confirmatory_dataset_scope(confirmatory_payload)


def test_assumption_package_verifies_exact_scenario_and_value_coverage():
    development, scenario, payload = _fixtures()
    evidence = parse_assumption_evidence_package(payload)

    evidence.verify(
        development_manifest=development,
        preregistration=_preregistration(),
        scenarios=[scenario],
    )


def test_assumption_coverage_is_scoped_per_scenario():
    development = parse_excluded_development_manifest(_development_payload())
    first = parse_power_scenario(_scenario_payload(scenario_id="scenario-low"))
    high_payload = _scenario_payload(scenario_id="scenario-high")
    high_payload["strata"][0]["validity"]["a_rate"] = 0.8
    second = parse_power_scenario(high_payload)
    payload = _evidence_payload(development, [first, second])
    evidence = parse_assumption_evidence_package(payload)

    evidence.verify(
        development_manifest=development,
        preregistration=_preregistration(),
        scenarios=[first, second],
    )
    a_rate_records = [
        item
        for item in payload["assumptions"]
        if item["assumption_key"].endswith("/validity/a_rate")
    ]
    assert len(a_rate_records) == 2
    assert len({item["assumption_value_hash"] for item in a_rate_records}) == 2


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_assumption_package_rejects_incomplete_or_duplicate_coverage(mutation):
    development, scenario, payload = _fixtures()
    if mutation == "missing":
        payload["assumptions"].pop()
        evidence = parse_assumption_evidence_package(payload)
        with pytest.raises(EvidenceV2Error, match="coverage differs"):
            evidence.verify(
                development_manifest=development,
                preregistration=_preregistration(),
                scenarios=[scenario],
            )
        return
    if mutation == "extra":
        extra = copy.deepcopy(payload["assumptions"][0])
        extra["assumption_key"] = "/testing/not-predeclared"
        payload["assumptions"].append(extra)
        evidence = parse_assumption_evidence_package(payload)
        with pytest.raises(EvidenceV2Error, match="coverage differs"):
            evidence.verify(
                development_manifest=development,
                preregistration=_preregistration(),
                scenarios=[scenario],
            )
        return

    payload["assumptions"].append(copy.deepcopy(payload["assumptions"][0]))
    with pytest.raises(EvidenceV2Error, match="Duplicate scenario assumption"):
        parse_assumption_evidence_package(payload)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("scenario_hash", "scenario hash drift"),
        ("assumption_value_hash", "value hash drift"),
    ],
)
def test_assumption_package_rejects_scenario_or_value_hash_drift(field, message):
    development, scenario, payload = _fixtures()
    payload["assumptions"][0][field] = _hash(f"drift:{field}")
    evidence = parse_assumption_evidence_package(payload)

    with pytest.raises(EvidenceV2Error, match=message):
        evidence.verify(
            development_manifest=development,
            preregistration=_preregistration(),
            scenarios=[scenario],
        )


def test_required_scenario_set_and_development_manifest_are_exactly_bound():
    development, scenario, payload = _fixtures()
    payload["scenario_policy"]["required_scenarios"][0]["scenario_hash"] = _hash(
        "wrong-scenario"
    )
    evidence = parse_assumption_evidence_package(payload)
    with pytest.raises(EvidenceV2Error, match="required set"):
        evidence.verify(
            development_manifest=development,
            preregistration=_preregistration(),
            scenarios=[scenario],
        )

    _, _, payload = _fixtures()
    payload["development_manifest_hash"] = _hash("wrong-development")
    evidence = parse_assumption_evidence_package(payload)
    with pytest.raises(EvidenceV2Error, match="development manifest"):
        evidence.verify(
            development_manifest=development,
            preregistration=_preregistration(),
            scenarios=[scenario],
        )


def test_duplicate_supplied_scenario_is_rejected():
    development, scenario, payload = _fixtures()
    evidence = parse_assumption_evidence_package(payload)

    with pytest.raises(EvidenceV2Error, match="Duplicate supplied scenario_id"):
        evidence.verify(
            development_manifest=development,
            preregistration=_preregistration(),
            scenarios=[scenario, scenario],
        )


@pytest.mark.parametrize(
    ("assumption_key", "source_class"),
    [
        ("/testing/noninferiority_margin", "historical_paper_v1"),
        ("/testing/noninferiority_margin", "excluded_development_pilot"),
        ("/testing/familywise_alpha", "historical_paper_v1"),
        (
            "/strata/by-id/synthetic-dataset-model/replicates",
            "excluded_development_pilot",
        ),
    ],
)
def test_inadmissible_assumption_sources_fail_closed(assumption_key, source_class):
    development, scenario, payload = _fixtures()
    record = _record(payload, assumption_key)
    record["source_class"] = source_class
    record["development_manifest_hash"] = (
        development.manifest_hash
        if source_class == "excluded_development_pilot"
        else None
    )
    evidence = parse_assumption_evidence_package(payload)

    with pytest.raises(EvidenceV2Error, match="inadmissible source class"):
        evidence.verify(
            development_manifest=development,
            preregistration=_preregistration(),
            scenarios=[scenario],
        )


def test_excluded_pilot_can_source_only_manifest_bound_nuisance_parameters():
    development, scenario, payload = _fixtures()
    record = _record(payload, "/strata/by-id/synthetic-dataset-model/validity/a_rate")
    record["source_class"] = "excluded_development_pilot"
    record["development_manifest_hash"] = development.manifest_hash
    evidence = parse_assumption_evidence_package(payload)
    evidence.verify(
        development_manifest=development,
        preregistration=_preregistration(),
        scenarios=[scenario],
    )

    record["development_manifest_hash"] = _hash("other-development")
    with pytest.raises(EvidenceV2Error, match="not bound"):
        parse_assumption_evidence_package(payload)


def test_non_pilot_source_cannot_claim_development_manifest():
    development, _, payload = _fixtures()
    record = _record(payload, "/strata/by-id/synthetic-dataset-model/validity/a_rate")
    assert record["source_class"] == "historical_paper_v1"
    record["development_manifest_hash"] = development.manifest_hash

    with pytest.raises(EvidenceV2Error, match="must not claim"):
        parse_assumption_evidence_package(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["scenario_policy"].update(
                {"selection_rule": "best_observed_scenario"}
            ),
            "robust-all-scenarios",
        ),
        (
            lambda payload: payload["scenario_policy"].update(
                {"favorable_row_selection_allowed": True}
            ),
            "Favorable-row",
        ),
        (
            lambda payload: payload.update({"confirmatory_outcomes_visible": True}),
            "outcomes to remain unseen",
        ),
    ],
)
def test_result_blind_robust_scenario_policy_is_mandatory(mutate, message):
    _, _, payload = _fixtures()
    mutate(payload)

    with pytest.raises(EvidenceV2Error, match=message):
        parse_assumption_evidence_package(payload)


def test_strict_json_loaders_reject_duplicate_unknown_and_non_file(tmp_path):
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(_development_payload()), encoding="utf-8")
    assert load_excluded_development_manifest(development_path).to_dict() == (
        _development_payload()
    )

    raw = json.dumps(_development_payload())
    development_path.write_text(
        raw.replace(
            '"schema_version": "paper-v2-excluded-development-manifest-v1",',
            '"schema_version": "paper-v2-excluded-development-manifest-v1", '
            '"schema_version": "paper-v2-excluded-development-manifest-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceV2Error, match="Duplicate JSON key"):
        load_excluded_development_manifest(development_path)

    unknown = _confirmatory_payload()
    unknown["selected_after_results"] = True
    with pytest.raises(EvidenceV2Error, match="fields differ"):
        parse_confirmatory_dataset_scope(unknown)

    with pytest.raises(EvidenceV2Error, match="regular file"):
        load_confirmatory_dataset_scope(tmp_path)


def test_all_m9_json_schemas_match_synthetic_fixtures():
    jsonschema = pytest.importorskip("jsonschema")
    development, scenario, evidence_payload = _fixtures()
    fixtures = [
        (
            "paper_v2_excluded_development_manifest.schema.json",
            development.to_dict(),
        ),
        (
            "paper_v2_confirmatory_dataset_scope.schema.json",
            _confirmatory_payload(),
        ),
        ("paper_v2_assumption_evidence.schema.json", evidence_payload),
    ]

    for filename, fixture in fixtures:
        schema = json.loads(
            (ROOT / "research/schemas" / filename).read_text(encoding="utf-8")
        )
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        assert list(validator.iter_errors(fixture)) == []

    invalid = _evidence_payload(development, [scenario])
    invalid["scenario_policy"]["favorable_row_selection_allowed"] = True
    schema = json.loads(
        (ROOT / "research/schemas/paper_v2_assumption_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(
        jsonschema.validators.validator_for(schema)(schema).iter_errors(invalid)
    )


def test_evidence_module_is_not_wired_to_experiment_runners():
    for relative in (
        "experiments/run_gym.py",
        "experiments/run_multishot.py",
        "experiments/run_baseline.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "evidence_v2" not in source
        assert "AssumptionEvidencePackage" not in source


def test_all_three_loaders_accept_synthetic_regular_files(tmp_path):
    development, scenario, evidence_payload = _fixtures()
    paths = {
        "development": tmp_path / "development.json",
        "confirmatory": tmp_path / "confirmatory.json",
        "evidence": tmp_path / "evidence.json",
    }
    paths["development"].write_text(json.dumps(development.to_dict()), encoding="utf-8")
    paths["confirmatory"].write_text(
        json.dumps(_confirmatory_payload()), encoding="utf-8"
    )
    paths["evidence"].write_text(json.dumps(evidence_payload), encoding="utf-8")

    assert load_excluded_development_manifest(paths["development"]).manifest_hash == (
        development.manifest_hash
    )
    assert load_confirmatory_dataset_scope(paths["confirmatory"]).scope_hash
    loaded = load_assumption_evidence_package(paths["evidence"])
    loaded.verify(
        development_manifest=development,
        preregistration=_preregistration(),
        scenarios=[scenario],
    )
