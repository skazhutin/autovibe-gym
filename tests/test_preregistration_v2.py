from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.preregistration_v2 import (
    PREREGISTRATION_FREEZE_CANDIDATE_STATUS,
    PREREGISTRATION_V2_VERSION,
    PreregistrationV2Error,
    load_preregistration_v2,
    parse_preregistration_v2,
)


ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = ROOT / "research/protocols/paper_v2/preregistration.draft.json"


def _draft_payload():
    return json.loads(DRAFT_PATH.read_text(encoding="utf-8"))


def _resolved_payload():
    payload = _draft_payload()
    payload["status"] = PREREGISTRATION_FREEZE_CANDIDATE_STATUS
    values = {
        "noninferiority_margin": 0.1,
        "familywise_alpha": 0.05,
        "target_power": 0.8,
        "replicates_per_stratum": 3,
        "resampling_draws": 100,
        "randomization_draws": 100,
        "analysis_seed": 7,
        "secondary_multiplicity_strategy": (
            "hierarchical_gatekeeping_with_secondary_holm"
        ),
        "dataset_model_scope": "a" * 64,
        "fanu_references": "b" * 64,
        "statistical_power_analysis": "c" * 64,
        "independent_statistical_review": "d" * 64,
    }
    for index, (decision_id, value) in enumerate(values.items()):
        payload["freeze_decisions"][decision_id] = {
            "status": "resolved",
            "value": value,
            "rationale": "Synthetic test-only rationale.",
            "evidence_hash": format(index + 1, "x") * 64,
        }
    return payload


class _ProtocolFixture:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return copy.deepcopy(self._payload)


def _bound_protocol_payload(preregistration):
    return {
        "selection_evidence": {
            "preregistration_hash": preregistration.preregistration_hash,
        },
        "study_contract": preregistration.component_hashes(),
    }


def test_committed_draft_is_valid_result_blind_and_explicitly_not_freeze_ready():
    preregistration = load_preregistration_v2(DRAFT_PATH)

    assert preregistration.status == "draft"
    assert preregistration.to_dict()["confirmatory_outcomes_visible"] is False
    assert preregistration.freeze_ready is False
    assert preregistration.unresolved_decisions() == (
        "noninferiority_margin",
        "familywise_alpha",
        "target_power",
        "replicates_per_stratum",
        "resampling_draws",
        "randomization_draws",
        "analysis_seed",
        "secondary_multiplicity_strategy",
        "dataset_model_scope",
        "fanu_references",
        "statistical_power_analysis",
        "independent_statistical_review",
    )
    with pytest.raises(PreregistrationV2Error, match="not freeze_candidate"):
        preregistration.assert_freeze_ready()


def test_preregistration_is_canonical_hash_stable_and_receipted():
    payload = _draft_payload()
    reordered = dict(reversed(list(payload.items())))

    first = parse_preregistration_v2(payload)
    second = parse_preregistration_v2(reordered)

    assert first.canonical_payload == second.canonical_payload
    assert first.preregistration_hash == second.preregistration_hash
    receipt = first.public_receipt()
    assert receipt["schema_version"] == PREREGISTRATION_V2_VERSION
    assert receipt["freeze_ready"] is False
    assert set(receipt["component_hashes"]) == {
        "protocol_document_hash",
        "hypotheses_hash",
        "analysis_plan_hash",
        "failure_policy_hash",
    }
    assert all(len(value) == 64 for value in receipt["component_hashes"].values())


def test_synthetic_resolved_fixture_can_become_freeze_candidate():
    preregistration = parse_preregistration_v2(_resolved_payload())

    assert preregistration.freeze_ready is True
    assert preregistration.unresolved_decisions() == ()
    preregistration.assert_freeze_ready()


def test_freeze_candidate_rejects_any_unresolved_decision():
    payload = _draft_payload()
    payload["status"] = PREREGISTRATION_FREEZE_CANDIDATE_STATUS

    with pytest.raises(PreregistrationV2Error, match="every decision"):
        parse_preregistration_v2(payload)


def test_protocol_reference_binding_requires_freeze_ready_preregistration():
    draft = parse_preregistration_v2(_draft_payload())
    with pytest.raises(PreregistrationV2Error, match="draft preregistration"):
        draft.verify_protocol_references(_ProtocolFixture({}))

    preregistration = parse_preregistration_v2(_resolved_payload())
    protocol = _ProtocolFixture(_bound_protocol_payload(preregistration))
    preregistration.verify_protocol_references(protocol)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (
            ("selection_evidence", "preregistration_hash"),
            "preregistration hash",
        ),
        (
            ("study_contract", "hypotheses_hash"),
            "hypotheses_hash",
        ),
        (
            ("study_contract", "analysis_plan_hash"),
            "analysis_plan_hash",
        ),
        (
            ("study_contract", "failure_policy_hash"),
            "failure_policy_hash",
        ),
        (
            ("study_contract", "protocol_document_hash"),
            "protocol_document_hash",
        ),
    ],
)
def test_protocol_reference_binding_rejects_hash_drift(path, message):
    preregistration = parse_preregistration_v2(_resolved_payload())
    protocol = _bound_protocol_payload(preregistration)
    protocol[path[0]][path[1]] = "f" * 64

    with pytest.raises(PreregistrationV2Error, match=message):
        preregistration.verify_protocol_references(_ProtocolFixture(protocol))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["hypotheses"][0].update(
                {"alternative": "superiority"}
            ),
            "hypotheses differ",
        ),
        (
            lambda payload: payload["analysis_plan"].update(
                {"successful_only_analysis": "primary"}
            ),
            "analysis_plan differs",
        ),
        (
            lambda payload: payload["failure_policy"].update(
                {"complete_case_primary_analysis": True}
            ),
            "failure_policy differs",
        ),
        (
            lambda payload: payload["mechanistic_plan"].update(
                {"pool_with_confirmatory_abc": True}
            ),
            "mechanistic_plan differs",
        ),
        (
            lambda payload: payload["claim_rules"].update(
                {"checklist_coverage_proves_quality": True}
            ),
            "claim_rules differs",
        ),
        (
            lambda payload: payload.update({"confirmatory_outcomes_visible": True}),
            "outcomes to remain unseen",
        ),
    ],
)
def test_preregistration_rejects_estimand_failure_and_claim_drift(mutate, message):
    payload = _draft_payload()
    mutate(payload)

    with pytest.raises(PreregistrationV2Error, match=message):
        parse_preregistration_v2(payload)


def test_decision_contract_rejects_pseudo_resolution_and_invalid_values():
    pseudo = _draft_payload()
    pseudo["freeze_decisions"]["noninferiority_margin"]["value"] = 0.1
    with pytest.raises(PreregistrationV2Error, match="must not contain a value"):
        parse_preregistration_v2(pseudo)

    invalid = _resolved_payload()
    invalid["freeze_decisions"]["noninferiority_margin"]["value"] = 1.0
    with pytest.raises(PreregistrationV2Error, match=r"must be in \(0, 0.5\)"):
        parse_preregistration_v2(invalid)

    boolean = _resolved_payload()
    boolean["freeze_decisions"]["replicates_per_stratum"]["value"] = True
    with pytest.raises(PreregistrationV2Error, match="positive integer"):
        parse_preregistration_v2(boolean)

    unreviewed = _resolved_payload()
    unreviewed["freeze_decisions"]["familywise_alpha"]["evidence_hash"] = None
    with pytest.raises(PreregistrationV2Error, match="non-empty string"):
        parse_preregistration_v2(unreviewed)


def test_preregistration_loader_rejects_duplicate_and_unknown_fields(tmp_path):
    raw = DRAFT_PATH.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        raw.replace(
            '"schema_version": "paper-v2-preregistration-v1",',
            '"schema_version": "paper-v2-preregistration-v1",\n'
            '  "schema_version": "paper-v2-preregistration-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationV2Error, match="Duplicate JSON key"):
        load_preregistration_v2(duplicate)

    unknown = _draft_payload()
    unknown["selected_after_results"] = True
    with pytest.raises(PreregistrationV2Error, match="fields differ"):
        parse_preregistration_v2(unknown)


def test_preregistration_is_not_wired_to_production_runners():
    for relative in (
        "experiments/run_gym.py",
        "experiments/run_multishot.py",
        "experiments/run_baseline.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "load_preregistration_v2" not in source
        assert "parse_preregistration_v2" not in source


def test_preregistration_json_schema_matches_draft_and_freeze_candidate():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "research/schemas/paper_v2_preregistration.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    assert list(validator.iter_errors(_draft_payload())) == []
    assert list(validator.iter_errors(_resolved_payload())) == []

    unresolved_freeze = _draft_payload()
    unresolved_freeze["status"] = PREREGISTRATION_FREEZE_CANDIDATE_STATUS
    assert list(validator.iter_errors(unresolved_freeze))
