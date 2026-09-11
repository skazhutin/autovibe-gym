from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gym.candidate_store import CANDIDATE_BUNDLE_SCHEMA_VERSION
from gym.context_compression import CONTEXT_COMPRESSION_POLICY_VERSION
from gym.finalization import PROTECTED_FINALIZATION_SCHEMA_VERSION
from gym.terminal_contract import TERMINAL_CONTRACT_VERSION
from research.protocol_v2 import (
    PROTOCOL_V2_BUNDLE_VERSION,
    ProtocolV2Error,
    load_protocol_v2_bundle,
    parse_protocol_v2_bundle,
)


def _bundle_payload():
    return {
        "schema_version": PROTOCOL_V2_BUNDLE_VERSION,
        "experiment_id": "paper-v2-synthetic-fixture",
        "execution_git_commit": "a" * 40,
        "selection_evidence": {
            "selection_scope": "excluded-development-only",
            "confirmatory_outcomes_visible": False,
            "excluded_task_manifest_hash": "b" * 64,
            "policy_selection_plan_hash": "c" * 64,
            "preregistration_hash": "d" * 64,
            "independent_review_hash": "e" * 64,
        },
        "study_contract": {
            "protocol_document_hash": "1" * 64,
            "hypotheses_hash": "2" * 64,
            "analysis_plan_hash": "3" * 64,
            "failure_policy_hash": "4" * 64,
        },
        "experiment_contract": {
            "dataset_manifest_hash": "5" * 64,
            "model_manifest_hash": "6" * 64,
            "prompt_manifest_hash": "7" * 64,
            "decoding_config_hash": "8" * 64,
            "matrix_specification_hash": "9" * 64,
            "randomization_plan_hash": "0" * 64,
        },
        "policies": {
            "episode_budget": {
                "total_token_limit": 1_000,
                "max_output_tokens_per_call": 100,
                "max_llm_calls": 10,
                "max_code_executions": 10,
                "max_tool_calls": 10,
                "wall_clock_limit_seconds": 120.0,
            },
            "finalization_reserve": {
                "total_token_limit": 100,
                "max_llm_calls": 0,
                "max_code_executions": 2,
                "max_tool_calls": 3,
                "wall_clock_limit_seconds": 60.0,
            },
            "validation_query": {
                "policy_version": "validation-query-v1",
                "max_queries": 10,
                "finalization_reserve_queries": 1,
                "feedback_numeric_decimals": 3,
            },
            "context_compression": {
                "policy_version": CONTEXT_COMPRESSION_POLICY_VERSION,
                "max_context_bytes": 10_000,
                "max_recent_messages": 4,
                "max_recent_message_bytes": 2_000,
                "max_feature_columns": 20,
                "max_tested_hypotheses": 10,
                "max_unresolved_errors": 10,
            },
            "stopping": {
                "schema_version": "stopping-policy-v1",
                "reserve_boundary_resources": [
                    "code_executions",
                    "tool_calls",
                    "wall_clock_seconds",
                    "validation_queries",
                ],
                "no_improvement_patience": 2,
                "stop_on_exploration_exhausted": True,
                "stop_on_agent_finalize_request": True,
                "stop_on_unrecoverable_failure": True,
            },
        },
        "terminal_contract": {
            "terminal_contract_version": TERMINAL_CONTRACT_VERSION,
            "candidate_bundle_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
            "protected_finalization_version": (
                PROTECTED_FINALIZATION_SCHEMA_VERSION
            ),
            "metric_direction": "higher",
            "score_tolerance": 1e-9,
            "hidden_evaluations_max": 1,
        },
        "execution_contract": {
            "repeated_executor_backend": "docker",
            "iterative_kernel_backend": "docker",
            "candidate_prediction_backend": "docker",
            "candidate_prediction_network": "none",
            "docker_image_digest": "sha256:" + "f" * 64,
            "sandbox_timeout_seconds": 120,
        },
        "arms": {
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
        },
    }


def test_protocol_bundle_is_canonical_hash_stable_and_receipted():
    payload = _bundle_payload()
    reordered = dict(reversed(list(payload.items())))

    first = parse_protocol_v2_bundle(payload)
    second = parse_protocol_v2_bundle(reordered)

    assert first.bundle_hash == second.bundle_hash
    assert first.canonical_payload == second.canonical_payload
    assert first.to_dict() == payload
    assert first.finalization_reserve.max_tool_calls == 3
    assert first.validation_query.finalization_reserve_queries == 1
    assert first.stopping.no_improvement_patience == 2
    receipt = first.public_receipt()
    assert receipt["bundle_hash"] == first.bundle_hash
    assert set(receipt["policy_hashes"]) == {
        "episode_budget",
        "finalization_reserve",
        "validation_query",
        "context_compression",
        "stopping",
    }
    assert all(len(value) == 64 for value in receipt["policy_hashes"].values())


def test_protocol_hash_normalizes_numeric_spelling_and_stopping_order():
    integer_spelling = _bundle_payload()
    integer_spelling["policies"]["episode_budget"][
        "wall_clock_limit_seconds"
    ] = 120
    integer_spelling["policies"]["finalization_reserve"][
        "wall_clock_limit_seconds"
    ] = 60
    integer_spelling["terminal_contract"]["score_tolerance"] = 0
    integer_spelling["policies"]["stopping"][
        "reserve_boundary_resources"
    ].reverse()

    float_spelling = _bundle_payload()
    float_spelling["terminal_contract"]["score_tolerance"] = 0.0

    first = parse_protocol_v2_bundle(integer_spelling)
    second = parse_protocol_v2_bundle(float_spelling)

    assert first.bundle_hash == second.bundle_hash
    assert first.canonical_payload == second.canonical_payload
    assert first.to_dict()["policies"]["episode_budget"][
        "wall_clock_limit_seconds"
    ] == 120.0


def test_protocol_file_loader_is_strict_and_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "paper-v2-protocol.json"
    path.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    assert load_protocol_v2_bundle(path).to_dict() == _bundle_payload()

    raw = json.dumps(_bundle_payload())
    path.write_text(
        raw.replace(
            '"schema_version": "paper-v2-protocol-bundle-v1",',
            '"schema_version": "paper-v2-protocol-bundle-v1", '
            '"schema_version": "paper-v2-protocol-bundle-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolV2Error, match="Duplicate JSON key"):
        load_protocol_v2_bundle(path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("selection_evidence", "confirmatory_outcomes_visible"),
            True,
            "outcomes to remain unseen",
        ),
        (
            ("selection_evidence", "selection_scope"),
            "all-tasks",
            "excluded development tasks",
        ),
        (
            ("selection_evidence", "preregistration_hash"),
            "not-a-hash",
            "invalid format",
        ),
        (
            ("study_contract", "analysis_plan_hash"),
            "not-a-hash",
            "invalid format",
        ),
        (
            ("experiment_contract", "prompt_manifest_hash"),
            "not-a-hash",
            "invalid format",
        ),
        (
            ("terminal_contract", "hidden_evaluations_max"),
            2,
            "must equal one",
        ),
        (
            ("terminal_contract", "terminal_contract_version"),
            "legacy-terminal",
            "incompatible",
        ),
        (
            ("execution_contract", "candidate_prediction_network"),
            "host",
            "must be 'none'",
        ),
        (
            ("execution_contract", "docker_image_digest"),
            "sha256:latest",
            "invalid format",
        ),
    ],
)
def test_protocol_bundle_rejects_evidence_terminal_and_isolation_drift(
    path,
    value,
    message,
):
    payload = _bundle_payload()
    payload[path[0]][path[1]] = value

    with pytest.raises(ProtocolV2Error, match=message):
        parse_protocol_v2_bundle(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["policies"]["finalization_reserve"].update(
                {"max_tool_calls": 2}
            ),
            "three reserved tool calls",
        ),
        (
            lambda payload: payload["policies"]["finalization_reserve"].update(
                {"max_code_executions": 0}
            ),
            "one reserved code execution",
        ),
        (
            lambda payload: payload["policies"]["validation_query"].update(
                {"finalization_reserve_queries": 0}
            ),
            "reserved validation query",
        ),
        (
            lambda payload: payload["policies"]["validation_query"].update(
                {"finalization_reserve_queries": 10}
            ),
            "positive exploration capacity",
        ),
        (
            lambda payload: payload["policies"]["finalization_reserve"].update(
                {"wall_clock_limit_seconds": 120.0}
            ),
            "positive exploration time",
        ),
        (
            lambda payload: payload["policies"]["episode_budget"].update(
                {"max_llm_calls": True}
            ),
            "positive integer",
        ),
    ],
)
def test_protocol_bundle_rejects_cross_policy_and_type_drift(mutate, message):
    payload = _bundle_payload()
    mutate(payload)

    with pytest.raises(ProtocolV2Error, match=message):
        parse_protocol_v2_bundle(payload)


def test_protocol_bundle_rejects_arm_mapping_or_unknown_fields():
    wrong_arm = _bundle_payload()
    wrong_arm["arms"]["A"]["runner"] = "experiments.run_gym"
    with pytest.raises(ProtocolV2Error, match="adapter mapping"):
        parse_protocol_v2_bundle(wrong_arm)

    unknown = _bundle_payload()
    unknown["policies"]["stopping"]["selected_after_results"] = True
    with pytest.raises(ProtocolV2Error, match="fields differ"):
        parse_protocol_v2_bundle(unknown)

    non_string_key = _bundle_payload()
    non_string_key[1] = "invalid"
    with pytest.raises(ProtocolV2Error, match="keys must be strings"):
        parse_protocol_v2_bundle(non_string_key)

    whitespace = _bundle_payload()
    whitespace["experiment_id"] = " paper-v2 "
    with pytest.raises(ProtocolV2Error, match="surrounding whitespace"):
        parse_protocol_v2_bundle(whitespace)


def test_protocol_bundle_is_not_wired_to_production_runners():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "experiments/run_gym.py",
        "experiments/run_multishot.py",
        "experiments/run_baseline.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "load_protocol_v2_bundle" not in source
        assert "parse_protocol_v2_bundle" not in source


def test_protocol_json_schema_matches_the_admitted_fixture():
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "research/schemas/paper_v2_protocol_bundle.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    assert list(validator.iter_errors(_bundle_payload())) == []
    drifted = _bundle_payload()
    drifted["arms"]["A"]["runner"] = "experiments.run_gym"
    assert list(validator.iter_errors(drifted))
