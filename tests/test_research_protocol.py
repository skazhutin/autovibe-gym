from copy import deepcopy
import json
from pathlib import Path

import yaml

from experiments.modes import MODE_BY_KEY
from research.run_artifacts import canonical_hash
from research.validate_protocol import load_protocol, validate_protocol


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "protocols"
    / "protocol_v1.yaml"
)


def test_protocol_v1_is_structurally_valid_and_frozen():
    protocol = load_protocol(PROTOCOL_PATH)

    assert validate_protocol(protocol) == []
    assert protocol["status"] == "frozen"
    assert protocol["freeze"]["allowed"] is True
    assert protocol["freeze"]["tag"] == "paper-v1-experiment-freeze-v2"
    assert protocol["freeze"]["supersedes_tag"] == "paper-v1-experiment-freeze"
    assert protocol["freeze"]["outcomes_visible_before_amendment"] is False
    assert protocol["matrix"]["planned_confirmatory_runs"] == 120
    assert protocol["budget"]["total_token_limit"] == 256000
    assert (
        protocol["submission_policy"]["candidate_prediction_backend"]
        == "isolated_ephemeral_docker"
    )
    assert protocol["submission_policy"]["host_side_candidate_predict_execution"] is False
    assert not any(
        todo["freeze_blocking"] and todo["status"] == "open"
        for todo in protocol["todos"]
    )


def test_protocol_validator_rejects_hidden_test_retry_loop():
    protocol = deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["submission_policy"]["max_hidden_evaluations_per_agent_outcome"] = 3

    errors = validate_protocol(protocol)

    assert "hidden evaluation must be attempted at most once per agent outcome" in errors


def test_protocol_validator_rejects_host_candidate_execution():
    protocol = deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["submission_policy"]["host_side_candidate_predict_execution"] = True

    errors = validate_protocol(protocol)

    assert "confirmatory candidate code must not execute in the host recorder" in errors


def test_protocol_arms_map_to_current_product_registry():
    protocol = load_protocol(PROTOCOL_PATH)

    mapped_modes = {
        arm: config["current_product_mode"]
        for arm, config in protocol["arms"].items()
    }

    assert mapped_modes == {
        "A": "repeated_single_shot",
        "B": "iterative_no_checklist",
        "C": "gym_with_checklist",
        "D": "fixed_transitions",
    }
    assert set(mapped_modes.values()) <= set(MODE_BY_KEY)


def test_frozen_protocol_binds_manifest_references_and_dataset_cards():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol_dir = PROTOCOL_PATH.parent
    manifest = json.loads((protocol_dir / "freeze_manifest_v1.json").read_text("utf-8"))
    references = json.loads(
        (protocol_dir / "fanu_reference_values_v1.json").read_text("utf-8")
    )
    cards = yaml.safe_load((protocol_dir / "dataset_cards_v1.yaml").read_text("utf-8"))

    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    reference_payload = {
        key: value for key, value in references.items() if key != "reference_hash"
    }
    assert canonical_hash(manifest_payload) == manifest["manifest_hash"]
    assert canonical_hash(reference_payload) == references["reference_hash"]
    assert protocol["freeze"]["frozen_input_manifest_hash"] == manifest["manifest_hash"]
    assert protocol["freeze"]["fanu_reference_hash"] == references["reference_hash"]

    protocol_datasets = {
        item["id"]: item["dataset_hash"]
        for item in protocol["matrix"]["datasets"]
    }
    manifest_datasets = {
        item["dataset_id"]: item["dataset_hash"]
        for item in manifest["datasets"]
    }
    assert protocol_datasets == manifest_datasets
    assert {item["dataset_id"] for item in cards["cards"]} == set(protocol_datasets)
