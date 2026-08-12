from copy import deepcopy
from pathlib import Path

from experiments.modes import MODE_BY_KEY
from research.validate_protocol import load_protocol, validate_protocol


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "protocols"
    / "protocol_v1.yaml"
)


def test_protocol_v1_is_structurally_valid_and_unfrozen():
    protocol = load_protocol(PROTOCOL_PATH)

    assert validate_protocol(protocol) == []
    assert protocol["status"] == "draft_unfrozen"
    assert protocol["freeze"]["allowed"] is False
    assert protocol["matrix"]["planned_confirmatory_runs"] == 120
    assert any(
        todo["freeze_blocking"] and todo["status"] == "open"
        for todo in protocol["todos"]
    )


def test_protocol_validator_rejects_hidden_test_retry_loop():
    protocol = deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["submission_policy"]["max_hidden_evaluations_per_agent_outcome"] = 3

    errors = validate_protocol(protocol)

    assert "hidden evaluation must be attempted at most once per agent outcome" in errors


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
