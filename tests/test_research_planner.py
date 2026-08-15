import copy
import json

import pytest

from research.planner import (
    INFRASTRUCTURE_FAILURES,
    PlanError,
    build_confirmatory_plan,
    load_plan,
    reconcile_plan,
    validate_plan,
    write_plan,
)


def _config(**overrides):
    config = {
        "schema_version": "1.0",
        "experiment_id": "paper-v1-confirmatory-fixture",
        "git_commit": "a" * 40,
        "budget_policy_hash": "b" * 64,
        "matrix": {
            "datasets": [
                {
                    "dataset_id": "dataset-b",
                    "dataset_hash": "c" * 64,
                    "split_id": "split-b",
                    "split_seed": 42,
                },
                {
                    "dataset_id": "dataset-a",
                    "dataset_hash": "d" * 64,
                    "split_id": "split-a",
                    "split_seed": 43,
                },
            ],
            "models": [
                {
                    "model_provider": "provider-b",
                    "model_id": "model-b",
                    "model_version": "2026-08-01",
                    "decoding_config_hash": "e" * 64,
                },
                {
                    "model_provider": "provider-a",
                    "model_id": "model-a",
                    "model_version": "2026-08-02",
                    "decoding_config_hash": "f" * 64,
                },
            ],
            "arms": [
                {
                    "arm": "C",
                    "product_mode": "gym_with_checklist",
                    "execution_policy_hash": "1" * 64,
                    "prompt_template_hash": "2" * 64,
                },
                {
                    "arm": "A",
                    "product_mode": "repeated_single_shot",
                    "execution_policy_hash": "3" * 64,
                    "prompt_template_hash": "4" * 64,
                },
                {
                    "arm": "B",
                    "product_mode": "iterative_no_checklist",
                    "execution_policy_hash": "5" * 64,
                    "prompt_template_hash": "6" * 64,
                },
            ],
            "replicates_per_cell": 2,
            "planned_conditions": 24,
            "randomization_seed": 20260812,
        },
    }
    config.update(overrides)
    return config


def _protocol():
    config = _config()
    return {
        "version": "paper-v1",
        "status": "draft_unfrozen",
        "arms": {
            arm["arm"]: {"current_product_mode": arm["product_mode"]}
            for arm in config["matrix"]["arms"]
        },
        "matrix": {
            "status": "owner_resolved_pre_freeze",
            "models": [{"id": item["model_id"]} for item in config["matrix"]["models"]],
            "datasets": [{"id": item["dataset_id"]} for item in config["matrix"]["datasets"]],
            "confirmatory_arms": ["A", "B", "C"],
            "replicates_per_cell": 2,
            "planned_confirmatory_runs": 24,
            "block_keys": ["dataset_id", "model_id", "replicate_index"],
            "randomization_seed": 20260812,
        },
        "budget": {"status": "frozen"},
    }


def _manifest(
    item,
    run_id,
    *,
    state="completed",
    category="success",
    rerun_of=None,
    rerun_reason=None,
):
    return {
        "schema_version": "1.0",
        "experiment_id": item["condition"]["experiment_id"],
        "condition_id": item["condition_id"],
        "run_id": run_id,
        "rerun_of": rerun_of,
        "rerun_reason": rerun_reason,
        "state": state,
        "started_at": f"2026-08-14T00:00:{run_id[-2:]}Z",
        "completed_at": None if state == "running" else "2026-08-14T00:01:00Z",
        "condition": copy.deepcopy(item["condition"]),
        "provenance": {"git_commit": item["condition"]["git_commit"]},
        "failure_category": None if state == "running" else category,
        "final_status": None if state == "running" else category,
        "summary": {},
        "ledger_totals": {},
        "artifacts": {},
    }


def test_plan_is_deterministic_and_input_order_independent():
    first = build_confirmatory_plan(_config(), protocol=_protocol())
    reordered = _config()
    reordered["matrix"]["datasets"].reverse()
    reordered["matrix"]["models"].reverse()
    reordered["matrix"]["arms"].reverse()
    second = build_confirmatory_plan(reordered, protocol=_protocol())

    assert first == second
    assert first["matrix"]["planned_conditions"] == 24
    assert len({item["condition_id"] for item in first["conditions"]}) == 24


def test_each_block_has_one_seeded_permutation_of_a_b_c():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    blocks = {}
    for item in plan["conditions"]:
        blocks.setdefault(item["block_id"], []).append(item)

    assert len(blocks) == 8
    for items in blocks.values():
        assert {item["condition"]["arm"] for item in items} == {"A", "B", "C"}
        assert [item["within_block_index"] for item in items] == [0, 1, 2]


def test_protocol_sized_matrix_precomputes_all_120_conditions():
    config = _config()
    config["matrix"]["datasets"] = [
        {
            "dataset_id": f"dataset-{index}",
            "dataset_hash": f"{index}" * 64,
            "split_id": f"split-{index}",
            "split_seed": 40 + index,
        }
        for index in range(1, 5)
    ]
    config["matrix"]["replicates_per_cell"] = 5
    config["matrix"]["planned_conditions"] = 120

    protocol = _protocol()
    protocol["matrix"]["datasets"] = [
        {"id": item["dataset_id"]} for item in config["matrix"]["datasets"]
    ]
    protocol["matrix"]["replicates_per_cell"] = 5
    protocol["matrix"]["planned_confirmatory_runs"] = 120
    plan = build_confirmatory_plan(config, protocol=protocol)

    assert len(plan["conditions"]) == 120
    assert plan["matrix"]["block_count"] == 40
    assert len({item["condition_id"] for item in plan["conditions"]}) == 120


def test_plan_rejects_unresolved_human_decisions():
    config = _config()
    config["matrix"]["models"][0]["model_id"] = "TODO"

    with pytest.raises(PlanError, match="placeholder"):
        build_confirmatory_plan(config, protocol=_protocol())


def test_plan_rejects_provisional_protocol_matrix():
    protocol = _protocol()
    protocol["matrix"]["status"] = "provisional_pending_freeze_decisions"

    with pytest.raises(PlanError, match="unresolved"):
        build_confirmatory_plan(_config(), protocol=protocol)


def test_plan_rejects_pilot_budget_status():
    protocol = _protocol()
    protocol["budget"]["status"] = "pilot_default_not_frozen"

    with pytest.raises(PlanError, match="final frozen budget"):
        build_confirmatory_plan(_config(), protocol=protocol)


def test_plan_rejects_wrong_declared_matrix_size():
    config = _config()
    config["matrix"]["planned_conditions"] = 120

    with pytest.raises(PlanError, match=r"datasets \* models \* arms \* replicates"):
        build_confirmatory_plan(config, protocol=_protocol())


def test_plan_write_is_atomic_idempotent_and_never_overwrites(tmp_path):
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    path = tmp_path / "confirmatory-plan.json"

    assert write_plan(plan, path) is True
    assert write_plan(plan, path) is False
    assert load_plan(path) == plan
    assert not list(tmp_path.glob(".*.tmp"))

    changed = build_confirmatory_plan(_config(experiment_id="different"), protocol=_protocol())
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_plan(changed, path)


def test_plan_validation_detects_tampering():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    plan["conditions"][0]["product_mode"] = "tampered"

    with pytest.raises(PlanError, match="plan_hash"):
        validate_plan(plan)


def test_empty_run_set_resumes_every_condition_in_precomputed_order():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    report = reconcile_plan(plan, [])

    assert report["counts"] == {
        "pending": 24,
        "running": 0,
        "terminal": 0,
        "replacement_required": 0,
    }
    assert [item["sequence_index"] for item in report["runnable"]] == list(range(24))
    assert report["complete"] is False


def test_agent_outcome_is_terminal_and_not_replaced():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first = plan["conditions"][0]
    report = reconcile_plan(
        plan,
        [_manifest(first, "run_agent_failure", category="invalid_submission")],
    )

    assert report["counts"]["terminal"] == 1
    assert report["replacement_queue"] == []
    assert report["conditions"][0]["state"] == "terminal"


@pytest.mark.parametrize("category", sorted(INFRASTRUCTURE_FAILURES))
def test_infrastructure_failure_is_queued_for_same_condition(category):
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first = plan["conditions"][0]
    report = reconcile_plan(plan, [_manifest(first, "run_infra_01", category=category)])

    replacement = report["replacement_queue"][0]
    assert replacement["condition_id"] == first["condition_id"]
    assert replacement["rerun_of"] == "run_infra_01"
    assert category in replacement["rerun_reason"]
    assert report["conditions"][0]["state"] == "replacement_required"


def test_successful_replacement_closes_infrastructure_chain():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first = plan["conditions"][0]
    manifests = [
        _manifest(first, "run_infra_01", category="sandbox_failure"),
        _manifest(
            first,
            "run_replacement_02",
            category="success",
            rerun_of="run_infra_01",
            rerun_reason="sandbox failed before agent execution",
        ),
    ]

    report = reconcile_plan(plan, manifests)

    assert report["conditions"][0]["state"] == "terminal"
    assert report["conditions"][0]["attempt_count"] == 2
    assert report["replacement_queue"] == []


def test_duplicate_terminal_outcomes_are_rejected():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first = plan["conditions"][0]
    manifests = [
        _manifest(first, "run_success_01"),
        _manifest(first, "run_success_02"),
    ]

    with pytest.raises(PlanError, match="original attempt|duplicate terminal"):
        reconcile_plan(plan, manifests)


def test_duplicate_run_id_across_conditions_is_rejected():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first, second = plan["conditions"][:2]

    with pytest.raises(PlanError, match="duplicate run_id"):
        reconcile_plan(
            plan,
            [_manifest(first, "run_duplicate"), _manifest(second, "run_duplicate")],
        )


def test_agent_failure_cannot_have_replacement():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first = plan["conditions"][0]
    manifests = [
        _manifest(first, "run_agent_01", category="budget_exhausted"),
        _manifest(
            first,
            "run_illegal_02",
            rerun_of="run_agent_01",
            rerun_reason="not allowed",
        ),
    ]

    with pytest.raises(PlanError, match="does not follow a completed infrastructure failure"):
        reconcile_plan(plan, manifests)


def test_unplanned_condition_is_rejected():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    foreign_plan = build_confirmatory_plan(
        _config(experiment_id="foreign-experiment"), protocol=_protocol()
    )

    with pytest.raises(PlanError, match="unplanned condition_id"):
        reconcile_plan(plan, [_manifest(foreign_plan["conditions"][0], "run_foreign")])


def test_report_is_deterministic_for_manifest_input_order():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    first, second = plan["conditions"][:2]
    manifests = [_manifest(first, "run_first_01"), _manifest(second, "run_second_02")]

    assert reconcile_plan(plan, manifests) == reconcile_plan(plan, list(reversed(manifests)))


def test_plan_json_has_no_outcomes_or_runtime_run_ids():
    plan = build_confirmatory_plan(_config(), protocol=_protocol())
    serialized = json.dumps(plan)

    assert "run_id" not in serialized
    assert "final_status" not in serialized
    assert "hidden" not in serialized
