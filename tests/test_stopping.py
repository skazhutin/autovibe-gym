from __future__ import annotations

import argparse
import json

import pytest

from research.budget import (
    EpisodeBudget,
    EpisodeBudgetPolicy,
    FinalizationReservePolicy,
    ValidationQueryPolicy,
)
from research.stopping import (
    STOPPING_POLICY_VERSION,
    FrozenStoppingPolicy,
    StoppingController,
    StoppingPolicyError,
    load_stopping_policy,
)
from research.runner_integration import (
    budget_policy_payload,
    create_stopping_policy,
)


def _policy(**overrides):
    values = {
        "reserve_boundary_resources": ("llm_calls", "code_executions"),
        "no_improvement_patience": 2,
        "stop_on_exploration_exhausted": True,
        "stop_on_agent_finalize_request": True,
        "stop_on_unrecoverable_failure": True,
    }
    values.update(overrides)
    return FrozenStoppingPolicy(**values)


def test_frozen_stopping_policy_is_explicit_canonical_and_hash_stable():
    first = _policy(
        reserve_boundary_resources=("code_executions", "llm_calls")
    )
    second = _policy(
        reserve_boundary_resources=("llm_calls", "code_executions")
    )

    assert first.to_dict() == second.to_dict()
    assert first.policy_hash == second.policy_hash
    assert first.to_dict() == {
        "schema_version": STOPPING_POLICY_VERSION,
        "reserve_boundary_resources": ["llm_calls", "code_executions"],
        "no_improvement_patience": 2,
        "stop_on_exploration_exhausted": True,
        "stop_on_agent_finalize_request": True,
        "stop_on_unrecoverable_failure": True,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"reserve_boundary_resources": ("unknown",)},
        {"reserve_boundary_resources": ("llm_calls", "llm_calls")},
        {"no_improvement_patience": 0},
        {"no_improvement_patience": True},
        {"stop_on_exploration_exhausted": 1},
        {
            "reserve_boundary_resources": (),
            "no_improvement_patience": None,
            "stop_on_exploration_exhausted": False,
            "stop_on_agent_finalize_request": False,
            "stop_on_unrecoverable_failure": False,
        },
    ],
)
def test_frozen_stopping_policy_rejects_implicit_or_invalid_contracts(overrides):
    with pytest.raises(StoppingPolicyError):
        _policy(**overrides)


def test_policy_file_loader_is_strict_and_rejects_duplicate_or_unknown_fields(
    tmp_path,
):
    path = tmp_path / "stopping.json"
    path.write_text(json.dumps(_policy().to_dict()), encoding="utf-8")
    assert load_stopping_policy(path) == _policy()

    path.write_text(
        '{"schema_version":"stopping-policy-v1",'
        '"schema_version":"stopping-policy-v1",'
        '"reserve_boundary_resources":[],"no_improvement_patience":1,'
        '"stop_on_exploration_exhausted":true,'
        '"stop_on_agent_finalize_request":true,'
        '"stop_on_unrecoverable_failure":true}',
        encoding="utf-8",
    )
    with pytest.raises(StoppingPolicyError, match="Duplicate JSON key"):
        load_stopping_policy(path)

    payload = _policy().to_dict()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoppingPolicyError, match="invalid shape"):
        load_stopping_policy(path)


def test_no_improvement_patience_counts_only_eligible_attempts_and_resets():
    controller = StoppingController(_policy(no_improvement_patience=2))

    assert controller.observe_validation_attempt(
        eligible=False,
        improved=False,
        candidate_id=None,
        incumbent_candidate_id="baseline",
    ) is None
    assert controller.no_improvement_streak == 0
    assert controller.observe_validation_attempt(
        eligible=True,
        improved=False,
        candidate_id="worse-1",
        incumbent_candidate_id="baseline",
    ) is None
    assert controller.observe_validation_attempt(
        eligible=True,
        improved=True,
        candidate_id="better",
        incumbent_candidate_id="better",
    ) is None
    assert controller.no_improvement_streak == 0
    assert controller.observe_validation_attempt(
        eligible=True,
        improved=False,
        candidate_id="worse-2",
        incumbent_candidate_id="better",
    ) is None
    decision = controller.observe_validation_attempt(
        eligible=True,
        improved=False,
        candidate_id="worse-3",
        incumbent_candidate_id="better",
    )

    assert decision is not None
    assert decision.reason == "no_validation_improvement"
    assert decision.transition == "finalize"
    assert decision.no_improvement_streak == 2


def test_first_trigger_wins_and_later_events_cannot_reclassify_stop():
    controller = StoppingController(_policy())
    first = controller.observe_reserve_boundary(
        boundary_resources=("code_executions",),
        incumbent_candidate_id="best",
    )
    later = controller.observe_unrecoverable_failure(
        detail_code="bundle_corrupt",
        incumbent_candidate_id="best",
    )

    assert first is not None
    assert first.reason == "reserve_boundary_reached"
    assert first.boundary_resources == ("code_executions",)
    assert later == first
    assert controller.snapshot()["events_observed"] == 2


def test_agent_request_requires_valid_incumbent_and_unrecoverable_is_terminate():
    controller = StoppingController(_policy())
    assert controller.observe_agent_finalization_request(
        has_valid_incumbent=False,
        incumbent_candidate_id=None,
    ) is None
    requested = controller.observe_agent_finalization_request(
        has_valid_incumbent=True,
        incumbent_candidate_id="best",
    )
    assert requested is not None
    assert requested.reason == "agent_requested_finalization"

    failure_controller = StoppingController(_policy())
    failed = failure_controller.observe_unrecoverable_failure(
        detail_code="candidate_store_integrity",
        incumbent_candidate_id=None,
    )
    assert failed is not None
    assert failed.reason == "unrecoverable_contract_failure"
    assert failed.transition == "terminate"


def test_exploration_exhaustion_is_policy_gated_and_preserves_detail_code():
    disabled = StoppingController(
        _policy(stop_on_exploration_exhausted=False)
    )
    assert disabled.observe_exploration_exhausted(
        detail_code="exploration_max_llm_calls",
        incumbent_candidate_id="best",
    ) is None

    enabled = StoppingController(_policy())
    decision = enabled.observe_exploration_exhausted(
        detail_code="exploration_max_llm_calls",
        incumbent_candidate_id="best",
    )
    assert decision is not None
    assert decision.reason == "exploration_pool_exhausted"
    assert decision.detail_code == "exploration_max_llm_calls"


def test_budget_boundary_introspection_is_read_only_and_resource_complete():
    budget = EpisodeBudget(
        EpisodeBudgetPolicy(
            total_token_limit=100,
            max_output_tokens_per_call=10,
            max_llm_calls=3,
            max_code_executions=3,
            max_tool_calls=3,
            wall_clock_limit_seconds=120,
        ),
        finalization_reserve=FinalizationReservePolicy(
            max_llm_calls=1,
            max_code_executions=1,
            max_tool_calls=1,
            wall_clock_limit_seconds=60,
        ),
        validation_query_policy=ValidationQueryPolicy(
            max_queries=3,
            finalization_reserve_queries=1,
            feedback_numeric_decimals=3,
        ),
    )
    budget.consume_code_execution()
    budget.consume_code_execution()
    budget.consume_tool_call("first")
    budget.consume_tool_call("second")
    budget.consume_validation_query("first")
    budget.consume_validation_query("second")
    snapshot_before = budget.snapshot()

    boundary = budget.exploration_boundary_resources()

    assert boundary == (
        "code_executions",
        "tool_calls",
        "validation_queries",
    )
    snapshot_after = budget.snapshot()
    assert {
        key: value
        for key, value in snapshot_after.items()
        if key != "elapsed_seconds"
    } == {
        key: value
        for key, value in snapshot_before.items()
        if key != "elapsed_seconds"
    }
    assert budget.phase == "exploration"


def test_runner_loader_requires_external_file_and_protected_reserve(tmp_path):
    path = tmp_path / "stopping.json"
    path.write_text(json.dumps(_policy().to_dict()), encoding="utf-8")
    args = argparse.Namespace(research_v2_stopping_policy=str(path))

    with pytest.raises(ValueError, match="protected finalization reserve"):
        create_stopping_policy(args, None)
    with pytest.raises(ValueError, match="protected finalization reserve"):
        create_stopping_policy(
            args,
            EpisodeBudget(EpisodeBudgetPolicy()),
        )

    budget = EpisodeBudget(
        EpisodeBudgetPolicy(wall_clock_limit_seconds=120),
        finalization_reserve=FinalizationReservePolicy(
            max_tool_calls=1,
            wall_clock_limit_seconds=60,
        ),
    )
    loaded = create_stopping_policy(args, budget)

    assert loaded == _policy()
    payload = budget_policy_payload(
        budget,
        fallback={"unused": True},
        stopping_policy=loaded,
    )
    assert payload["stopping_policy"] == _policy().to_dict()
    assert payload["stopping_policy_hash"] == _policy().policy_hash


def test_runner_loader_is_noop_without_explicit_policy_path():
    args = argparse.Namespace(research_v2_stopping_policy=None)
    assert create_stopping_policy(args, None) is None
    assert budget_policy_payload(
        None,
        fallback={"logical_llm_calls": 3},
    ) == {"logical_llm_calls": 3}
