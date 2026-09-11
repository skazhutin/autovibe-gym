from dataclasses import dataclass

import pytest

from research.budget import (
    BudgetExhausted,
    BudgetInvariantError,
    BudgetedExecutor,
    BudgetedLLMClient,
    EpisodeBudget,
    EpisodeBudgetPolicy,
    ExplorationBudgetExhausted,
    FinalizationReservePolicy,
    VALIDATION_QUERY_POLICY_VERSION,
    ValidationQueryPolicy,
    conservative_input_token_bound,
)


@dataclass
class _Response:
    text: str = "ok"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int = 2


class _Client:
    def __init__(self, response=None):
        self.response = response or _Response()
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _Executor:
    def __init__(self):
        self.calls = 0

    def run(self, code, namespace=None):
        self.calls += 1
        return "", "", namespace or {}


def _policy(**overrides):
    values = {
        "total_token_limit": 10_000,
        "max_output_tokens_per_call": 100,
        "max_llm_calls": 2,
        "max_code_executions": 2,
        "max_tool_calls": 2,
        "wall_clock_limit_seconds": 60,
    }
    values.update(overrides)
    return EpisodeBudgetPolicy(**values)


def _reserve(**overrides):
    values = {
        "total_token_limit": 1_000,
        "max_llm_calls": 1,
        "max_code_executions": 1,
        "max_tool_calls": 1,
        "wall_clock_limit_seconds": 10,
    }
    values.update(overrides)
    return FinalizationReservePolicy(**values)


def _validation_policy(**overrides):
    values = {
        "max_queries": 5,
        "finalization_reserve_queries": 0,
        "feedback_numeric_decimals": 3,
    }
    values.update(overrides)
    return ValidationQueryPolicy(**values)


def test_budgeted_llm_caps_output_and_records_reported_usage():
    budget = EpisodeBudget(_policy())
    client = _Client()
    wrapped = BudgetedLLMClient(client, budget)

    response = wrapped.complete(system="system", messages=[], model="m", max_tokens=999)

    assert response.text == "ok"
    assert client.calls[0]["max_tokens"] == 100
    assert budget.llm_calls == 1
    assert budget.total_tokens == 17
    assert budget.snapshot()["reasoning_tokens"] == 2


def test_reasoning_and_visible_output_share_the_reserved_output_cap():
    budget = EpisodeBudget(_policy(max_output_tokens_per_call=10))
    client = _Client(_Response(input_tokens=10, output_tokens=8, reasoning_tokens=2))

    BudgetedLLMClient(client, budget).complete(
        system="s", messages=[], model="m", max_tokens=10
    )

    assert budget.total_tokens == 20
    assert budget.snapshot()["output_tokens"] == 8
    assert budget.snapshot()["reasoning_tokens"] == 2


def test_pre_call_enforcement_stops_without_invoking_provider():
    messages = [{"role": "user", "content": "x" * 100}]
    reservation = conservative_input_token_bound("system", messages) + 50
    budget = EpisodeBudget(_policy(total_token_limit=reservation - 1, max_output_tokens_per_call=50))
    client = _Client()
    wrapped = BudgetedLLMClient(client, budget)

    with pytest.raises(BudgetExhausted, match="pre_call_token_reservation"):
        wrapped.complete(system="system", messages=messages, model="m", max_tokens=50)

    assert client.calls == []
    assert budget.llm_calls == 0


def test_llm_call_limit_is_checked_before_provider_call():
    budget = EpisodeBudget(_policy(max_llm_calls=1))
    client = _Client()
    wrapped = BudgetedLLMClient(client, budget)
    wrapped.complete(system="s", messages=[], model="m", max_tokens=10)

    with pytest.raises(BudgetExhausted, match="max_llm_calls"):
        wrapped.complete(system="s", messages=[], model="m", max_tokens=10)

    assert len(client.calls) == 1


def test_code_and_tool_limits_are_independently_enforced():
    budget = EpisodeBudget(_policy(max_code_executions=1, max_tool_calls=1))
    executor = _Executor()
    wrapped = BudgetedExecutor(executor, budget)

    wrapped.run("x = 1")
    budget.consume_tool_call("validate")
    with pytest.raises(BudgetExhausted, match="max_code_executions"):
        wrapped.run("x = 2")
    assert executor.calls == 1


def test_wall_clock_limit_is_pre_action_enforced():
    now = [0.0]
    budget = EpisodeBudget(_policy(wall_clock_limit_seconds=5), clock=lambda: now[0])
    now[0] = 5.0

    with pytest.raises(BudgetExhausted, match="wall_clock_limit"):
        budget.consume_tool_call("validate")


def test_provider_usage_above_reservation_is_an_invariant_failure():
    budget = EpisodeBudget(_policy())
    client = _Client(_Response(input_tokens=50_000, output_tokens=1, reasoning_tokens=0))
    wrapped = BudgetedLLMClient(client, budget)

    with pytest.raises(BudgetInvariantError, match="exceeded conservative reservation"):
        wrapped.complete(system="s", messages=[], model="m", max_tokens=10)


def test_budget_events_are_auditable():
    events = []
    budget = EpisodeBudget(_policy(), event_sink=events.append)
    BudgetedLLMClient(_Client(), budget).complete(
        system="s", messages=[], model="m", max_tokens=10
    )

    assert [event["event"] for event in events] == [
        "budget_llm_call_reserved",
        "budget_llm_call_completed",
    ]


def test_unpartitioned_budget_keeps_v1_payload_and_snapshot_shape():
    budget = EpisodeBudget(_policy(), clock=lambda: 0.0)

    assert budget.policy_payload() == budget.policy.to_dict()
    assert budget.snapshot() == {
        "policy": budget.policy.to_dict(),
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "code_executions": 0,
        "tool_calls": 0,
        "elapsed_seconds": 0.0,
        "exhausted": False,
        "exhausted_reason": None,
    }


def test_reserve_must_fit_inside_global_policy_and_protect_real_resources():
    with pytest.raises(ValueError, match="exceeds the global"):
        EpisodeBudget(
            _policy(max_code_executions=1),
            finalization_reserve=_reserve(max_code_executions=2),
        )
    with pytest.raises(ValueError, match="protect at least one resource"):
        FinalizationReservePolicy(wall_clock_limit_seconds=1)
    with pytest.raises(ValueError, match="non-negative"):
        _reserve(max_tool_calls=-1)
    with pytest.raises(ValueError, match="non-negative"):
        _reserve(max_llm_calls=1.5)
    with pytest.raises(ValueError, match="positive"):
        _reserve(wall_clock_limit_seconds=float("nan"))


def test_exploration_stops_before_reserved_code_execution_without_global_exhaustion():
    budget = EpisodeBudget(
        _policy(max_code_executions=3),
        finalization_reserve=_reserve(max_code_executions=1),
    )

    budget.consume_code_execution()
    budget.consume_code_execution()
    with pytest.raises(
        ExplorationBudgetExhausted,
        match="exploration_max_code_executions",
    ):
        budget.consume_code_execution()

    snapshot = budget.snapshot()
    assert budget.code_executions == 2
    assert budget.exhausted_reason is None
    assert snapshot["exhausted"] is False
    assert snapshot["exploration_exhausted"] is True
    assert snapshot["exploration_limits"]["max_code_executions"] == 2


def test_finalization_is_one_way_and_cannot_borrow_unused_exploration_capacity():
    events = []
    budget = EpisodeBudget(
        _policy(max_code_executions=5),
        finalization_reserve=_reserve(max_code_executions=1),
        event_sink=events.append,
    )
    budget.consume_code_execution()

    budget.begin_finalization(trigger="agent_requested")
    budget.begin_finalization(trigger="duplicate_request")
    budget.consume_code_execution()
    with pytest.raises(BudgetExhausted, match="finalization_max_code_executions"):
        budget.consume_code_execution()

    assert budget.code_executions == 2
    assert budget.phase == "finalization"
    assert budget.snapshot()["finalization_usage"]["code_executions"] == 1
    assert [event["event"] for event in events].count(
        "budget_finalization_started"
    ) == 1


def test_zero_finalization_allocation_forbids_that_resource():
    budget = EpisodeBudget(
        _policy(max_tool_calls=4),
        finalization_reserve=_reserve(max_tool_calls=0),
    )
    budget.begin_finalization(trigger="host")

    with pytest.raises(BudgetExhausted, match="finalization_max_tool_calls"):
        budget.consume_tool_call("submit")
    assert budget.tool_calls == 0


def test_finalization_llm_limits_apply_to_calls_and_reserved_tokens():
    budget = EpisodeBudget(
        _policy(max_llm_calls=3, total_token_limit=10_000),
        finalization_reserve=_reserve(
            max_llm_calls=1,
            total_token_limit=1_000,
        ),
    )
    budget.begin_finalization(trigger="host")
    BudgetedLLMClient(_Client(), budget).complete(
        system="s",
        messages=[],
        model="m",
        max_tokens=10,
    )

    with pytest.raises(BudgetExhausted, match="finalization_max_llm_calls"):
        budget.begin_llm_call(
            system="s",
            messages=[],
            requested_max_output_tokens=10,
        )
    assert budget.snapshot()["finalization_usage"]["total_tokens"] == 17


def test_exploration_and_finalization_token_reservations_use_separate_pools():
    small_reservation = conservative_input_token_bound("s", []) + 10
    exploration_budget = EpisodeBudget(
        _policy(
            total_token_limit=small_reservation + 100,
            max_llm_calls=3,
        ),
        finalization_reserve=_reserve(
            total_token_limit=101,
            max_llm_calls=1,
        ),
    )
    with pytest.raises(
        ExplorationBudgetExhausted,
        match="exploration_pre_call_token_reservation",
    ):
        exploration_budget.begin_llm_call(
            system="s",
            messages=[],
            requested_max_output_tokens=10,
        )

    finalization_budget = EpisodeBudget(
        _policy(total_token_limit=10_000, max_llm_calls=3),
        finalization_reserve=_reserve(
            total_token_limit=small_reservation - 1,
            max_llm_calls=2,
        ),
    )
    finalization_budget.begin_finalization(trigger="host")
    with pytest.raises(
        BudgetExhausted,
        match="finalization_pre_call_token_reservation",
    ):
        finalization_budget.begin_llm_call(
            system="s",
            messages=[],
            requested_max_output_tokens=10,
        )
    assert finalization_budget.llm_calls == 0


def test_finalization_wall_clock_cannot_borrow_early_exploration_time():
    now = [0.0]
    budget = EpisodeBudget(
        _policy(wall_clock_limit_seconds=100),
        finalization_reserve=_reserve(wall_clock_limit_seconds=10),
        clock=lambda: now[0],
    )
    now[0] = 5.0
    budget.begin_finalization(trigger="early")
    now[0] = 15.0

    with pytest.raises(BudgetExhausted, match="finalization_wall_clock_limit"):
        budget.ensure_active()


def test_exploration_wall_boundary_preserves_reserve_for_transition():
    now = [0.0]
    budget = EpisodeBudget(
        _policy(wall_clock_limit_seconds=100),
        finalization_reserve=_reserve(wall_clock_limit_seconds=20),
        clock=lambda: now[0],
    )
    now[0] = 80.0
    with pytest.raises(
        ExplorationBudgetExhausted,
        match="exploration_wall_clock_limit",
    ):
        budget.ensure_active()

    budget.begin_finalization(trigger="exploration_exhausted")
    now[0] = 99.0
    budget.ensure_active()


def test_pending_llm_reservation_blocks_phase_transition():
    budget = EpisodeBudget(
        _policy(),
        finalization_reserve=_reserve(),
    )
    budget.begin_llm_call(
        system="s",
        messages=[],
        requested_max_output_tokens=10,
    )

    with pytest.raises(BudgetInvariantError, match="pending LLM reservation"):
        budget.begin_finalization(trigger="invalid")


def test_partition_events_and_policy_payload_are_explicit():
    events = []
    reserve = _reserve()
    budget = EpisodeBudget(
        _policy(),
        finalization_reserve=reserve,
        event_sink=events.append,
    )
    budget.begin_finalization(trigger="agent_requested")

    assert budget.policy_payload()["finalization_reserve"] == reserve.to_dict()
    assert events == [
        {
            "event": "budget_finalization_started",
            "trigger": "agent_requested",
            "exploration_exhausted_reason": None,
            "start_usage": {
                "llm_calls": 0,
                "total_tokens": 0,
                "code_executions": 0,
                "tool_calls": 0,
            },
            "budget_phase": "finalization",
        }
    ]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_queries": 0}, "positive integer"),
        ({"max_queries": 1.5}, "positive integer"),
        ({"finalization_reserve_queries": -1}, "non-negative integer"),
        (
            {"max_queries": 2, "finalization_reserve_queries": 3},
            "exceeds the global",
        ),
        ({"feedback_numeric_decimals": -1}, "between 0 and 15"),
        ({"feedback_numeric_decimals": 16}, "between 0 and 15"),
    ],
)
def test_validation_query_policy_requires_explicit_valid_bounds(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validation_policy(**overrides)


def test_validation_query_reserve_requires_finalization_partition():
    with pytest.raises(ValueError, match="requires a finalization reserve"):
        EpisodeBudget(
            _policy(),
            validation_query_policy=_validation_policy(
                finalization_reserve_queries=1
            ),
        )


def test_validation_queries_charge_repeated_cached_and_failed_sources_independently():
    events = []
    policy = _validation_policy(max_queries=4)
    budget = EpisodeBudget(
        _policy(),
        validation_query_policy=policy,
        event_sink=events.append,
    )

    budget.consume_validation_query("validate")
    budget.consume_validation_query("validate")
    budget.consume_validation_query("cached_score")
    budget.consume_validation_query("failed_candidate")

    snapshot = budget.snapshot()["validation_query_budget"]
    assert snapshot == {
        "policy": {
            "policy_version": VALIDATION_QUERY_POLICY_VERSION,
            "max_queries": 4,
            "finalization_reserve_queries": 0,
            "feedback_numeric_decimals": 3,
        },
        "budget_phase": "exploration",
        "queries": 4,
        "remaining_queries": 0,
        "global_remaining_queries": 0,
        "queries_by_source": {
            "cached_score": 1,
            "failed_candidate": 1,
            "validate": 2,
        },
    }
    assert [event["query_number"] for event in events] == [1, 2, 3, 4]

    with pytest.raises(BudgetExhausted, match="max_validation_queries"):
        budget.consume_validation_query("over_cap")
    assert budget.validation_queries == 4
    assert "over_cap" not in budget.validation_queries_by_source


def test_validation_query_partition_preserves_one_terminal_replay_query():
    budget = EpisodeBudget(
        _policy(),
        finalization_reserve=_reserve(),
        validation_query_policy=_validation_policy(
            max_queries=3,
            finalization_reserve_queries=1,
        ),
    )

    budget.consume_validation_query("validate")
    budget.consume_validation_query("quick_validate")
    with pytest.raises(
        ExplorationBudgetExhausted,
        match="exploration_max_validation_queries",
    ):
        budget.consume_validation_query("exploration_overshoot")

    assert budget.validation_queries == 2
    assert budget.exhausted_reason is None
    budget.begin_finalization(trigger="query_cap")
    budget.consume_validation_query("protected_finalization")
    with pytest.raises(
        BudgetExhausted,
        match="finalization_max_validation_queries",
    ):
        budget.consume_validation_query("terminal_overshoot")

    snapshot = budget.snapshot()
    assert snapshot["exploration_limits"]["max_validation_queries"] == 2
    assert snapshot["finalization_usage"]["validation_queries"] == 1
    assert snapshot["validation_query_budget"]["queries"] == 3


def test_feedback_metric_precision_is_frozen_and_used_for_selection_values():
    budget = EpisodeBudget(
        _policy(),
        validation_query_policy=_validation_policy(
            feedback_numeric_decimals=3
        ),
    )

    assert budget.normalize_feedback_metric(0.12344) == 0.123
    assert budget.normalize_feedback_metric(0.12355) == 0.124
    assert budget.policy_payload()["validation_query_policy"] == (
        budget.validation_query_policy.to_dict()
    )
    with pytest.raises(BudgetInvariantError, match="finite"):
        budget.normalize_feedback_metric(float("nan"))
