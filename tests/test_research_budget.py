from dataclasses import dataclass

import pytest

from research.budget import (
    BudgetExhausted,
    BudgetInvariantError,
    BudgetedExecutor,
    BudgetedLLMClient,
    EpisodeBudget,
    EpisodeBudgetPolicy,
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
