"""Global, pre-call-enforced episode budgets for Paper V1 arms."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class EpisodeBudgetPolicy:
    total_token_limit: int = 64_000
    max_output_tokens_per_call: int = 4_096
    max_llm_calls: int = 12
    max_code_executions: int = 20
    max_tool_calls: int = 20
    wall_clock_limit_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class BudgetExhausted(RuntimeError):
    def __init__(self, reason: str, snapshot: Mapping[str, Any]):
        self.reason = reason
        self.snapshot = dict(snapshot)
        super().__init__(f"Episode budget exhausted: {reason}")


class BudgetInvariantError(RuntimeError):
    pass


class EpisodeBudget:
    """Mutable usage state shared by one arm's LLM, executor, and tools."""

    def __init__(
        self,
        policy: EpisodeBudgetPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        event_sink: Callable[[Mapping[str, Any]], Any] | None = None,
    ):
        self.policy = policy
        self._clock = clock
        self._started_at = clock()
        self._event_sink = event_sink
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.code_executions = 0
        self.tool_calls = 0
        self.exhausted_reason: str | None = None
        self._pending_input_reservation = 0
        self._pending_output_reservation = 0

    def set_event_sink(self, sink: Callable[[Mapping[str, Any]], Any] | None) -> None:
        self._event_sink = sink

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "code_executions": self.code_executions,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed_seconds(), 6),
            "exhausted": self.exhausted_reason is not None,
            "exhausted_reason": self.exhausted_reason,
        }

    def ensure_active(self) -> None:
        if self.exhausted_reason:
            raise BudgetExhausted(self.exhausted_reason, self.snapshot())
        if self.elapsed_seconds() >= self.policy.wall_clock_limit_seconds:
            self._exhaust("wall_clock_limit")

    def begin_llm_call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        requested_max_output_tokens: int,
    ) -> int:
        self.ensure_active()
        if self.llm_calls >= self.policy.max_llm_calls:
            self._exhaust("max_llm_calls")
        output_limit = min(
            max(1, int(requested_max_output_tokens)),
            self.policy.max_output_tokens_per_call,
        )
        input_reservation = conservative_input_token_bound(system, messages)
        if self.total_tokens + input_reservation + output_limit > self.policy.total_token_limit:
            self._exhaust("pre_call_token_reservation")
        self.llm_calls += 1
        self._pending_input_reservation = input_reservation
        self._pending_output_reservation = output_limit
        self._emit(
            {
                "event": "budget_llm_call_reserved",
                "llm_call": self.llm_calls,
                "input_token_reservation": input_reservation,
                "max_output_tokens": output_limit,
            }
        )
        return output_limit

    def finish_llm_call(self, response: Any) -> None:
        input_tokens = max(0, int(getattr(response, "input_tokens", 0) or 0))
        output_tokens = max(0, int(getattr(response, "output_tokens", 0) or 0))
        reasoning_tokens = max(0, int(getattr(response, "reasoning_tokens", 0) or 0))
        actual = input_tokens + output_tokens + reasoning_tokens
        reserved = self._pending_input_reservation + self._pending_output_reservation
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens
        self._clear_reservation()
        self._emit(
            {
                "event": "budget_llm_call_completed",
                "llm_call": self.llm_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "actual_tokens": actual,
                "reserved_tokens": reserved,
            }
        )
        if actual > reserved:
            raise BudgetInvariantError(
                f"Provider usage {actual} exceeded conservative reservation {reserved}"
            )
        if self.total_tokens > self.policy.total_token_limit:
            raise BudgetInvariantError("Provider usage exceeded the global token limit")

    def fail_llm_call(self, error: BaseException) -> None:
        self._clear_reservation()
        self._emit(
            {
                "event": "budget_llm_call_failed",
                "llm_call": self.llm_calls,
                "error_type": type(error).__name__,
            }
        )

    def consume_code_execution(self) -> None:
        self.ensure_active()
        if self.code_executions >= self.policy.max_code_executions:
            self._exhaust("max_code_executions")
        self.code_executions += 1
        self._emit(
            {
                "event": "budget_code_execution_consumed",
                "code_executions": self.code_executions,
            }
        )

    def consume_tool_call(self, tool: str) -> None:
        self.ensure_active()
        if self.tool_calls >= self.policy.max_tool_calls:
            self._exhaust("max_tool_calls")
        self.tool_calls += 1
        self._emit(
            {
                "event": "budget_tool_call_consumed",
                "tool": tool,
                "tool_calls": self.tool_calls,
            }
        )

    def mark_exhausted(self, reason: str) -> None:
        if not self.exhausted_reason:
            self.exhausted_reason = reason
            self._emit({"event": "budget_exhausted", "reason": reason})

    def _exhaust(self, reason: str) -> None:
        self.mark_exhausted(reason)
        raise BudgetExhausted(reason, self.snapshot())

    def _clear_reservation(self) -> None:
        self._pending_input_reservation = 0
        self._pending_output_reservation = 0

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(dict(event))


class BudgetedLLMClient:
    def __init__(self, client: Any, budget: EpisodeBudget):
        self._client = client
        self.budget = budget

    def complete(self, *, system: str, messages: list[dict], model: str, max_tokens: int):
        output_limit = self.budget.begin_llm_call(
            system=system,
            messages=messages,
            requested_max_output_tokens=max_tokens,
        )
        try:
            response = self._client.complete(
                system=system,
                messages=messages,
                model=model,
                max_tokens=output_limit,
            )
        except Exception as error:
            self.budget.fail_llm_call(error)
            raise
        self.budget.finish_llm_call(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class BudgetedExecutor:
    def __init__(self, executor: Any, budget: EpisodeBudget):
        self._executor = executor
        self.budget = budget

    def run(self, code: str, namespace: dict | None = None):
        self.budget.consume_code_execution()
        return self._executor.run(code, namespace)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


def conservative_input_token_bound(system: str, messages: list[dict[str, Any]]) -> int:
    """Provider-neutral upper reservation based on UTF-8 bytes plus chat framing."""
    payload_bytes = len(str(system).encode("utf-8"))
    payload_bytes += sum(
        len(str(message.get("role", "")).encode("utf-8"))
        + len(str(message.get("content", "")).encode("utf-8"))
        for message in messages
    )
    framing = 128 + 64 * (len(messages) + 1)
    return max(1, payload_bytes + framing)
