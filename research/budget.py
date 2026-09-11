"""Pre-call episode budgets plus an opt-in Paper V2 finalization partition."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Callable, Mapping


VALIDATION_QUERY_POLICY_VERSION = "validation-query-v1"


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


@dataclass(frozen=True)
class FinalizationReservePolicy:
    """Resources withheld from exploration for an explicit terminal phase.

    This policy is opt-in and intentionally separate from the frozen Paper V1
    ``EpisodeBudgetPolicy`` payload. Zero means that a resource is unavailable
    during finalization, not that it may be borrowed from exploration.
    """

    total_token_limit: int = 0
    max_llm_calls: int = 0
    max_code_executions: int = 0
    max_tool_calls: int = 0
    wall_clock_limit_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "total_token_limit",
            "max_llm_calls",
            "max_code_executions",
            "max_tool_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        wall_limit = self.wall_clock_limit_seconds
        if (
            isinstance(wall_limit, bool)
            or not isinstance(wall_limit, (int, float))
            or not math.isfinite(float(wall_limit))
            or wall_limit <= 0
        ):
            raise ValueError("wall_clock_limit_seconds must be positive")
        if not any(
            (
                self.total_token_limit,
                self.max_llm_calls,
                self.max_code_executions,
                self.max_tool_calls,
            )
        ):
            raise ValueError("finalization reserve must protect at least one resource")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def validate_against(self, policy: EpisodeBudgetPolicy) -> None:
        limits = {
            "total_token_limit": policy.total_token_limit,
            "max_llm_calls": policy.max_llm_calls,
            "max_code_executions": policy.max_code_executions,
            "max_tool_calls": policy.max_tool_calls,
            "wall_clock_limit_seconds": policy.wall_clock_limit_seconds,
        }
        for name, global_limit in limits.items():
            reserve = getattr(self, name)
            if reserve > global_limit:
                raise ValueError(f"{name} reserve exceeds the global episode limit")


@dataclass(frozen=True)
class ValidationQueryPolicy:
    """Explicit cap and public precision for feedback-validation information.

    Values intentionally have no defaults. They must be selected on excluded
    development tasks and frozen before an experimental condition can enable
    this policy.
    """

    max_queries: int
    finalization_reserve_queries: int
    feedback_numeric_decimals: int

    def __post_init__(self) -> None:
        if type(self.max_queries) is not int or self.max_queries <= 0:
            raise ValueError("max_queries must be a positive integer")
        if (
            type(self.finalization_reserve_queries) is not int
            or self.finalization_reserve_queries < 0
        ):
            raise ValueError(
                "finalization_reserve_queries must be a non-negative integer"
            )
        if self.finalization_reserve_queries > self.max_queries:
            raise ValueError(
                "finalization_reserve_queries exceeds the global validation-query limit"
            )
        if (
            type(self.feedback_numeric_decimals) is not int
            or not 0 <= self.feedback_numeric_decimals <= 15
        ):
            raise ValueError("feedback_numeric_decimals must be between 0 and 15")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "policy_version": VALIDATION_QUERY_POLICY_VERSION,
            "max_queries": self.max_queries,
            "finalization_reserve_queries": self.finalization_reserve_queries,
            "feedback_numeric_decimals": self.feedback_numeric_decimals,
        }


class BudgetExhausted(RuntimeError):
    def __init__(self, reason: str, snapshot: Mapping[str, Any]):
        self.reason = reason
        self.snapshot = dict(snapshot)
        super().__init__(f"Episode budget exhausted: {reason}")


class BudgetInvariantError(RuntimeError):
    pass


class ExplorationBudgetExhausted(BudgetExhausted):
    """Exploration stopped at the protected boundary; reserve remains intact."""


class EpisodeBudget:
    """Mutable usage state shared by one arm's LLM, executor, and tools.

    With no ``finalization_reserve`` this retains the Paper V1 global-budget
    contract and serialization shape. A reserve enables a one-way V2 phase
    transition while keeping the same global upper caps.
    """

    def __init__(
        self,
        policy: EpisodeBudgetPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        event_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        finalization_reserve: FinalizationReservePolicy | None = None,
        validation_query_policy: ValidationQueryPolicy | None = None,
    ):
        if finalization_reserve is not None:
            finalization_reserve.validate_against(policy)
        if (
            validation_query_policy is not None
            and validation_query_policy.finalization_reserve_queries > 0
            and finalization_reserve is None
        ):
            raise ValueError(
                "validation-query finalization reserve requires a finalization reserve"
            )
        self.policy = policy
        self.finalization_reserve = finalization_reserve
        self.validation_query_policy = validation_query_policy
        self._clock = clock
        self._started_at = clock()
        self._event_sink = event_sink
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.code_executions = 0
        self.tool_calls = 0
        self.validation_queries = 0
        self.validation_queries_by_source: dict[str, int] = {}
        self.exhausted_reason: str | None = None
        self.phase = "exploration"
        self.exploration_exhausted_reason: str | None = None
        self._finalization_started_at: float | None = None
        self._finalization_start_usage: dict[str, int] | None = None
        self._pending_input_reservation = 0
        self._pending_output_reservation = 0

    def set_event_sink(self, sink: Callable[[Mapping[str, Any]], Any] | None) -> None:
        self._event_sink = sink

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def has_finalization_reserve(self) -> bool:
        return self.finalization_reserve is not None

    @property
    def has_validation_query_policy(self) -> bool:
        return self.validation_query_policy is not None

    def policy_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = self.policy.to_dict()
        if self.finalization_reserve is not None:
            payload["finalization_reserve"] = self.finalization_reserve.to_dict()
        if self.validation_query_policy is not None:
            payload["validation_query_policy"] = (
                self.validation_query_policy.to_dict()
            )
        return payload

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
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
        if self.finalization_reserve is not None:
            snapshot.update(
                {
                    "budget_phase": self.phase,
                    "finalization_reserve": self.finalization_reserve.to_dict(),
                    "exploration_limits": self._exploration_limits(),
                    "exploration_exhausted": self.exploration_exhausted_reason
                    is not None,
                    "exploration_exhausted_reason": self.exploration_exhausted_reason,
                    "finalization_started": self._finalization_start_usage is not None,
                    "finalization_usage": self._finalization_usage(),
                }
            )
        if self.validation_query_policy is not None:
            snapshot["validation_query_budget"] = self._validation_query_snapshot()
        return snapshot

    def ensure_active(self) -> None:
        if self.exhausted_reason:
            raise BudgetExhausted(self.exhausted_reason, self.snapshot())
        if self.finalization_reserve is None:
            if self.elapsed_seconds() >= self.policy.wall_clock_limit_seconds:
                self._exhaust("wall_clock_limit")
            return

        if self.phase == "exploration":
            if self.exploration_exhausted_reason:
                raise ExplorationBudgetExhausted(
                    self.exploration_exhausted_reason,
                    self.snapshot(),
                )
            if (
                self.elapsed_seconds()
                >= self.policy.wall_clock_limit_seconds
                - self.finalization_reserve.wall_clock_limit_seconds
            ):
                self._exhaust_exploration("exploration_wall_clock_limit")
            return

        if self._finalization_started_at is None:
            raise BudgetInvariantError("Finalization phase has no start timestamp")
        finalization_elapsed = max(0.0, self._clock() - self._finalization_started_at)
        if finalization_elapsed >= self.finalization_reserve.wall_clock_limit_seconds:
            self._exhaust("finalization_wall_clock_limit")
        if self.elapsed_seconds() >= self.policy.wall_clock_limit_seconds:
            self._exhaust("wall_clock_limit")

    def begin_finalization(self, *, trigger: str) -> None:
        if self.finalization_reserve is None:
            raise BudgetInvariantError("No finalization reserve is configured")
        if self.phase == "finalization":
            return
        if self.exhausted_reason:
            raise BudgetExhausted(self.exhausted_reason, self.snapshot())
        if self.elapsed_seconds() >= self.policy.wall_clock_limit_seconds:
            self._exhaust("wall_clock_limit")
        if self._pending_input_reservation or self._pending_output_reservation:
            raise BudgetInvariantError(
                "Cannot enter finalization with a pending LLM reservation"
            )
        if not isinstance(trigger, str) or not trigger.strip():
            raise BudgetInvariantError("Finalization trigger must not be empty")
        self.phase = "finalization"
        self._finalization_started_at = self._clock()
        self._finalization_start_usage = {
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "code_executions": self.code_executions,
            "tool_calls": self.tool_calls,
        }
        if self.validation_query_policy is not None:
            self._finalization_start_usage["validation_queries"] = (
                self.validation_queries
            )
        self._emit(
            {
                "event": "budget_finalization_started",
                "trigger": trigger.strip(),
                "exploration_exhausted_reason": self.exploration_exhausted_reason,
                "start_usage": dict(self._finalization_start_usage),
            }
        )

    def begin_llm_call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        requested_max_output_tokens: int,
    ) -> int:
        self.ensure_active()
        self._ensure_counter_available(
            resource="llm_calls",
            current=self.llm_calls,
            global_limit=self.policy.max_llm_calls,
            reserve=(
                self.finalization_reserve.max_llm_calls
                if self.finalization_reserve is not None
                else 0
            ),
            global_reason="max_llm_calls",
        )
        output_limit = min(
            max(1, int(requested_max_output_tokens)),
            self.policy.max_output_tokens_per_call,
        )
        input_reservation = conservative_input_token_bound(system, messages)
        self._ensure_token_reservation_available(input_reservation + output_limit)
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
        self._ensure_counter_available(
            resource="code_executions",
            current=self.code_executions,
            global_limit=self.policy.max_code_executions,
            reserve=(
                self.finalization_reserve.max_code_executions
                if self.finalization_reserve is not None
                else 0
            ),
            global_reason="max_code_executions",
        )
        self.code_executions += 1
        self._emit(
            {
                "event": "budget_code_execution_consumed",
                "code_executions": self.code_executions,
            }
        )

    def consume_tool_call(self, tool: str) -> None:
        self.ensure_active()
        self._ensure_counter_available(
            resource="tool_calls",
            current=self.tool_calls,
            global_limit=self.policy.max_tool_calls,
            reserve=(
                self.finalization_reserve.max_tool_calls
                if self.finalization_reserve is not None
                else 0
            ),
            global_reason="max_tool_calls",
        )
        self.tool_calls += 1
        self._emit(
            {
                "event": "budget_tool_call_consumed",
                "tool": tool,
                "tool_calls": self.tool_calls,
            }
        )

    def consume_validation_query(self, source: str) -> None:
        """Charge one validation-information access before it is evaluated.

        Repeated, failed, cached, and same-revision accesses are deliberately
        charged independently. The caller must invoke this immediately before
        any feedback-validation features, labels, predictions, diagnostics, or
        score are inspected.
        """

        policy = self.validation_query_policy
        if policy is None:
            raise BudgetInvariantError(
                "No validation-query policy is configured"
            )
        if not isinstance(source, str) or not source.strip():
            raise BudgetInvariantError("Validation-query source must not be empty")
        normalized_source = source.strip()
        if len(normalized_source) > 128:
            raise BudgetInvariantError(
                "Validation-query source must be at most 128 characters"
            )

        self.ensure_active()
        if self.finalization_reserve is None:
            if self.validation_queries >= policy.max_queries:
                self._exhaust("max_validation_queries")
        elif self.phase == "exploration":
            exploration_limit = (
                policy.max_queries - policy.finalization_reserve_queries
            )
            if self.validation_queries >= exploration_limit:
                self._exhaust_exploration("exploration_max_validation_queries")
        else:
            if self._finalization_start_usage is None:
                raise BudgetInvariantError(
                    "Finalization phase has no usage snapshot"
                )
            phase_used = (
                self.validation_queries
                - self._finalization_start_usage["validation_queries"]
            )
            if phase_used >= policy.finalization_reserve_queries:
                self._exhaust("finalization_max_validation_queries")
            if self.validation_queries >= policy.max_queries:
                self._exhaust("max_validation_queries")

        self.validation_queries += 1
        self.validation_queries_by_source[normalized_source] = (
            self.validation_queries_by_source.get(normalized_source, 0) + 1
        )
        self._emit(
            {
                "event": "budget_validation_query_consumed",
                "query_number": self.validation_queries,
                "source": normalized_source,
                "remaining_queries": self.validation_queries_remaining(),
            }
        )

    def validation_queries_remaining(self) -> int:
        policy = self.validation_query_policy
        if policy is None:
            raise BudgetInvariantError(
                "No validation-query policy is configured"
            )
        if self.finalization_reserve is None:
            limit = policy.max_queries
        elif self.phase == "exploration":
            limit = policy.max_queries - policy.finalization_reserve_queries
        else:
            if self._finalization_start_usage is None:
                raise BudgetInvariantError(
                    "Finalization phase has no usage snapshot"
                )
            phase_used = (
                self.validation_queries
                - self._finalization_start_usage["validation_queries"]
            )
            return max(0, policy.finalization_reserve_queries - phase_used)
        return max(0, limit - self.validation_queries)

    def exploration_boundary_resources(self) -> tuple[str, ...]:
        """Return resources whose exploration allowance is exactly exhausted.

        This is read-only introspection for a separately frozen stopping policy.
        It does not choose which resources should stop an arm and does not
        transition the budget phase.
        """

        if self.finalization_reserve is None or self.phase != "exploration":
            return ()
        limits = self._exploration_limits()
        boundary: list[str] = []
        if self.total_tokens >= int(limits["total_token_limit"]):
            boundary.append("total_tokens")
        if self.llm_calls >= int(limits["max_llm_calls"]):
            boundary.append("llm_calls")
        if self.code_executions >= int(limits["max_code_executions"]):
            boundary.append("code_executions")
        if self.tool_calls >= int(limits["max_tool_calls"]):
            boundary.append("tool_calls")
        if self.elapsed_seconds() >= float(limits["wall_clock_limit_seconds"]):
            boundary.append("wall_clock_seconds")
        if (
            self.validation_query_policy is not None
            and self.validation_queries
            >= int(limits["max_validation_queries"])
        ):
            boundary.append("validation_queries")
        return tuple(boundary)

    def normalize_feedback_metric(self, value: Any) -> float:
        """Return the score at the frozen public precision used for selection."""

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise BudgetInvariantError(
                "Validation metric must be finite"
            ) from exc
        if not math.isfinite(numeric):
            raise BudgetInvariantError("Validation metric must be finite")
        policy = self.validation_query_policy
        if policy is None:
            return numeric
        quantum = Decimal(1).scaleb(-policy.feedback_numeric_decimals)
        try:
            normalized = Decimal(str(numeric)).quantize(
                quantum,
                rounding=ROUND_HALF_EVEN,
            )
        except InvalidOperation as exc:
            raise BudgetInvariantError(
                "Validation metric could not be normalized"
            ) from exc
        return float(normalized)

    def mark_exhausted(self, reason: str) -> None:
        if not self.exhausted_reason:
            self.exhausted_reason = reason
            self._emit({"event": "budget_exhausted", "reason": reason})

    def _exhaust(self, reason: str) -> None:
        self.mark_exhausted(reason)
        raise BudgetExhausted(reason, self.snapshot())

    def _exhaust_exploration(self, reason: str) -> None:
        if not self.exploration_exhausted_reason:
            self.exploration_exhausted_reason = reason
            self._emit({"event": "budget_exploration_exhausted", "reason": reason})
        raise ExplorationBudgetExhausted(reason, self.snapshot())

    def _ensure_counter_available(
        self,
        *,
        resource: str,
        current: int,
        global_limit: int,
        reserve: int,
        global_reason: str,
    ) -> None:
        if self.finalization_reserve is None:
            if current >= global_limit:
                self._exhaust(global_reason)
            return
        if self.phase == "exploration":
            if current >= global_limit - reserve:
                self._exhaust_exploration(f"exploration_{global_reason}")
            return
        if self._finalization_start_usage is None:
            raise BudgetInvariantError("Finalization phase has no usage snapshot")
        phase_used = current - self._finalization_start_usage[resource]
        if phase_used >= reserve:
            self._exhaust(f"finalization_{global_reason}")
        if current >= global_limit:
            self._exhaust(global_reason)

    def _ensure_token_reservation_available(self, reservation: int) -> None:
        projected_global = self.total_tokens + reservation
        if projected_global > self.policy.total_token_limit:
            self._exhaust("pre_call_token_reservation")
        if self.finalization_reserve is None:
            return
        reserve = self.finalization_reserve.total_token_limit
        if self.phase == "exploration":
            exploration_limit = self.policy.total_token_limit - reserve
            if projected_global > exploration_limit:
                self._exhaust_exploration(
                    "exploration_pre_call_token_reservation"
                )
            return
        if self._finalization_start_usage is None:
            raise BudgetInvariantError("Finalization phase has no usage snapshot")
        phase_used = self.total_tokens - self._finalization_start_usage["total_tokens"]
        if phase_used + reservation > reserve:
            self._exhaust("finalization_pre_call_token_reservation")

    def _exploration_limits(self) -> dict[str, int | float]:
        if self.finalization_reserve is None:
            raise BudgetInvariantError("No finalization reserve is configured")
        limits: dict[str, int | float] = {
            "total_token_limit": self.policy.total_token_limit
            - self.finalization_reserve.total_token_limit,
            "max_llm_calls": self.policy.max_llm_calls
            - self.finalization_reserve.max_llm_calls,
            "max_code_executions": self.policy.max_code_executions
            - self.finalization_reserve.max_code_executions,
            "max_tool_calls": self.policy.max_tool_calls
            - self.finalization_reserve.max_tool_calls,
            "wall_clock_limit_seconds": self.policy.wall_clock_limit_seconds
            - self.finalization_reserve.wall_clock_limit_seconds,
        }
        if self.validation_query_policy is not None:
            limits["max_validation_queries"] = (
                self.validation_query_policy.max_queries
                - self.validation_query_policy.finalization_reserve_queries
            )
        return limits

    def _finalization_usage(self) -> dict[str, int | float]:
        if self._finalization_start_usage is None:
            usage: dict[str, int | float] = {
                "llm_calls": 0,
                "total_tokens": 0,
                "code_executions": 0,
                "tool_calls": 0,
                "wall_clock_seconds": 0.0,
            }
            if self.validation_query_policy is not None:
                usage["validation_queries"] = 0
            return usage
        elapsed = 0.0
        if self._finalization_started_at is not None:
            elapsed = max(0.0, self._clock() - self._finalization_started_at)
        usage: dict[str, int | float] = {
            "llm_calls": self.llm_calls
            - self._finalization_start_usage["llm_calls"],
            "total_tokens": self.total_tokens
            - self._finalization_start_usage["total_tokens"],
            "code_executions": self.code_executions
            - self._finalization_start_usage["code_executions"],
            "tool_calls": self.tool_calls
            - self._finalization_start_usage["tool_calls"],
            "wall_clock_seconds": round(elapsed, 6),
        }
        if self.validation_query_policy is not None:
            usage["validation_queries"] = (
                self.validation_queries
                - self._finalization_start_usage["validation_queries"]
            )
        return usage

    def _validation_query_snapshot(self) -> dict[str, Any]:
        policy = self.validation_query_policy
        if policy is None:
            raise BudgetInvariantError(
                "No validation-query policy is configured"
            )
        return {
            "policy": policy.to_dict(),
            "budget_phase": self.phase,
            "queries": self.validation_queries,
            "remaining_queries": self.validation_queries_remaining(),
            "global_remaining_queries": max(
                0, policy.max_queries - self.validation_queries
            ),
            "queries_by_source": {
                source: self.validation_queries_by_source[source]
                for source in sorted(self.validation_queries_by_source)
            },
        }

    def _clear_reservation(self) -> None:
        self._pending_input_reservation = 0
        self._pending_output_reservation = 0

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._event_sink is not None:
            payload = dict(event)
            if self.finalization_reserve is not None:
                payload.setdefault("budget_phase", self.phase)
            self._event_sink(payload)


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
