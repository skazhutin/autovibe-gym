"""Frozen, result-blind stopping policy for Paper V2 exploration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


STOPPING_POLICY_VERSION = "stopping-policy-v1"
RESERVE_BOUNDARY_RESOURCES = (
    "total_tokens",
    "llm_calls",
    "code_executions",
    "tool_calls",
    "wall_clock_seconds",
    "validation_queries",
)
STOPPING_REASONS = (
    "exploration_pool_exhausted",
    "reserve_boundary_reached",
    "no_validation_improvement",
    "agent_requested_finalization",
    "unrecoverable_contract_failure",
)

StoppingReason = Literal[
    "exploration_pool_exhausted",
    "reserve_boundary_reached",
    "no_validation_improvement",
    "agent_requested_finalization",
    "unrecoverable_contract_failure",
]
StoppingTransition = Literal["finalize", "terminate"]

_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "reserve_boundary_resources",
        "no_improvement_patience",
        "stop_on_exploration_exhausted",
        "stop_on_agent_finalize_request",
        "stop_on_unrecoverable_failure",
    }
)


class StoppingPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenStoppingPolicy:
    """Every field is explicit; the implementation selects no thresholds."""

    reserve_boundary_resources: tuple[str, ...]
    no_improvement_patience: int | None
    stop_on_exploration_exhausted: bool
    stop_on_agent_finalize_request: bool
    stop_on_unrecoverable_failure: bool

    def __post_init__(self) -> None:
        raw_resources = self.reserve_boundary_resources
        if not isinstance(raw_resources, (tuple, list)):
            raise StoppingPolicyError(
                "reserve_boundary_resources must be a list or tuple"
            )
        resources = tuple(raw_resources)
        if any(type(item) is not str for item in resources):
            raise StoppingPolicyError(
                "reserve_boundary_resources must contain strings"
            )
        unknown = set(resources) - set(RESERVE_BOUNDARY_RESOURCES)
        if unknown:
            raise StoppingPolicyError(
                f"Unknown reserve-boundary resources: {sorted(unknown)}"
            )
        if len(resources) != len(set(resources)):
            raise StoppingPolicyError(
                "reserve_boundary_resources must not contain duplicates"
            )
        canonical = tuple(
            resource
            for resource in RESERVE_BOUNDARY_RESOURCES
            if resource in resources
        )
        object.__setattr__(self, "reserve_boundary_resources", canonical)

        patience = self.no_improvement_patience
        if patience is not None and (
            type(patience) is not int or patience <= 0
        ):
            raise StoppingPolicyError(
                "no_improvement_patience must be null or a positive integer"
            )
        for name in (
            "stop_on_exploration_exhausted",
            "stop_on_agent_finalize_request",
            "stop_on_unrecoverable_failure",
        ):
            if type(getattr(self, name)) is not bool:
                raise StoppingPolicyError(f"{name} must be boolean")
        if not any(
            (
                canonical,
                patience is not None,
                self.stop_on_exploration_exhausted,
                self.stop_on_agent_finalize_request,
                self.stop_on_unrecoverable_failure,
            )
        ):
            raise StoppingPolicyError(
                "Stopping policy must enable at least one condition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STOPPING_POLICY_VERSION,
            "reserve_boundary_resources": list(
                self.reserve_boundary_resources
            ),
            "no_improvement_patience": self.no_improvement_patience,
            "stop_on_exploration_exhausted": (
                self.stop_on_exploration_exhausted
            ),
            "stop_on_agent_finalize_request": (
                self.stop_on_agent_finalize_request
            ),
            "stop_on_unrecoverable_failure": (
                self.stop_on_unrecoverable_failure
            ),
        }

    @property
    def policy_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenStoppingPolicy":
        if not isinstance(payload, Mapping) or set(payload) != _POLICY_KEYS:
            raise StoppingPolicyError("Stopping policy has an invalid shape")
        if payload["schema_version"] != STOPPING_POLICY_VERSION:
            raise StoppingPolicyError("Unsupported stopping-policy schema")
        resources = payload["reserve_boundary_resources"]
        if not isinstance(resources, list):
            raise StoppingPolicyError(
                "reserve_boundary_resources must be a JSON array"
            )
        return cls(
            reserve_boundary_resources=tuple(resources),
            no_improvement_patience=payload["no_improvement_patience"],
            stop_on_exploration_exhausted=payload[
                "stop_on_exploration_exhausted"
            ],
            stop_on_agent_finalize_request=payload[
                "stop_on_agent_finalize_request"
            ],
            stop_on_unrecoverable_failure=payload[
                "stop_on_unrecoverable_failure"
            ],
        )


@dataclass(frozen=True)
class StoppingDecision:
    reason: StoppingReason
    transition: StoppingTransition
    event_index: int
    incumbent_candidate_id: str | None
    no_improvement_streak: int
    boundary_resources: tuple[str, ...] = ()
    detail_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "transition": self.transition,
            "event_index": self.event_index,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "no_improvement_streak": self.no_improvement_streak,
            "boundary_resources": list(self.boundary_resources),
            "detail_code": self.detail_code,
        }


class StoppingController:
    """Deterministic first-trigger-wins controller for one episode."""

    def __init__(self, policy: FrozenStoppingPolicy) -> None:
        self.policy = policy
        self.no_improvement_streak = 0
        self.decision: StoppingDecision | None = None
        self.events: list[dict[str, Any]] = []

    def observe_validation_attempt(
        self,
        *,
        eligible: bool,
        improved: bool,
        candidate_id: str | None,
        incumbent_candidate_id: str | None,
    ) -> StoppingDecision | None:
        if type(eligible) is not bool or type(improved) is not bool:
            raise StoppingPolicyError("eligible and improved must be boolean")
        if improved and not eligible:
            raise StoppingPolicyError(
                "An ineligible validation attempt cannot be an improvement"
            )
        if eligible:
            self.no_improvement_streak = (
                0 if improved else self.no_improvement_streak + 1
            )
        event_index = self._record(
            "validation_attempt",
            {
                "eligible": eligible,
                "improved": improved,
                "candidate_id": candidate_id,
                "incumbent_candidate_id": incumbent_candidate_id,
                "no_improvement_streak": self.no_improvement_streak,
            },
        )
        patience = self.policy.no_improvement_patience
        if (
            self.decision is None
            and eligible
            and not improved
            and patience is not None
            and self.no_improvement_streak >= patience
        ):
            self._latch(
                reason="no_validation_improvement",
                transition="finalize",
                event_index=event_index,
                incumbent_candidate_id=incumbent_candidate_id,
            )
        return self.decision

    def observe_reserve_boundary(
        self,
        *,
        boundary_resources: tuple[str, ...] | list[str],
        incumbent_candidate_id: str | None,
    ) -> StoppingDecision | None:
        available = set(boundary_resources)
        unknown = available - set(RESERVE_BOUNDARY_RESOURCES)
        if unknown:
            raise StoppingPolicyError(
                f"Unknown boundary resources: {sorted(unknown)}"
            )
        matched = tuple(
            resource
            for resource in self.policy.reserve_boundary_resources
            if resource in available
        )
        event_index = self._record(
            "reserve_boundary",
            {
                "boundary_resources": list(
                    resource
                    for resource in RESERVE_BOUNDARY_RESOURCES
                    if resource in available
                ),
                "matched_resources": list(matched),
                "incumbent_candidate_id": incumbent_candidate_id,
            },
        )
        if self.decision is None and matched:
            self._latch(
                reason="reserve_boundary_reached",
                transition="finalize",
                event_index=event_index,
                incumbent_candidate_id=incumbent_candidate_id,
                boundary_resources=matched,
            )
        return self.decision

    def observe_exploration_exhausted(
        self,
        *,
        detail_code: str,
        incumbent_candidate_id: str | None,
    ) -> StoppingDecision | None:
        if not isinstance(detail_code, str) or not detail_code.strip():
            raise StoppingPolicyError("detail_code must not be empty")
        event_index = self._record(
            "exploration_exhausted",
            {
                "detail_code": detail_code.strip(),
                "incumbent_candidate_id": incumbent_candidate_id,
            },
        )
        if (
            self.decision is None
            and self.policy.stop_on_exploration_exhausted
        ):
            self._latch(
                reason="exploration_pool_exhausted",
                transition="finalize",
                event_index=event_index,
                incumbent_candidate_id=incumbent_candidate_id,
                detail_code=detail_code.strip(),
            )
        return self.decision

    def observe_agent_finalization_request(
        self,
        *,
        has_valid_incumbent: bool,
        incumbent_candidate_id: str | None,
    ) -> StoppingDecision | None:
        if type(has_valid_incumbent) is not bool:
            raise StoppingPolicyError("has_valid_incumbent must be boolean")
        event_index = self._record(
            "agent_finalization_request",
            {
                "has_valid_incumbent": has_valid_incumbent,
                "incumbent_candidate_id": incumbent_candidate_id,
            },
        )
        if (
            self.decision is None
            and self.policy.stop_on_agent_finalize_request
            and has_valid_incumbent
        ):
            self._latch(
                reason="agent_requested_finalization",
                transition="finalize",
                event_index=event_index,
                incumbent_candidate_id=incumbent_candidate_id,
            )
        return self.decision

    def observe_unrecoverable_failure(
        self,
        *,
        detail_code: str,
        incumbent_candidate_id: str | None,
    ) -> StoppingDecision | None:
        if not isinstance(detail_code, str) or not detail_code.strip():
            raise StoppingPolicyError("detail_code must not be empty")
        event_index = self._record(
            "unrecoverable_failure",
            {
                "detail_code": detail_code.strip(),
                "incumbent_candidate_id": incumbent_candidate_id,
            },
        )
        if self.decision is None and self.policy.stop_on_unrecoverable_failure:
            self._latch(
                reason="unrecoverable_contract_failure",
                transition="terminate",
                event_index=event_index,
                incumbent_candidate_id=incumbent_candidate_id,
                detail_code=detail_code.strip(),
            )
        return self.decision

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_version": STOPPING_POLICY_VERSION,
            "policy_hash": self.policy.policy_hash,
            "no_improvement_streak": self.no_improvement_streak,
            "decision": self.decision.to_dict() if self.decision else None,
            "events_observed": len(self.events),
        }

    def _record(self, event: str, payload: dict[str, Any]) -> int:
        event_index = len(self.events) + 1
        self.events.append(
            {
                "event_index": event_index,
                "event": event,
                **payload,
            }
        )
        return event_index

    def _latch(
        self,
        *,
        reason: StoppingReason,
        transition: StoppingTransition,
        event_index: int,
        incumbent_candidate_id: str | None,
        boundary_resources: tuple[str, ...] = (),
        detail_code: str | None = None,
    ) -> None:
        if self.decision is not None:
            return
        self.decision = StoppingDecision(
            reason=reason,
            transition=transition,
            event_index=event_index,
            incumbent_candidate_id=incumbent_candidate_id,
            no_improvement_streak=self.no_improvement_streak,
            boundary_resources=boundary_resources,
            detail_code=detail_code,
        )


def load_stopping_policy(path: str | Path) -> FrozenStoppingPolicy:
    policy_path = Path(path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise StoppingPolicyError(
            "Stopping policy path must be a regular file"
        )
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StoppingPolicyError("Could not read stopping policy") from exc
    return FrozenStoppingPolicy.from_dict(_strict_json(raw))


def _strict_json(raw: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StoppingPolicyError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise StoppingPolicyError(f"Non-standard JSON constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except StoppingPolicyError:
        raise
    except json.JSONDecodeError as exc:
        raise StoppingPolicyError("Stopping policy is not valid JSON") from exc


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
