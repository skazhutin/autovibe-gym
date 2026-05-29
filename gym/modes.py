from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMode:
    name: str
    notebook_enabled: bool = True
    runtime_feedback_enabled: bool = True
    contract_feedback_enabled: bool = True
    checklist_feedback_enabled: bool = True


ITERATIVE_NO_CHECKLIST = EpisodeMode(
    name="iterative_no_checklist",
    checklist_feedback_enabled=False,
)

GYM_WITH_CHECKLIST = EpisodeMode(name="gym_with_checklist")


EPISODE_MODES: dict[str, EpisodeMode] = {
    ITERATIVE_NO_CHECKLIST.name: ITERATIVE_NO_CHECKLIST,
    GYM_WITH_CHECKLIST.name: GYM_WITH_CHECKLIST,
}


def resolve_episode_mode(mode: str | EpisodeMode | None) -> EpisodeMode:
    if isinstance(mode, EpisodeMode):
        return mode
    if mode is None:
        return GYM_WITH_CHECKLIST
    try:
        return EPISODE_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported episode mode: {mode!r}") from exc
