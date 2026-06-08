from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMode:
    name: str
    notebook_enabled: bool = True
    runtime_feedback_enabled: bool = True
    contract_feedback_enabled: bool = True
    checklist_feedback_enabled: bool = True


FREE_GYM = EpisodeMode(
    name="free_gym",
    checklist_feedback_enabled=False,
)

DIRECTIVE_GYM = EpisodeMode(name="directive_gym")


EPISODE_MODES: dict[str, EpisodeMode] = {
    FREE_GYM.name: FREE_GYM,
    DIRECTIVE_GYM.name: DIRECTIVE_GYM,
}


def resolve_episode_mode(mode: str | EpisodeMode | None) -> EpisodeMode:
    if isinstance(mode, EpisodeMode):
        return mode
    if mode is None:
        return DIRECTIVE_GYM
    try:
        return EPISODE_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported episode mode: {mode!r}") from exc
