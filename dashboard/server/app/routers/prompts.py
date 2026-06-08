"""System-prompt presets API.

Endpoints:
    GET    /api/prompts             — list (summary; `default` always first)
    GET    /api/prompts/default     — the canonical (code-derived) preset
    GET    /api/prompts/{id}        — full preset detail with resolved blocks
    POST   /api/prompts             — create or update a preset
    DELETE /api/prompts/{id}        — delete a user preset (default is reserved)

The `default` preset is synthesized from gym.prompts at read time, never
stored on disk. Locked-block overrides (kernel_vars and any future locked
blocks) are rejected with HTTP 422 because they break the agent↔parser
contract.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import prompt_store

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptPayload(BaseModel):
    """Create / update preset payload.

    Only block overrides are stored — empty ``blocks`` means the preset
    inherits everything from the code defaults (useful as a named "clone" of
    default). ``thoughts_on`` / ``thoughts_off`` may be ``None`` to fall back
    to the default thoughts-toggle text.
    """

    id: str = Field(..., min_length=1, max_length=40)
    name: str = Field(..., min_length=1, max_length=80)
    blocks: dict[str, str] = Field(default_factory=dict)
    thoughts_on: str | None = None
    thoughts_off: str | None = None


@router.get("")
def list_prompts() -> dict:
    return {
        "items": prompt_store.list_presets(),
        "default_id": prompt_store.DEFAULT_PRESET_ID,
    }


@router.get("/default")
def get_default() -> dict:
    return prompt_store.get_default_preset()


@router.get("/{preset_id}")
def get_one(preset_id: str) -> dict:
    try:
        return prompt_store.get_preset(preset_id)
    except KeyError:
        raise HTTPException(404, f"prompt preset '{preset_id}' not found")


@router.post("")
def upsert(payload: PromptPayload) -> dict:
    try:
        return prompt_store.save_preset(payload.model_dump())
    except prompt_store.PresetValidationError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/{preset_id}")
def delete(preset_id: str) -> dict:
    try:
        existed = prompt_store.delete_preset(preset_id)
    except prompt_store.PresetValidationError as exc:
        raise HTTPException(422, str(exc))
    if not existed:
        raise HTTPException(404, f"prompt preset '{preset_id}' not found")
    return {"deleted": preset_id}
