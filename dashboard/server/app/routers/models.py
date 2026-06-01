"""Models registry API: list/create/update/delete + connectivity health check."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import model_store

router = APIRouter(prefix="/models", tags=["models"])


class ModelPayload(BaseModel):
    name: str | None = None
    provider: str | None = None
    baseUrl: str | None = None
    apiKey: str | None = None
    apiKeyEnv: str | None = None
    ctx: int | None = None
    temp: float | None = None
    maxTokens: int | None = None


def _public(model: dict) -> dict:
    """Never leak the raw API key to the frontend."""
    out = dict(model)
    if out.get("apiKey"):
        out["apiKey"] = "********"
        out["hasApiKey"] = True
    else:
        out["hasApiKey"] = False
    return out


@router.get("")
def list_models() -> list[dict]:
    return [_public(m) for m in model_store.list_models()]


@router.get("/providers")
def providers() -> list[str]:
    return model_store.PROVIDERS


@router.post("")
def create(payload: ModelPayload) -> dict:
    if not payload.name:
        raise HTTPException(422, "name is required")
    return _public(model_store.create_model(payload.model_dump(exclude_none=True)))


@router.put("/{model_id}")
def update(model_id: str, payload: ModelPayload) -> dict:
    updated = model_store.update_model(model_id, payload.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return _public(updated)


@router.delete("/{model_id}")
def delete(model_id: str) -> dict:
    if not model_store.delete_model(model_id):
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"deleted": model_id}


@router.post("/{model_id}/health")
def health(model_id: str) -> dict:
    return model_store.check_health(model_id)
