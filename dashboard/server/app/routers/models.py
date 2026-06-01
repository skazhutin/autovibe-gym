"""Models registry API. Implemented in the next commit (services/model_store)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models() -> list[dict]:
    return []  # TODO(next commit): read data/models.json + health checks
