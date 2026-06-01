"""Runs API. Implemented in a following commit (MLflow store + launcher)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def list_runs() -> list[dict]:
    return []  # TODO: read MLflow runs + live subprocess registry
