"""Datasets API. Implemented in the next commit (services/dataset_store)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> list[dict]:
    return []  # TODO(next commit): read datasets/<name>/prepared/meta.json
