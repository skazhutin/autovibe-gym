"""Persistent set of archived model IDs stored in data/models_archive.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings


def _path() -> Path:
    return get_settings().data_dir / "models_archive.json"


def _load() -> set[str]:
    p = _path()
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save(ids: set[str]) -> None:
    _path().write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


def is_archived(model_id: str) -> bool:
    return model_id in _load()


def archive(model_ids: list[str]) -> None:
    ids = _load()
    ids.update(model_ids)
    _save(ids)


def unarchive(model_ids: list[str]) -> None:
    ids = _load()
    ids.difference_update(model_ids)
    _save(ids)


def list_archived() -> set[str]:
    return _load()
