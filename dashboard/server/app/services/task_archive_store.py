"""Persistent set of archived task IDs stored in data/tasks_archive.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings


def _path() -> Path:
    return get_settings().data_dir / "tasks_archive.json"


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


def is_archived(task_id: str) -> bool:
    return task_id in _load()


def archive(task_ids: list[str]) -> None:
    ids = _load()
    ids.update(task_ids)
    _save(ids)


def unarchive(task_ids: list[str]) -> None:
    ids = _load()
    ids.difference_update(task_ids)
    _save(ids)


def list_archived() -> set[str]:
    return _load()
