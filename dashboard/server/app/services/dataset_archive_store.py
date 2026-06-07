"""Persistent set of archived dataset IDs stored in data/datasets_archive.json."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings


def _path() -> Path:
    return get_settings().data_dir / "datasets_archive.json"


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


def is_archived(dataset_id: str) -> bool:
    return dataset_id in _load()


def archive(dataset_ids: list[str]) -> None:
    ids = _load()
    ids.update(dataset_ids)
    _save(ids)


def unarchive(dataset_ids: list[str]) -> None:
    ids = _load()
    ids.difference_update(dataset_ids)
    _save(ids)


def list_archived() -> set[str]:
    return _load()
