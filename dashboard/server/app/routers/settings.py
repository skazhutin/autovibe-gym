"""User-editable dashboard settings (persisted to data/settings.json).

Holds connection info shown on the Settings screen: backend/MLflow URIs,
data paths, and appearance prefs (theme/accent/radius mirror the frontend but
are stored here too so the choice survives a browser reset)."""
from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import get_settings

router = APIRouter(tags=["settings"])

DEFAULTS = {
    "mlflow_tracking_uri": "",
    "datasets_dir": "",
    "default_mode": "local",
    "default_episode": "gym_with_checklist",
    "theme": "light",
    "accent": "#FFDD2D",
    "radius": 18,
}


class SettingsPayload(BaseModel):
    mlflow_tracking_uri: str | None = None
    datasets_dir: str | None = None
    default_mode: str | None = None
    default_episode: str | None = None
    theme: str | None = None
    accent: str | None = None
    radius: int | None = None


def _load() -> dict:
    s = get_settings()
    data = dict(DEFAULTS)
    if s.settings_file.exists():
        try:
            data.update(json.loads(s.settings_file.read_text("utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    # Fill resolved live defaults if user has not overridden them.
    data["mlflow_tracking_uri"] = data["mlflow_tracking_uri"] or s.mlflow_tracking_uri
    data["datasets_dir"] = data["datasets_dir"] or str(s.datasets_dir)
    return data


def _save(data: dict) -> None:
    s = get_settings()
    s.settings_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


@router.get("/settings")
def read_settings() -> dict:
    return _load()


@router.put("/settings")
def update_settings(payload: SettingsPayload) -> dict:
    data = _load()
    data.update({k: v for k, v in payload.model_dump().items() if v is not None})
    _save(data)
    return data
