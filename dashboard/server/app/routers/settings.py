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
    "default_episode": "directive_gym",
    "date_format": "mdy",
    "theme": "light",
    "accent": "#FFDD2D",
    "radius": 18,
    # Remote execution: run the gym ON the server over SSH (site stays local).
    "remote_enabled": False,
    "remote_ssh": "",          # user@host  (e.g. booml@10.8.52.11)
    "remote_ssh_opts": "",     # extra ssh opts, e.g. "-p 2222"
    "remote_repo": "",         # server repo path, e.g. /home/booml/autovibe-gym-current
    "remote_python": "",       # server venv python, e.g. /home/booml/autovibe-gym/.venv/bin/python
    "remote_runs_dir": "",     # server scratch dir for run workspaces, e.g. /home/booml/dash_runs
    "remote_password": "",     # optional; prefer an SSH key. Stored only in local data/.
}


class SettingsPayload(BaseModel):
    mlflow_tracking_uri: str | None = None
    datasets_dir: str | None = None
    default_mode: str | None = None
    default_episode: str | None = None
    date_format: str | None = None
    theme: str | None = None
    accent: str | None = None
    radius: int | None = None
    remote_enabled: bool | None = None
    remote_ssh: str | None = None
    remote_ssh_opts: str | None = None
    remote_repo: str | None = None
    remote_python: str | None = None
    remote_runs_dir: str | None = None
    remote_password: str | None = None


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
    data = _load()
    if data.get("remote_password"):
        data = {**data, "remote_password": "********", "remote_has_password": True}
    else:
        data["remote_has_password"] = False
    return data


@router.post("/settings/remote-check")
def remote_check() -> dict:
    """Probe SSH connectivity + that the server repo/gym are usable."""
    from ..services import remote_exec

    return remote_exec.check()


@router.put("/settings")
def update_settings(payload: SettingsPayload) -> dict:
    data = _load()
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    # Never overwrite the stored password with the masked placeholder.
    if updates.get("remote_password") == "********":
        updates.pop("remote_password")
    data.update(updates)
    _save(data)
    return read_settings()
