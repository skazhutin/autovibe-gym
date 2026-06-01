"""Runtime configuration and path resolution for the dashboard backend.

The dashboard lives in ``<repo>/dashboard/server`` and drives the AutoVibe Gym
project that sits at the repository root (it launches the ``experiments.*``
runners and reads their MLflow tracking store).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# .../dashboard/server/app/config.py -> repo root is 4 parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"


class Settings:
    """Resolved paths and tunables. Mutable subset is persisted to settings.json."""

    def __init__(self) -> None:
        self.repo_root: Path = REPO_ROOT
        self.datasets_dir: Path = REPO_ROOT / "datasets"
        self.python_bin: str = os.getenv("AUTOVIBE_PYTHON", str(REPO_ROOT / ".venv" / "bin" / "python"))
        # MLflow store (file-backed sqlite at repo root by default).
        self.mlflow_db: Path = REPO_ROOT / "mlflow.db"
        self.mlflow_tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
        )
        self.mlruns_dir: Path = REPO_ROOT / "mlruns"
        # Dashboard-local state.
        self.data_dir: Path = DATA_DIR
        self.models_config: Path = DATA_DIR / "models.json"
        self.settings_file: Path = DATA_DIR / "settings.json"
        self.runs_dir: Path = DATA_DIR / "runs"
        self.uploads_dir: Path = DATA_DIR / "uploads"
        self.cors_origins: list[str] = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.runs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
