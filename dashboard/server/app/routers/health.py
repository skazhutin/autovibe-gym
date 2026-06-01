"""Health / readiness endpoint consumed by the header status pill."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings
from ..services import model_store

router = APIRouter(tags=["health"])


@router.get("/server-health")
def server_health() -> dict:
    """Reachability of the configured LLM server(s) — powers the header pill."""
    return model_store.server_health()


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "online",
        "service": "autovibe-gym-dashboard",
        "repo_root": str(s.repo_root),
        "mlflow_tracking_uri": s.mlflow_tracking_uri,
        "mlflow_store_present": s.mlflow_db.exists(),
        "datasets_dir_present": s.datasets_dir.exists(),
        "python_bin": s.python_bin,
        "python_bin_present": __import__("os").path.exists(s.python_bin),
    }
