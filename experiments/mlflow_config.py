import os
from pathlib import Path


def configure_mlflow_tracking(mlflow_module, default_uri: str | None = None) -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or default_uri or local_tracking_uri()
    if _looks_like_placeholder(tracking_uri):
        tracking_uri = local_tracking_uri()
    mlflow_module.set_tracking_uri(tracking_uri)
    return tracking_uri


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return "<" in normalized or ">" in normalized or "server-ip" in normalized


def local_tracking_uri() -> str:
    db_path = Path(os.getenv("AUTOVIBE_MLFLOW_DB", "mlflow.db")).expanduser()
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    return f"sqlite:///{db_path.resolve().as_posix()}"
