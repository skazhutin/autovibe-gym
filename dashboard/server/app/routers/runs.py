"""Runs API: history (MLflow) + live launches, plus per-tab detail."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import dataset_store, mlflow_store, run_launcher

router = APIRouter(prefix="/runs", tags=["runs"])


class LaunchPayload(BaseModel):
    modelId: str | None = None
    model: str | None = None
    mode: str  # single | repeated | iterative | gym
    datasetId: str
    budgetMode: str = "local"  # local | cloud
    maxSteps: int | None = None
    maxTokens: int | None = None
    shots: int | None = None
    temp: float | None = None
    seed: int | None = None


def _target_col(dataset_id: str) -> str:
    ds = dataset_store.get_dataset(dataset_id)
    return (ds or {}).get("target") if ds and ds.get("target") != "—" else ""


def _resolve(run_id: str) -> tuple[str, dict | None]:
    """Return (mlflow_id_for_artifacts, live_meta_or_None)."""
    if run_id.startswith("live_"):
        live = run_launcher.get_live(run_id)
        if live is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return (live.get("mlflowId") or run_id, live)
    return (run_id, None)


@router.get("")
def list_runs() -> list[dict]:
    live = run_launcher.list_live()
    live_mlflow_ids = {m["mlflowId"] for m in live if m.get("mlflowId")}
    history = [r for r in mlflow_store.list_runs() if r["id"] not in live_mlflow_ids]
    return live + history


@router.post("")
def launch(payload: LaunchPayload) -> dict:
    ds = dataset_store.get_dataset(payload.datasetId)
    if ds is None:
        raise HTTPException(404, f"Dataset '{payload.datasetId}' not found")
    if not ds.get("prepared"):
        raise HTTPException(400, f"Dataset '{payload.datasetId}' is not prepared (no train/val/test).")
    cfg = payload.model_dump()
    cfg["dataset"] = ds["name"]
    cfg["datasetDir"] = ds["datasetDir"]
    try:
        return run_launcher.launch(cfg)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id.startswith("live_"):
        live = run_launcher.get_live(run_id)
        if live is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        # If finished and linked to MLflow, enrich with the full record.
        if live.get("mlflowId"):
            rec = mlflow_store.get_run(live["mlflowId"])
            if rec:
                merged = {**rec, **{k: v for k, v in live.items() if v is not None}}
                merged["id"] = run_id
                return merged
        return live
    rec = mlflow_store.get_run(run_id)
    if rec is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return rec


@router.post("/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    if not run_launcher.stop(run_id):
        raise HTTPException(400, "Run is not active or already finished")
    return {"stopped": run_id}


@router.get("/{run_id}/notebook")
def notebook(run_id: str) -> dict:
    mlflow_id, _ = _resolve(run_id)
    return mlflow_store.notebook(mlflow_id)


@router.get("/{run_id}/trajectory")
def trajectory(run_id: str) -> list[dict]:
    mlflow_id, _ = _resolve(run_id)
    return mlflow_store.trajectory(mlflow_id)


@router.get("/{run_id}/errors")
def errors(run_id: str) -> list[dict]:
    mlflow_id, _ = _resolve(run_id)
    return mlflow_store.errors(mlflow_id)


@router.get("/{run_id}/logs")
def logs(run_id: str) -> dict:
    mlflow_id, live = _resolve(run_id)
    # A still-running live run only has its process log to show.
    if live and live.get("status") == "running":
        return {"messages": [], "processLog": run_launcher.read_log(run_id)}
    msgs = mlflow_store.logs(mlflow_id)
    process_log = run_launcher.read_log(run_id) if run_id.startswith("live_") else ""
    return {"messages": msgs, "processLog": process_log}


@router.get("/{run_id}/checklist")
def checklist(run_id: str) -> dict:
    mlflow_id, _ = _resolve(run_id)
    rec = mlflow_store.get_run(mlflow_id) or {}
    target = _target_col(rec.get("dataset", "")) if rec.get("dataset") else ""
    return mlflow_store.checklist(mlflow_id, target_col=target)
