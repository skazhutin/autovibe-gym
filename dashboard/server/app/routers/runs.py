"""Runs API: history (MLflow) + live launches, plus per-tab detail.

Detail tabs are served from the run's *episode directory*: the live
`data/runs/<id>/workspace` dir while a run is in progress (the gym flushes
artifacts after every step), or the finished MLflow `.../artifacts/episode` dir
otherwise. Running runs are enriched with live step/checklist/error progress so
the header, ring and chips advance during the run.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import dataset_store, mlflow_store, run_launcher

router = APIRouter(prefix="/runs", tags=["runs"])


class LaunchPayload(BaseModel):
    modelId: str | None = None
    model: str | None = None
    mode: str  # single | repeated | iterative | gym | fixed | batch
    modes: list[str] | None = None
    datasetId: str
    budgetMode: str = "local"  # local | cloud
    maxSteps: int | None = None
    maxTokens: int | None = None
    shots: int | None = None
    temp: float | None = None
    seed: int | None = None
    execution: str | None = None  # "server" | "local" | None (use default)
    enableThoughts: bool | None = None  # agent scratchpad (gym/iterative only)


def _target_col(dataset_id: str | None) -> str:
    if not dataset_id:
        return ""
    ds = dataset_store.get_dataset(dataset_id)
    if ds and ds.get("target") and ds["target"] != "—":
        return ds["target"]
    return ""


def _target_for_run(run: dict) -> str:
    ds_dir = run.get("datasetDir")
    ds_id = Path(ds_dir).name if ds_dir else run.get("dataset")
    return _target_col(ds_id)


def _episode_dir(run_id: str) -> Path | None:
    """Where this run's episode artifacts live (workspace for live, else MLflow)."""
    if run_id.startswith("live_"):
        wd = run_launcher.workspace_dir(run_id)
        if wd:
            return wd
        meta = run_launcher.get_live(run_id)
        if meta and meta.get("mlflowId"):
            return mlflow_store.mlflow_episode_dir(meta["mlflowId"])
        return None
    return mlflow_store.mlflow_episode_dir(run_id)


def _enrich_live(meta: dict) -> dict:
    """For a running live run, fold in step/checklist/error counts derived from
    the in-flight workspace artifacts."""
    if meta.get("status") != "running":
        return meta
    wd = run_launcher.workspace_dir(meta["id"])
    if not wd:
        return meta
    prog = mlflow_store.episode_progress(wd, _target_for_run(meta))
    merged = dict(meta)
    merged["step"] = prog["step"] or meta.get("step", 0)
    merged["errors"] = prog["errors"]
    merged["checklist"] = prog["checklist"]
    merged["checklistTotal"] = prog["checklistTotal"]
    merged["checklistCoverage"] = prog["checklistCoverage"]
    return merged


@router.get("")
def list_runs() -> list[dict]:
    live = [_enrich_live(m) for m in run_launcher.list_live()]
    # Hide the MLflow run each live launch produces (matched by id or run-name),
    # so a dashboard launch never shows up twice — including while it runs.
    live_mlflow_ids = {m["mlflowId"] for m in live if m.get("mlflowId")}
    live_run_names = {m.get("runName") for m in live if m.get("runName")}
    history = [
        r for r in mlflow_store.list_runs()
        if r["id"] not in live_mlflow_ids and r.get("runName") not in live_run_names
    ]
    return live + history


@router.post("")
def launch(payload: LaunchPayload) -> dict:
    ds = dataset_store.get_dataset(payload.datasetId)
    if ds is None:
        raise HTTPException(404, f"Dataset '{payload.datasetId}' not found")
    if not ds.get("prepared"):
        raise HTTPException(400, f"Dataset '{payload.datasetId}' is not prepared (no train/val/test).")
    cfg = payload.model_dump()
    selected_modes = [m for m in (cfg.get("modes") or [cfg["mode"]]) if m]
    if len(selected_modes) > 1:
        cfg["mode"] = "batch"
        cfg["modes"] = selected_modes
    elif selected_modes:
        cfg["mode"] = selected_modes[0]
        cfg["modes"] = selected_modes
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
        if live.get("status") == "running":
            return _enrich_live(live)
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
    return mlflow_store.notebook(_episode_dir(run_id))


@router.get("/{run_id}/trajectory")
def trajectory(run_id: str) -> list[dict]:
    return mlflow_store.trajectory(_episode_dir(run_id))


@router.get("/{run_id}/thoughts")
def thoughts(run_id: str) -> list[dict]:
    return mlflow_store.thoughts(_episode_dir(run_id))


@router.get("/{run_id}/errors")
def errors(run_id: str) -> list[dict]:
    return mlflow_store.errors(_episode_dir(run_id))


@router.get("/{run_id}/logs")
def logs(run_id: str) -> dict:
    episode = _episode_dir(run_id)
    msgs = mlflow_store.logs(episode)
    process_log = run_launcher.read_log(run_id) if run_id.startswith("live_") else ""
    return {"messages": msgs, "processLog": process_log}


@router.get("/{run_id}/checklist")
def checklist(run_id: str) -> dict:
    target = ""
    if run_id.startswith("live_"):
        meta = run_launcher.get_live(run_id) or {}
        target = _target_for_run(meta)
        fallback = meta.get("checklistCoverage")
    else:
        rec = mlflow_store.get_run(run_id) or {}
        target = _target_col(rec.get("dataset"))
        fallback = rec.get("checklistCoverage")
    return mlflow_store.checklist(_episode_dir(run_id), target_col=target, fallback_coverage=fallback)
