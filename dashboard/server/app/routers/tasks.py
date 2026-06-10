"""Tasks API: discovery, staging uploads, preparation, config, and edits."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..services import task_archive_store, task_store

router = APIRouter(prefix="/tasks", tags=["tasks"])


class BulkPayload(BaseModel):
    ids: list[str]


class TaskMetaUpdate(BaseModel):
    name: str | None = None
    target: str | None = None
    metric: str | None = None
    task_type: str | None = None
    seed: int | None = None
    desc: str | None = None


class UrlUpload(BaseModel):
    url: str
    upload_id: str | None = Field(default=None, alias="uploadId")


class ExtractRequest(BaseModel):
    path: str | None = None


class TaskConfigPayload(BaseModel):
    id: str | None = None
    name: str | None = None
    upload_id: str | None = Field(default=None, alias="uploadId")
    task: dict[str, Any] = Field(default_factory=dict)
    splits: dict[str, Any] = Field(default_factory=dict)
    agent_notes: dict[str, Any] = Field(default_factory=dict, alias="agentNotes")
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    desc: str | None = None


@router.get("")
def list_tasks(deep: bool = True) -> list[dict]:
    archived = task_archive_store.list_archived()
    return [d for d in task_store.list_tasks(deep=deep) if d.get("id") not in archived]


@router.get("/archived")
def list_archived_tasks(deep: bool = True) -> list[dict]:
    archived = task_archive_store.list_archived()
    return [d for d in task_store.list_tasks(deep=deep) if d.get("id") in archived]


@router.post("/archive")
def bulk_archive_tasks(payload: BulkPayload) -> dict:
    task_archive_store.archive(payload.ids)
    return {"archived": payload.ids}


@router.post("/unarchive")
def bulk_unarchive_tasks(payload: BulkPayload) -> dict:
    task_archive_store.unarchive(payload.ids)
    return {"unarchived": payload.ids}


@router.post("/uploads")
async def upload_task_file(
    file: UploadFile = File(...),
    upload_id: str | None = Form(None),
) -> dict:
    try:
        return task_store.upload_file(file.filename or "upload", await file.read(), upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/uploads/from-url")
def upload_task_from_url(payload: UrlUpload) -> dict:
    try:
        return task_store.upload_from_url(payload.url, payload.upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(400, f"Download failed: {exc}") from exc


@router.get("/uploads/{upload_id}/files")
def list_uploaded_files(upload_id: str) -> dict:
    try:
        return task_store.list_uploaded_files(upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/uploads/{upload_id}/preview")
def preview_uploaded_table(upload_id: str, path: str, limit: int = 50) -> dict:
    try:
        return task_store.preview_upload(upload_id, path, limit=limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/uploads/{upload_id}/extract")
def extract_uploaded_archive(upload_id: str, payload: ExtractRequest | None = None) -> dict:
    try:
        return task_store.extract_upload_archive(upload_id, payload.path if payload else None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/create-from-config")
def create_from_config(payload: TaskConfigPayload) -> dict:
    try:
        return task_store.create_from_config(payload.model_dump(by_alias=False))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{task_id}/config")
def get_config(task_id: str) -> dict:
    cfg = task_store.get_task_config(task_id)
    if cfg is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return cfg


@router.put("/{task_id}/config")
def update_config(task_id: str, payload: dict[str, Any]) -> dict:
    try:
        updated = task_store.update_task_config(task_id, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    cfg = task_store.get_task_config(task_id)
    return cfg or {}


@router.post("/{task_id}/prepare")
def prepare_task(task_id: str) -> dict:
    try:
        updated = task_store.prepare_task(task_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return updated


@router.get("/{task_id}")
def get_task(task_id: str) -> dict:
    ds = task_store.get_task(task_id)
    if ds is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return ds


@router.get("/{task_id}/preview")
def preview(task_id: str, split: str = "train", limit: int = 50) -> dict:
    try:
        return task_store.preview_rows(task_id, split=split, limit=limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{task_id}/columns")
def columns(task_id: str, split: str = "train") -> list[dict]:
    try:
        return task_store.column_stats(task_id, split=split)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/{task_id}")
def update(task_id: str, payload: TaskMetaUpdate) -> dict:
    try:
        updated = task_store.update_meta(task_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return updated


@router.post("")
async def create(
    name: str = Form(...),
    target: str | None = Form(None),
    metric: str | None = Form(None),
    task_type: str | None = Form(None),
    seed: int | None = Form(None),
    desc: str | None = Form(None),
    train: UploadFile | None = File(None),
    val: UploadFile | None = File(None),
    test: UploadFile | None = File(None),
) -> dict:
    files: dict[str, bytes] = {}
    for key, up in (("train", train), ("val", val), ("test", test)):
        if up is not None:
            files[key] = await up.read()
    meta = {
        "name": name,
        "target": target,
        "metric": metric,
        "task_type": task_type,
        "seed": seed,
        "desc": desc,
    }
    try:
        return task_store.create_task(name, files, meta)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{task_id}")
def delete(task_id: str) -> dict:
    ok = task_store.delete_task(task_id)
    if not ok:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return {"deleted": task_id}
