"""Datasets API: discovery, staging uploads, preparation, config, and edits."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..services import dataset_store

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetMetaUpdate(BaseModel):
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


class DatasetConfigPayload(BaseModel):
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
def list_datasets(deep: bool = True) -> list[dict]:
    return dataset_store.list_datasets(deep=deep)


@router.post("/uploads")
async def upload_dataset_file(
    file: UploadFile = File(...),
    upload_id: str | None = Form(None),
) -> dict:
    try:
        return dataset_store.upload_file(file.filename or "upload", await file.read(), upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/uploads/from-url")
def upload_dataset_from_url(payload: UrlUpload) -> dict:
    try:
        return dataset_store.upload_from_url(payload.url, payload.upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(400, f"Download failed: {exc}") from exc


@router.get("/uploads/{upload_id}/files")
def list_uploaded_files(upload_id: str) -> dict:
    try:
        return dataset_store.list_uploaded_files(upload_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/uploads/{upload_id}/preview")
def preview_uploaded_table(upload_id: str, path: str, limit: int = 50) -> dict:
    try:
        return dataset_store.preview_upload(upload_id, path, limit=limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/uploads/{upload_id}/extract")
def extract_uploaded_archive(upload_id: str, payload: ExtractRequest | None = None) -> dict:
    try:
        return dataset_store.extract_upload_archive(upload_id, payload.path if payload else None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/create-from-config")
def create_from_config(payload: DatasetConfigPayload) -> dict:
    try:
        return dataset_store.create_from_config(payload.model_dump(by_alias=False))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{dataset_id}/config")
def get_config(dataset_id: str) -> dict:
    cfg = dataset_store.get_dataset_config(dataset_id)
    if cfg is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return cfg


@router.put("/{dataset_id}/config")
def update_config(dataset_id: str, payload: dict[str, Any]) -> dict:
    try:
        updated = dataset_store.update_dataset_config(dataset_id, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    cfg = dataset_store.get_dataset_config(dataset_id)
    return cfg or {}


@router.post("/{dataset_id}/prepare")
def prepare_dataset(dataset_id: str) -> dict:
    try:
        updated = dataset_store.prepare_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return updated


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    ds = dataset_store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, split: str = "train", limit: int = 50) -> dict:
    try:
        return dataset_store.preview_rows(dataset_id, split=split, limit=limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{dataset_id}/columns")
def columns(dataset_id: str, split: str = "train") -> list[dict]:
    try:
        return dataset_store.column_stats(dataset_id, split=split)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/{dataset_id}")
def update(dataset_id: str, payload: DatasetMetaUpdate) -> dict:
    try:
        updated = dataset_store.update_meta(dataset_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
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
        return dataset_store.create_dataset(name, files, meta)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{dataset_id}")
def delete(dataset_id: str) -> dict:
    ok = dataset_store.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return {"deleted": dataset_id}
