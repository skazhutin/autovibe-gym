"""Datasets API: config CRUD, uploads/downloads, preview, validate, prepare."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..services import dataset_store

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetCreatePayload(BaseModel):
    name: str


class FileEntryModel(BaseModel):
    logical_name: str
    role: str
    source_type: str = "upload"
    url: str = ""
    path: str = ""
    format: str = "auto"
    read_options: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    archive_member: str = ""


class JoinModel(BaseModel):
    left_table: str
    right_table: str
    how: str = "left"
    left_on: list[str] = Field(default_factory=list)
    right_on: list[str] = Field(default_factory=list)


class DatasetConfigPayload(BaseModel):
    name: str
    suite: str = "custom"
    source: dict[str, Any] = Field(default_factory=dict)
    dataset_notes: dict[str, Any] = Field(default_factory=dict)
    ingestion: dict[str, Any]
    relations: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any]
    split: dict[str, Any]
    preparation: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class DownloadRequest(BaseModel):
    url: str
    suggested_name: str | None = None


class PreviewRequest(BaseModel):
    logical_name: str | None = None
    split_role: str | None = None
    joined: bool = False
    limit: int = 20


class LegacyDatasetMetaUpdate(BaseModel):
    name: str | None = None
    target: str | None = None
    metric: str | None = None
    task_type: str | None = None
    seed: int | None = None
    desc: str | None = None


@router.get("")
def list_datasets(deep: bool = True) -> list[dict[str, Any]]:
    return dataset_store.list_datasets(deep=deep)


@router.post("/create")
def create_dataset(payload: DatasetCreatePayload) -> dict[str, Any]:
    try:
        return dataset_store.create_dataset(payload.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    ds = dataset_store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds


@router.get("/{dataset_id}/config")
def get_dataset_config(dataset_id: str) -> dict[str, Any]:
    ds = dataset_store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds.get("config") or {}


@router.put("/{dataset_id}/config")
def save_dataset_config(dataset_id: str, payload: DatasetConfigPayload) -> dict[str, Any]:
    try:
        return dataset_store.save_config(dataset_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{dataset_id}/files")
async def upload_dataset_files(
    dataset_id: str,
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    if not files:
        raise HTTPException(400, "At least one file is required.")
    items: list[tuple[str, bytes]] = []
    for upload in files:
        items.append((upload.filename or "upload.bin", await upload.read()))
    try:
        return dataset_store.upload_files(dataset_id, items)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{dataset_id}/downloads")
def download_dataset_files(dataset_id: str, payload: list[DownloadRequest]) -> list[dict[str, Any]]:
    try:
        return dataset_store.download_files(dataset_id, [item.model_dump() for item in payload])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, payload: PreviewRequest) -> dict[str, Any]:
    try:
        return dataset_store.preview(
            dataset_id,
            logical_name=payload.logical_name,
            split_role=payload.split_role,
            joined=payload.joined,
            limit=payload.limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{dataset_id}/validate")
def validate_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return dataset_store.validate(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{dataset_id}/prepare")
def prepare_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return dataset_store.prepare(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{dataset_id}/preview")
def preview_prepared_split(dataset_id: str, split: str = "train", limit: int = 50) -> dict[str, Any]:
    return dataset_store.preview_rows(dataset_id, split=split, limit=limit)


@router.get("/{dataset_id}/columns")
def prepared_columns(dataset_id: str, split: str = "train") -> list[dict[str, Any]]:
    return dataset_store.column_stats(dataset_id, split=split)


@router.put("/{dataset_id}")
def update_legacy_dataset(dataset_id: str, payload: LegacyDatasetMetaUpdate) -> dict[str, Any]:
    existing = dataset_store.get_dataset(dataset_id)
    if existing is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    config = dict(existing.get("config") or {})
    if not config:
        raise HTTPException(400, "Dataset has no editable config.")
    if payload.name:
        config["name"] = payload.name
    task = dict(config.get("task") or {})
    split = dict(config.get("split") or {})
    dataset_notes = dict(config.get("dataset_notes") or {})
    if payload.target is not None:
        task["target_col"] = payload.target
    if payload.metric is not None:
        task["metric"] = payload.metric
    if payload.task_type is not None:
        task["type"] = payload.task_type
    if payload.seed is not None:
        split["seed"] = payload.seed
    if payload.desc is not None:
        dataset_notes["short_description"] = payload.desc
    config["task"] = task
    config["split"] = split
    config["dataset_notes"] = dataset_notes
    try:
        return dataset_store.save_config(dataset_id, config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("")
async def create_legacy_prepared_dataset(
    name: str = Form(...),
    target: str | None = Form(None),
    metric: str | None = Form(None),
    task_type: str | None = Form(None),
    seed: int | None = Form(None),
    desc: str | None = Form(None),
    train: UploadFile | None = File(None),
    val: UploadFile | None = File(None),
    test: UploadFile | None = File(None),
) -> dict[str, Any]:
    dataset = dataset_store.create_dataset(name)
    uploads = [(role, upload) for role, upload in (("train", train), ("val", val), ("test", test)) if upload is not None]
    if uploads:
        uploaded: list[tuple[str, bytes]] = []
        for _, upload in uploads:
            uploaded.append((upload.filename or "upload.csv", await upload.read()))
        saved = dataset_store.upload_files(dataset["id"], uploaded)
        config = dataset["config"]
        config["ingestion"] = {
            "mode": "pre_split",
            "files": [],
        }
        for (role, _), saved_item in zip(uploads, saved):
            config["ingestion"]["files"].append(
                {
                    "logical_name": role,
                    "role": role,
                    "source_type": "upload",
                    "url": "",
                    "path": saved_item["path"],
                    "format": "auto",
                    "read_options": {},
                    "optional": role == "val",
                    "archive_member": "",
                }
            )
        config["task"]["target_col"] = target or ""
        config["task"]["metric"] = metric or ""
        config["task"]["type"] = task_type or config["task"].get("type", "classification")
        if seed is not None:
            config["split"]["seed"] = seed
        if desc:
            config["dataset_notes"]["short_description"] = desc
        dataset = dataset_store.save_config(dataset["id"], config)
    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    ok = dataset_store.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return {"deleted": dataset_id}
