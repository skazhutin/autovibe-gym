"""Datasets API: list/detail, data preview, column stats, upload, edit, delete."""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..services import dataset_store

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetMetaUpdate(BaseModel):
    name: str | None = None
    target: str | None = None
    metric: str | None = None
    task_type: str | None = None
    seed: int | None = None
    desc: str | None = None


@router.get("")
def list_datasets(deep: bool = True) -> list[dict]:
    return dataset_store.list_datasets(deep=deep)


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    ds = dataset_store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, split: str = "train", limit: int = 50) -> dict:
    return dataset_store.preview_rows(dataset_id, split=split, limit=limit)


@router.get("/{dataset_id}/columns")
def columns(dataset_id: str, split: str = "train") -> list[dict]:
    return dataset_store.column_stats(dataset_id, split=split)


@router.put("/{dataset_id}")
def update(dataset_id: str, payload: DatasetMetaUpdate) -> dict:
    updated = dataset_store.update_meta(dataset_id, payload.model_dump())
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
    return dataset_store.create_dataset(name, files, meta)


@router.delete("/{dataset_id}")
def delete(dataset_id: str) -> dict:
    ok = dataset_store.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return {"deleted": dataset_id}
