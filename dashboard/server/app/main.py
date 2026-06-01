"""FastAPI entrypoint for the AutoVibe Gym dashboard.

Run (from repo root, using the project venv that has fastapi/uvicorn/mlflow):

    .venv/bin/python -m uvicorn dashboard.server.app.main:app --reload --port 8000

or use the helper:  dashboard/server/run.sh
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import datasets, health, models, runs, settings as settings_router

settings = get_settings()

app = FastAPI(
    title="AutoVibe Gym Dashboard API",
    version="0.1.0",
    description="Local control panel for running LLM agents on the AutoVibe Gym.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(runs.router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"service": "autovibe-gym-dashboard", "docs": "/docs", "api": "/api"}
