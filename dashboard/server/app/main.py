"""FastAPI entrypoint for the AutoVibe Gym dashboard.

Run (from repo root, using the project venv that has fastapi/uvicorn/mlflow):

    .venv/bin/python -m uvicorn dashboard.server.app.main:app --reload --port 8000

or use the helper:  dashboard/server/run.sh
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from .config import get_settings
from .routers import datasets, health, models, runs, settings as settings_router

# Keep dashboard uploads useful for real tabular files while preserving a cap.
MultiPartParser.max_part_size = 250 * 1024 * 1024

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


# --- Single-app mode: serve the built SPA when present ---------------------
# Lets the whole dashboard run from ONE process on the server (all compute
# server-side; the Mac just renders). In dev the SPA is served by Vite instead.
_web = settings.web_dist
if (_web / "index.html").exists():
    if (_web / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=_web / "assets"), name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(_web / "index.html")

    @app.get("/{path:path}")
    def _spa(path: str) -> FileResponse:
        # Serve real static files; fall back to index.html for client routes.
        candidate = _web / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_web / "index.html")
else:

    @app.get("/")
    def root() -> dict:
        return {"service": "autovibe-gym-dashboard", "docs": "/docs", "api": "/api"}
