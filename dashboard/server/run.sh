#!/usr/bin/env bash
# Launch the dashboard API using the project's venv (has fastapi/uvicorn/mlflow).
# Run from anywhere; paths are resolved relative to the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${AUTOVIBE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PORT="${PORT:-8000}"

cd "$REPO_ROOT"
exec "$PY" -m uvicorn dashboard.server.app.main:app --reload --port "$PORT"
