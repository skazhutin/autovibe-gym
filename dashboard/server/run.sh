#!/usr/bin/env bash
# Launch the dashboard API using the project's venv (has fastapi/uvicorn/mlflow).
# Run from anywhere; paths are resolved relative to the repo root.
#
# Auto-reload is OFF by default: the gym writes notebook .py artifacts into
# dashboard/server/data/runs/<id>/ during a run, which would otherwise make
# uvicorn --reload restart the server mid-run (orphaning the run process).
# Set RELOAD=1 for code-editing sessions; the data dir is excluded from the watch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${AUTOVIBE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PORT="${PORT:-8000}"

cd "$REPO_ROOT"
ARGS=(-m uvicorn dashboard.server.app.main:app --port "$PORT")
if [[ "${RELOAD:-0}" == "1" ]]; then
  ARGS+=(--reload --reload-dir dashboard/server/app)
fi
exec "$PY" "${ARGS[@]}"
