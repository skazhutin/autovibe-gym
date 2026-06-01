#!/usr/bin/env bash
# Serve the WHOLE dashboard (API + built UI) from one process — meant to run on
# the GPU server so all compute (gym + notebook kernels + LLM) stays server-side
# and your laptop only renders the UI in a browser.
#
#   BUILD=1 dashboard/server/serve.sh        # (re)build the frontend first
#   PORT=8011 HOST=0.0.0.0 dashboard/server/serve.sh
#
# Then open it from your Mac at  http://<server-lan-ip>:<PORT>  (e.g. 10.8.45.1:8011).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${AUTOVIBE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8011}"

if [[ "${BUILD:-0}" == "1" ]]; then
  ( cd "$REPO_ROOT/dashboard/web" && npm install && npm run build )
fi

cd "$REPO_ROOT"
echo "Serving AutoVibe Gym dashboard on http://$HOST:$PORT  (UI + API, one process)"
exec "$PY" -m uvicorn dashboard.server.app.main:app --host "$HOST" --port "$PORT"
