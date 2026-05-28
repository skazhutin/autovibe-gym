#!/bin/bash
# Run once on the server as booml to set up autovibe-gym.
# Usage: bash scripts/server_setup.sh
set -e

PROJECT_DIR="$HOME/autovibe-gym"
REPO="https://github.com/skazhutin/autovibe-gym.git"
VENV="$PROJECT_DIR/.venv"
MLFLOW_PORT=8002
VLLM_PORT=8003

echo "=== AutoVibe Gym — Server Setup ==="
echo "Project dir: $PROJECT_DIR"

# 1. Clone or update repo
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[1/6] Pulling latest main..."
    git -C "$PROJECT_DIR" fetch origin
    git -C "$PROJECT_DIR" checkout main
    git -C "$PROJECT_DIR" pull --ff-only origin main
else
    echo "[1/6] Cloning repo..."
    git clone "$REPO" "$PROJECT_DIR"
fi

# 2. Python venv + deps
echo "[2/6] Setting up Python virtualenv..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q
echo "      venv: $VENV"

# 3. Prepare datasets
echo "[3/6] Downloading and splitting datasets..."
cd "$PROJECT_DIR"
"$VENV/bin/python" scripts/prepare_datasets.py

# 4. Create .env if missing
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[4/6] Creating .env from template..."
    cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
    # Patch defaults for local server
    sed -i "s|http://<server-ip>:8000/v1|http://localhost:$VLLM_PORT/v1|g" "$ENV_FILE"
    sed -i "s|http://<server-ip>:5000|http://localhost:$MLFLOW_PORT|g" "$ENV_FILE"
    echo "      .env created — edit LLM_MODEL if needed"
else
    echo "[4/6] .env already exists, skipping"
fi

# 5. Start MLflow server (detached)
echo "[5/6] Starting MLflow on port $MLFLOW_PORT..."
"$VENV/bin/mlflow" server \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$PROJECT_DIR/mlruns" \
    >> "$PROJECT_DIR/mlflow.log" 2>&1 &
echo "      MLflow PID: $! — UI: http://10.8.45.1:$MLFLOW_PORT"
echo "      (or http://10.8.52.11:$MLFLOW_PORT from school)"

# 6. Check GPU
echo "[6/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    || echo "      nvidia-smi not found or no GPU visible"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $ENV_FILE — set LLM_MODEL"
echo "  2. Start vLLM:"
echo "     bash $PROJECT_DIR/scripts/start_vllm.sh Qwen/Qwen2.5-Coder-7B-Instruct $VLLM_PORT"
echo "  3. Run first experiment:"
echo "     cd $PROJECT_DIR && .venv/bin/python -m experiments.run_gym \\"
echo "       --dataset-dir datasets/wine_quality --mode local"
echo "  4. Compare runs:"
echo "     .venv/bin/python -m experiments.compare"
echo ""
echo "MLflow UI: http://10.8.45.1:$MLFLOW_PORT"
