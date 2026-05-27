#!/bin/bash
# Start a vLLM server for a specific model.
# Usage: ./scripts/start_vllm.sh <model_id> [port]
#
# Examples:
#   ./scripts/start_vllm.sh Qwen/Qwen2.5-Coder-7B-Instruct
#   ./scripts/start_vllm.sh Qwen/Qwen2.5-Coder-32B-Instruct 8001
#   ./scripts/start_vllm.sh Qwen/Qwen2.5-72B-Instruct-AWQ 8002

MODEL=${1:?"Usage: $0 <model_id> [port]"}
PORT=${2:-8000}

echo "Starting vLLM: model=$MODEL port=$PORT"

# Detect AWQ quantization from model name
QUANTIZATION_ARGS=""
if [[ "$MODEL" == *"AWQ"* ]] || [[ "$MODEL" == *"awq"* ]]; then
    QUANTIZATION_ARGS="--quantization awq"
    echo "  → AWQ quantization enabled"
fi

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    $QUANTIZATION_ARGS \
    --served-model-name "$MODEL"
