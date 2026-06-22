#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8100}"
MODEL_API_KEY="${MODEL_API_KEY:-donotusethisserver}"
CACHE_PATH="${CACHE_PATH:-.cache/llm-cluster/vllm-qwen3-8b-comparisons.sqlite}"

ANCHORS=(
  2765
  2432
  3779
  3347
  1648
  611
  1158
  4489
  4285
  1863
)

mkdir -p results .cache/llm-cluster

for anchor_index in "${ANCHORS[@]}"; do
  output_path="results/clinc_anchor${anchor_index}_full_vllm_qwen3-8b.json"
  echo "Running anchor ${anchor_index}; writing ${output_path}" >&2

  "${PYTHON_BIN}" -m llm_cluster.cli \
    --provider vllm \
    --model-name auto \
    --model-base-url "${MODEL_BASE_URL}" \
    --model-api-key "${MODEL_API_KEY}" \
    --anchor-index "${anchor_index}" \
    --candidate-limit 0 \
    --candidate-seed 0 \
    --temperature 0.6 \
    --sort-seed "${anchor_index}" \
    --comparison-concurrency 32 \
    --comparison-batch-size 64 \
    --comparison-retries 20 \
    --comparison-cache-path "${CACHE_PATH}" \
    > "${output_path}"
done
