#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8100}"
MODEL_API_KEY="${MODEL_API_KEY:-donotusethisserver}"
CACHE_PATH="${CACHE_PATH:-.cache/llm-cluster/vllm-qwen3-8b-comparisons.sqlite}"

CLUSTER_COUNT="${CLUSTER_COUNT:-150}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
CANDIDATE_SEED="${CANDIDATE_SEED:-0}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
CLUSTER_SAMPLE_MULTIPLIER="${CLUSTER_SAMPLE_MULTIPLIER:-1.0}"
CLUSTER_COVER_FRACTION="${CLUSTER_COVER_FRACTION:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.6}"
COMPARISON_CONCURRENCY="${COMPARISON_CONCURRENCY:-32}"
COMPARISON_BATCH_SIZE="${COMPARISON_BATCH_SIZE:-64}"
COMPARISON_RETRIES="${COMPARISON_RETRIES:-20}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
EXACT_K="${EXACT_K:-1}"

mkdir -p results .cache/llm-cluster

if [[ "${CANDIDATE_LIMIT}" == "0" ]]; then
  limit_name="full"
else
  limit_name="limit${CANDIDATE_LIMIT}"
fi

if [[ "${EXACT_K}" == "0" ]]; then
  exact_k_name="raw"
else
  exact_k_name="exactk"
fi

output_path="${OUTPUT_PATH:-results/clinc_cluster_k${CLUSTER_COUNT}_${limit_name}_${exact_k_name}_vllm_qwen3-8b.json}"

cmd=(
  "${PYTHON_BIN}" -m llm_cluster.cli
  --task cluster
  --provider vllm
  --model-name auto
  --model-base-url "${MODEL_BASE_URL}"
  --model-api-key "${MODEL_API_KEY}"
  --cluster-count "${CLUSTER_COUNT}"
  --cluster-sample-multiplier "${CLUSTER_SAMPLE_MULTIPLIER}"
  --cluster-cover-fraction "${CLUSTER_COVER_FRACTION}"
  --cluster-seed "${CLUSTER_SEED}"
  --candidate-limit "${CANDIDATE_LIMIT}"
  --candidate-seed "${CANDIDATE_SEED}"
  --temperature "${TEMPERATURE}"
  --comparison-concurrency "${COMPARISON_CONCURRENCY}"
  --comparison-batch-size "${COMPARISON_BATCH_SIZE}"
  --comparison-retries "${COMPARISON_RETRIES}"
  --comparison-cache-path "${CACHE_PATH}"
  --progress-interval "${PROGRESS_INTERVAL}"
)

if [[ "${EXACT_K}" == "0" ]]; then
  cmd+=(--no-cluster-exact-k)
fi

echo "Running Mettu-Plaxton successive-sampling clustering; writing ${output_path}" >&2
echo "k=${CLUSTER_COUNT} candidate_limit=${CANDIDATE_LIMIT} exact_k=${EXACT_K} seed=${CLUSTER_SEED}" >&2

"${cmd[@]}" > "${output_path}"
