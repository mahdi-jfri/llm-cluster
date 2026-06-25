#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-hkunlp/instructor-large}"
EMBEDDING_PROMPT="${EMBEDDING_PROMPT:-Represent utterances for intent classification: }"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
EMBEDDING_KMEANS_N_INIT="${EMBEDDING_KMEANS_N_INIT:-10}"
EMBEDDING_KMEANS_MAX_ITER="${EMBEDDING_KMEANS_MAX_ITER:-300}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
CANDIDATE_SEED="${CANDIDATE_SEED:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"

mkdir -p results

output_path="${OUTPUT_PATH:-results/clinc_embedding_cluster_k1929_full_instructor-large.json}"

"${PYTHON_BIN}" -m llm_cluster.cli \
  --task embedding-cluster \
  --cluster-count 1929 \
  --cluster-seed "${CLUSTER_SEED}" \
  --candidate-limit 0 \
  --candidate-seed "${CANDIDATE_SEED}" \
  --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
  --embedding-prompt "${EMBEDDING_PROMPT}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --embedding-kmeans-n-init "${EMBEDDING_KMEANS_N_INIT}" \
  --embedding-kmeans-max-iter "${EMBEDDING_KMEANS_MAX_ITER}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  > "${output_path}"
