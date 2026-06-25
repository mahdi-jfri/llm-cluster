#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-hkunlp/instructor-large}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

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

mkdir -p results

for anchor_index in "${ANCHORS[@]}"; do
  output_path="results/clinc_anchor${anchor_index}_full_instructor-large.json"
  echo "Running anchor ${anchor_index} with embedding comparisons; writing ${output_path}" >&2

  "${PYTHON_BIN}" -m llm_cluster.cli \
    --anchor-index "${anchor_index}" \
    --candidate-limit 0 \
    --candidate-seed 0 \
    --sort-seed "${anchor_index}" \
    --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    > "${output_path}"
done
