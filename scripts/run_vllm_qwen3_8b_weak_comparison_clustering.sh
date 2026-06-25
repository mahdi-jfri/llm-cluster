#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-hkunlp/instructor-large}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
COMPARISON_BATCH_SIZE="${COMPARISON_BATCH_SIZE:-8192}"

CLUSTER_COUNT="${CLUSTER_COUNT:-150}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
CANDIDATE_SEED="${CANDIDATE_SEED:-0}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
EXACT_K="${EXACT_K:-1}"

mkdir -p results
cpu_threads="${CPU_THREADS:-$(nproc)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cpu_threads}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$cpu_threads}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cpu_threads}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$cpu_threads}"

detected_device="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
except ImportError:
    print("cpu")
else:
    print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-${detected_device}}"
COMPARISON_DEVICE="${COMPARISON_DEVICE:-${EMBEDDING_DEVICE}}"

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

output_path="${OUTPUT_PATH:-results/clinc_weak_comparison_alg_g_k${CLUSTER_COUNT}_${limit_name}_${exact_k_name}_sort_instructor-large.json}"

cmd=(
  "${PYTHON_BIN}" -m llm_cluster.cli
  --task weak-comparison-cluster
  --cluster-count "${CLUSTER_COUNT}"
  --cluster-seed "${CLUSTER_SEED}"
  --candidate-limit "${CANDIDATE_LIMIT}"
  --candidate-seed "${CANDIDATE_SEED}"
  --embedding-model-name "${EMBEDDING_MODEL_NAME}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --embedding-device "${EMBEDDING_DEVICE}"
  --comparison-device "${COMPARISON_DEVICE}"
  --comparison-batch-size "${COMPARISON_BATCH_SIZE}"
  --progress-interval "${PROGRESS_INTERVAL}"
)

if [[ "${EXACT_K}" == "0" ]]; then
  cmd+=(--no-cluster-exact-k)
fi

echo "Running weak-comparison Alg-G clustering with embedding comparisons; writing ${output_path}" >&2
echo "k=${CLUSTER_COUNT} candidate_limit=${CANDIDATE_LIMIT} exact_k=${EXACT_K} seed=${CLUSTER_SEED} cpu_threads=${cpu_threads} embedding_device=${EMBEDDING_DEVICE} comparison_device=${COMPARISON_DEVICE}" >&2

"${cmd[@]}" > "${output_path}"
