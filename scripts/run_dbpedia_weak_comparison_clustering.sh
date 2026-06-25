#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

python_bin="${PYTHON_BIN:-.venv/bin/python}"
cpu_threads="${CPU_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpu_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpu_threads}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${cpu_threads}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${cpu_threads}}"

detected_device="$("${python_bin}" - <<'PY'
try:
    import torch
except ImportError:
    print("cpu")
else:
    print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
embedding_device="${EMBEDDING_DEVICE:-${detected_device}}"
comparison_device="${COMPARISON_DEVICE:-${embedding_device}}"

limit=100000
echo "Running DBpedia weak-comparison clustering with embedding_device=${embedding_device} comparison_device=${comparison_device}" >&2

"${python_bin}" -m llm_cluster.cli \
  --dataset dbpedia \
  --task weak-comparison-cluster \
  --cluster-count 14 \
  --cluster-seed 0 \
  --candidate-limit $limit \
  --candidate-seed 0 \
  --embedding-model-name hkunlp/instructor-large \
  --embedding-prompt "Represent Wikipedia articles for ontology classification: " \
  --embedding-batch-size 512 \
  --embedding-device "${embedding_device}" \
  --comparison-device "${comparison_device}" \
  --embedding-progress \
  --comparison-batch-size 1048576 \
  --progress-interval 10 \
  > results/dbpedia_weak_comparison_alg_g_k14_limit${limit}_exactk_sort_instructor-large.json
