#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m llm_cluster.cli \
  --task embedding-cluster \
  --cluster-count 150 \
  --cluster-seed 0 \
  --candidate-limit 0 \
  --candidate-seed 0 \
  --embedding-model-name hkunlp/instructor-large \
  --embedding-prompt "Represent utterances for intent classification: " \
  --embedding-batch-size 64 \
  --embedding-kmeans-n-init 10 \
  --embedding-kmeans-max-iter 300 \
  --progress-interval 10 \
  > results/clinc_embedding_cluster_k150_full_instructor-large.json
