#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

limit=100000

.venv/bin/python -m llm_cluster.cli \
  --dataset dbpedia \
  --task embedding-cluster \
  --cluster-count 226 \
  --cluster-seed 0 \
  --candidate-limit $limit \
  --candidate-seed 0 \
  --embedding-model-name hkunlp/instructor-large \
  --embedding-prompt "Represent Wikipedia articles for ontology classification: " \
  --embedding-batch-size 512 \
  --embedding-progress \
  --embedding-kmeans-n-init 10 \
  --embedding-kmeans-max-iter 300 \
  --progress-interval 10 \
  > results/dbpedia_embedding_cluster_k14_limit${limit}_instructor-large.json
