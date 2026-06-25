# LLM Cluster

Starter infrastructure for embedding-based text clustering experiments.

## What Is Included

- `load_model(provider, model_name, **kwargs)` with OpenRouter and vLLM backends using the OpenAI Python client for direct LLM-comparison experiments.
- A normalized `TextRow` representation with CLINC150 and DBpedia Ontology loaders.
- An embedding-backed comparison oracle for deciding whether `d(a,b) < d(c,d)` from the same INSTRUCTOR vectors used by embedding clustering.
- Ranking rows by distance from an anchor row with randomized quicksort.
- Mettu-Plaxton-style successive sampling clustering using the same comparison oracle.
- Weak Comparison Alg-G Coreset+ clustering for noisy quadruplet oracles with assumed
  per-query correctness probability `p > 0.75`.
- Embedding-based CLINC clustering with INSTRUCTOR embeddings and KMeans.
- A persistent SQLite comparison cache for the optional LLM-backed comparator.
- Batched embedding comparisons for quicksort partitions and clustering oracle calls.
- An in-cluster ranking metric based on inversions between in-cluster rows (`1`) and out-cluster rows (`0`).
- Label-aware clustering metrics including purity, pairwise F1, adjusted Rand index, and normalized mutual information.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -e .
```

The CLI comparison tasks use embeddings and do not need an LLM API key. If you
use the optional LLM-backed comparator directly, credentials can live in
`api-keys.json`:

```json
{
  "openrouter": "..."
}
```

Optional OpenRouter headers:

```bash
export OPENROUTER_SITE_URL="https://your-site.example"
export OPENROUTER_APP_NAME="llm-cluster"
```

For direct LLM model experiments with a local vLLM OpenAI-compatible server, use
`provider=vllm`. The default base URL is `http://localhost:8100/v1`; override it
with `VLLM_BASE_URL` or the model loader's `base_url` argument. Use model name
`auto` to use the first model reported by `/v1/models`.

## Run CLINC Ranking

The default dataset is the CLINC OOS `plus` test split with OOS rows removed,
which should produce 4,500 in-scope queries.

```bash
.venv/bin/python -m llm_cluster.cli \
  --anchor-index 0 \
  --candidate-limit 20 \
  --candidate-seed 0 \
  --embedding-batch-size 64
```

Candidates are shuffled before `--candidate-limit` is applied. The limit
defaults to `20` for quick experiments. Use `--candidate-limit 0` to rank all
other rows.

Long runs write the final JSON only after sorting completes. Progress is printed
to stderr every 10 seconds by default; use `--progress-interval 0` to disable it.
The comparison oracle precomputes one embedding per unique selected text with
the same `--embedding-model-name`, `--embedding-prompt`, and normalization
settings used by `--task embedding-cluster`.

## Run CLINC Clustering

Use `--task cluster` to run Mettu-Plaxton-style successive sampling. Each round
samples `O(k)` rows, assigns active rows to their nearest sampled center using
the current comparison oracle, sorts rows by that assigned distance, and removes
the closest fraction.

```bash
.venv/bin/python -m llm_cluster.cli \
  --task cluster \
  --cluster-count 5 \
  --candidate-limit 50 \
  --candidate-seed 0 \
  --cluster-seed 0 \
  --embedding-batch-size 64
```

By default, the raw sampled clusters are compressed back to exactly
`--cluster-count` centers by keeping the largest sampled clusters and reassigning
all rows to those centers. Use `--no-cluster-exact-k` to return the raw
`O(k log(n/k))` sampled-center clustering instead. Use `--candidate-limit 0` to
cluster all loaded rows. The clustering JSON includes `metrics` for the final
clusters and `candidate_metrics` for the raw sampled-center clusters.

## Run Weak Comparison Alg-G Clustering

Use `--task weak-comparison-cluster` to run the general-metric Alg-G Coreset+
construction from Metric k-clustering using only Weak Comparison Oracles. This
uses guard/kernel filtering and Alg-Tester majority comparisons under the weak
comparison oracle assumption that each persistent quadruplet answer is correct
with probability `p > 0.75`. In this implementation, each quadruplet answer is
computed from INSTRUCTOR embedding distances rather than queried from an LLM.

```bash
.venv/bin/python -m llm_cluster.cli \
  --task weak-comparison-cluster \
  --cluster-count 5 \
  --candidate-limit 50 \
  --candidate-seed 0 \
  --cluster-seed 0 \
  --weak-comparison-correctness-probability 0.85 \
  --embedding-batch-size 64
```

By default, the raw Coreset+ centers are compressed back to exactly
`--cluster-count` centers by keeping the largest coreset clusters and
reassigning all rows to their nearest selected center with the same weak
comparison oracle. Use `--no-cluster-exact-k` to return the raw Coreset+
mapping instead. The JSON includes both `metrics` for the final clusters and
`coreset_metrics` for the raw Coreset+ clusters.

Alg-G references `ProbSort` and `AdvSort` as external primitives, so this code
uses local comparison-sort helpers with bounded comparison batches. The
`--weak-comparison-nearest-edge-strategy` knob controls the expensive
S1-nearest edge step: `sort` preserves the notebook behavior by sorting all
S1 x V' edges, while `pick-mins` first picks the minimum S1 edge per filtered
row and then sorts only those minima for the safe prefix. The notebook kernel
size is `max(12, int(1.2 * floor(log2(n_i))))`, and the notebook stopping
threshold is 100 active rows.

For a full CLINC run with exact-k compression and the `pick-mins` nearest-edge
strategy:

```bash
scripts/run_vllm_qwen3_8b_weak_comparison_pick_mins.sh
```

## Run CLINC Embedding Clustering

Use `--task embedding-cluster` to encode CLINC utterances with
`hkunlp/instructor-large` and cluster the vectors with KMeans using k-means++
initialization. The default INSTRUCTOR prompt is exactly:

```python
prompt = "Represent utterances for intent classification: "
```

The prompt includes the trailing colon and space.

```bash
.venv/bin/python -m llm_cluster.cli \
  --task embedding-cluster \
  --cluster-count 150 \
  --candidate-limit 0 \
  --candidate-seed 0 \
  --cluster-seed 0 \
  --embedding-batch-size 64
```

Or use the script:

```bash
scripts/run_instructor_embedding_clustering.sh
```

Override `--embedding-device cuda` or `--embedding-device cpu` when you need to
pin the model to a specific torch device. The JSON output includes the embedding
model, prompt, normalization setting, KMeans initialization/settings, metrics,
and clusters.

## Run DBpedia Ontology

Use `--dataset dbpedia` to load the DBpedia Ontology 14-class article dataset
from Hugging Face (`fancyzhx/dbpedia_14`). The DBpedia default is the full
`train` split with 560,000 rows; use `--split test` for the 70,000-row test
split. Each row text is built from the article title followed by the article
content, and the labels are the 14 ontology classes.

```bash
.venv/bin/python -m llm_cluster.cli \
  --dataset dbpedia \
  --task embedding-cluster \
  --cluster-count 14 \
  --candidate-limit 1000 \
  --candidate-seed 0 \
  --cluster-seed 0 \
  --embedding-batch-size 64
```

Use `--candidate-limit 0` to cluster all 560,000 training rows. DBpedia runs use
the default prompt:

```python
prompt = "Represent Wikipedia articles for ontology classification: "
```

## Sources

- CLINC150 / CLINC OOS source dataset: https://github.com/clinc/oos-eval
- DBpedia Ontology / DBpedia 14 Hugging Face dataset: https://huggingface.co/datasets/fancyzhx/dbpedia_14
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter Qwen3.5-9B model id: https://openrouter.ai/qwen/qwen3.5-9b
