# LLM Cluster

Starter infrastructure for LLM-assisted text clustering experiments.

## What Is Included

- `load_model(provider, model_name, **kwargs)` with OpenRouter and vLLM backends using the OpenAI Python client.
- A normalized `TextRow` representation and a CLINC150 loader.
- An LLM-backed `compare(a, b, c, d)` implementation for deciding whether `d(a,b) < d(c,d)`.
- Ranking rows by distance from an anchor row with randomized quicksort.
- Mettu-Plaxton-style successive sampling clustering using the same comparison oracle.
- Embedding-based CLINC clustering with INSTRUCTOR embeddings and KMeans.
- A persistent SQLite comparison cache so repeated `compare(a, b, c, d)` calls reuse the same result.
- Async parallel comparison batches for quicksort partitions with bounded API concurrency.
- An in-cluster ranking metric based on inversions between in-cluster rows (`1`) and out-cluster rows (`0`).
- Label-aware clustering metrics including purity, pairwise F1, adjusted Rand index, and normalized mutual information.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -e .
export OPENROUTER_API_KEY="..."
```

Instead of `OPENROUTER_API_KEY`, local credentials can live in `api-keys.json`:

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

For a local vLLM OpenAI-compatible server, use `provider=vllm`. The default
base URL is `http://localhost:8100/v1`; override it with `VLLM_BASE_URL` or
`--model-base-url`. Use `--model-name auto` to use the first model reported by
`/v1/models`.

## Run CLINC Ranking

The default dataset is the CLINC OOS `plus` test split with OOS rows removed,
which should produce 4,500 in-scope queries.

```bash
.venv/bin/python -m llm_cluster.cli \
  --anchor-index 0 \
  --candidate-limit 20 \
  --candidate-seed 0 \
  --comparison-concurrency 4 \
  --comparison-retries 2 \
  --comparison-cache-path .cache/llm-cluster/comparisons.sqlite \
  --model-name qwen/qwen3.5-9b
```

Local vLLM example:

```bash
.venv/bin/python -m llm_cluster.cli \
  --provider vllm \
  --model-name auto \
  --model-base-url http://localhost:8100 \
  --anchor-index 0 \
  --candidate-limit 20 \
  --candidate-seed 0 \
  --comparison-concurrency 4 \
  --comparison-batch-size 64 \
  --comparison-retries 2 \
  --comparison-cache-path .cache/llm-cluster/vllm-comparisons.sqlite
```

Candidates are shuffled before `--candidate-limit` is applied. The limit
defaults to `20` to avoid accidental large API runs. Use `--candidate-limit 0`
to rank all other rows.

Long runs write the final JSON only after sorting completes. Progress is printed
to stderr every 10 seconds by default; use `--progress-interval 0` to disable it.
For vLLM, `--comparison-batch-size` controls how many comparisons are sent in one
online batch request, while `--comparison-concurrency` controls how many batch
requests can run at once. The approximate maximum comparisons in flight is their
product, though cache hits and small partitions reduce it. OpenRouter ignores the
batch size and keeps using single-comparison requests.

SQLite cache writes are buffered in memory and flushed every 5 seconds or every
1,000 pending writes by default. Tune this with `--comparison-cache-sync-interval`
and `--comparison-cache-flush-size`, or use `--no-comparison-cache` to disable
both persistent and in-memory comparison caching.

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
  --comparison-concurrency 4 \
  --comparison-retries 2 \
  --comparison-cache-path .cache/llm-cluster/comparisons.sqlite \
  --model-name qwen/qwen3.5-9b
```

By default, the raw sampled clusters are compressed back to exactly
`--cluster-count` centers by keeping the largest sampled clusters and reassigning
all rows to those centers. Use `--no-cluster-exact-k` to return the raw
`O(k log(n/k))` sampled-center clustering instead. Use `--candidate-limit 0` to
cluster all loaded rows. The clustering JSON includes `metrics` for the final
clusters and `candidate_metrics` for the raw sampled-center clusters.

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

## Sources

- CLINC150 / CLINC OOS source dataset: https://github.com/clinc/oos-eval
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter Qwen3.5-9B model id: https://openrouter.ai/qwen/qwen3.5-9b
