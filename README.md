# LLM Cluster

Starter infrastructure for LLM-assisted text clustering experiments.

## What Is Included

- `load_model(provider, model_name, **kwargs)` with an OpenRouter backend using the OpenAI Python client.
- A normalized `TextRow` representation and a CLINC150 loader.
- An LLM-backed `compare(a, b, c, d)` implementation for deciding whether `d(a,b) < d(c,d)`.
- Ranking rows by distance from an anchor row with randomized quicksort.
- A persistent comparison cache so repeated `compare(a, b, c, d)` calls reuse the same result.
- Async parallel comparison batches for quicksort partitions with bounded API concurrency.
- An in-cluster ranking metric based on inversions between in-cluster rows (`1`) and out-cluster rows (`0`).

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
  --comparison-cache-path .cache/llm-cluster/comparisons.jsonl \
  --model-name qwen/qwen3.5-9b
```

Candidates are shuffled before `--candidate-limit` is applied. The limit
defaults to `20` to avoid accidental large API runs. Use `--candidate-limit 0`
to rank all other rows.

## Sources

- CLINC150 / CLINC OOS source dataset: https://github.com/clinc/oos-eval
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter Qwen3.5-9B model id: https://openrouter.ai/qwen/qwen3.5-9b
