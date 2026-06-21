from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Sequence

from llm_cluster.comparison import ComparisonCache, LLMDistanceComparator
from llm_cluster.data import TextRow, load_clinc
from llm_cluster.models import load_model
from llm_cluster.ranking import evaluate_in_cluster_ranking, sort_by_distance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank CLINC rows by LLM-judged distance to one anchor row."
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--config", default="plus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-oos", action="store_true")
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=20,
        help="Limit candidates to avoid accidental large API runs. Use 0 for all rows.",
    )
    parser.add_argument("--candidate-seed", type=int, default=None)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model-name", default="qwen/qwen3.5-9b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--comparison-concurrency", type=int, default=4)
    parser.add_argument("--comparison-retries", type=int, default=2)
    parser.add_argument("--comparison-cache-path", default=".cache/llm-cluster/comparisons.jsonl")
    parser.add_argument("--sort-seed", type=int, default=None)
    args = parser.parse_args(argv)

    rows = load_clinc(
        split=args.split,
        config=args.config,
        dataset_id=args.dataset_id,
        remove_oos=not args.include_oos,
    )
    if not rows:
        raise RuntimeError("Loaded zero rows.")
    if not 0 <= args.anchor_index < len(rows):
        raise IndexError(
            f"--anchor-index must be in [0, {len(rows) - 1}], got {args.anchor_index}."
        )

    anchor = rows[args.anchor_index]
    candidates = [row for row in rows if row.id != anchor.id]
    random.Random(args.candidate_seed).shuffle(candidates)
    if args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]

    model = load_model(args.provider, args.model_name)
    comparator = LLMDistanceComparator(
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.comparison_concurrency,
        parse_retries=args.comparison_retries,
        cache=ComparisonCache(args.comparison_cache_path),
    )

    ranked_rows = sort_by_distance(anchor, candidates, comparator, seed=args.sort_seed)
    metrics = evaluate_in_cluster_ranking(anchor, ranked_rows)

    output = {
        "dataset": {
            "split": args.split,
            "config": args.config,
            "n_loaded_rows": len(rows),
            "n_ranked_candidates": len(ranked_rows),
            "oos_removed": not args.include_oos,
            "candidate_seed": args.candidate_seed,
            "sort_seed": args.sort_seed,
            "comparison_concurrency": args.comparison_concurrency,
            "comparison_retries": args.comparison_retries,
        },
        "model": {
            "provider": args.provider,
            "model_name": args.model_name,
        },
        "anchor": _row_to_dict(anchor),
        "metrics": metrics.as_dict(),
        "ranked_rows": [_row_to_dict(row) for row in ranked_rows],
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _row_to_dict(row: TextRow) -> dict[str, object]:
    return {
        "id": row.id,
        "text": row.text,
        "label": row.label,
        "label_name": row.label_name,
    }


if __name__ == "__main__":
    raise SystemExit(main())
