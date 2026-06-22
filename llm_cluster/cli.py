from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Sequence

from llm_cluster.clustering import (
    SuccessiveSamplingResult,
    TextCluster,
    successive_sampling_cluster,
)
from llm_cluster.comparison import ComparisonCache, LLMDistanceComparator
from llm_cluster.data import TextRow, load_clinc
from llm_cluster.embedding_clustering import (
    CLINC_INTENT_INSTRUCTOR_PROMPT,
    DEFAULT_INSTRUCTOR_MODEL_NAME,
    EmbeddingClusteringResult,
    embedding_kmeans_cluster,
)
from llm_cluster.metrics import evaluate_clustering
from llm_cluster.models import load_model
from llm_cluster.ranking import evaluate_in_cluster_ranking, sort_by_distance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank or cluster CLINC rows with LLM comparisons or embeddings."
    )
    parser.add_argument(
        "--task",
        choices=("rank", "cluster", "embedding-cluster"),
        default="rank",
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
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help=(
            "Target k for clustering. Required with --task cluster and "
            "--task embedding-cluster."
        ),
    )
    parser.add_argument(
        "--cluster-sample-multiplier",
        type=float,
        default=1.0,
        help="Sample this many times k points per successive-sampling round.",
    )
    parser.add_argument(
        "--cluster-cover-fraction",
        type=float,
        default=0.5,
        help="Fraction of remaining points removed each successive-sampling round.",
    )
    parser.add_argument("--cluster-seed", type=int, default=None)
    parser.add_argument(
        "--no-cluster-exact-k",
        action="store_true",
        help=(
            "Return raw O(k log(n/k)) sampled clusters instead of compressing "
            "to exactly --cluster-count centers."
        ),
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_INSTRUCTOR_MODEL_NAME,
        help="Embedding model used with --task embedding-cluster.",
    )
    parser.add_argument(
        "--embedding-prompt",
        default=CLINC_INTENT_INSTRUCTOR_PROMPT,
        help="INSTRUCTOR prompt prepended to each utterance.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding model encoding.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional torch device for the embedding model, such as cuda or cpu.",
    )
    parser.add_argument(
        "--no-embedding-normalize",
        dest="embedding_normalize",
        action="store_false",
        help="Disable L2 normalization before KMeans.",
    )
    parser.set_defaults(embedding_normalize=True)
    parser.add_argument(
        "--embedding-progress",
        action="store_true",
        help="Show the embedding model's encode progress bar.",
    )
    parser.add_argument(
        "--embedding-kmeans-n-init",
        type=int,
        default=10,
        help="Number of KMeans initializations.",
    )
    parser.add_argument(
        "--embedding-kmeans-max-iter",
        type=int,
        default=300,
        help="Maximum KMeans iterations.",
    )
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model-name", default="qwen/qwen3.5-9b")
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--model-api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--comparison-concurrency", type=int, default=4)
    parser.add_argument(
        "--comparison-batch-size",
        type=int,
        default=64,
        help=(
            "Comparisons per model request when the backend supports online "
            "batching. Currently only vLLM uses this."
        ),
    )
    parser.add_argument("--comparison-retries", type=int, default=2)
    parser.add_argument(
        "--comparison-cache-path",
        default=".cache/llm-cluster/comparisons.sqlite",
    )
    parser.add_argument(
        "--comparison-cache-sync-interval",
        type=float,
        default=5.0,
        help="Seconds between SQLite cache flushes. Use 0 to flush every write.",
    )
    parser.add_argument(
        "--comparison-cache-flush-size",
        type=int,
        default=1000,
        help="Flush the SQLite cache after this many pending writes. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-comparison-cache",
        action="store_true",
        help="Disable persistent and in-memory comparison caching.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress lines on stderr. Use 0 to disable.",
    )
    parser.add_argument("--sort-seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.comparison_batch_size < 1:
        raise ValueError("--comparison-batch-size must be at least 1.")
    if args.comparison_cache_sync_interval < 0:
        raise ValueError("--comparison-cache-sync-interval must be non-negative.")
    if args.comparison_cache_flush_size < 0:
        raise ValueError("--comparison-cache-flush-size must be non-negative.")
    cluster_tasks = {"cluster", "embedding-cluster"}
    if args.task in cluster_tasks:
        if args.cluster_count is None:
            raise ValueError(
                "--cluster-count is required when --task cluster or "
                "--task embedding-cluster."
            )
        if args.cluster_count < 1:
            raise ValueError("--cluster-count must be at least 1.")
    if args.task == "cluster":
        if args.cluster_sample_multiplier <= 0:
            raise ValueError("--cluster-sample-multiplier must be positive.")
        if not 0 < args.cluster_cover_fraction < 1:
            raise ValueError("--cluster-cover-fraction must be between 0 and 1.")
    if args.task == "embedding-cluster":
        if args.embedding_batch_size < 1:
            raise ValueError("--embedding-batch-size must be at least 1.")
        if args.embedding_kmeans_n_init < 1:
            raise ValueError("--embedding-kmeans-n-init must be at least 1.")
        if args.embedding_kmeans_max_iter < 1:
            raise ValueError("--embedding-kmeans-max-iter must be at least 1.")

    rows = load_clinc(
        split=args.split,
        config=args.config,
        dataset_id=args.dataset_id,
        remove_oos=not args.include_oos,
    )
    if not rows:
        raise RuntimeError("Loaded zero rows.")

    anchor = None
    candidates: list[TextRow] = []
    cluster_rows: list[TextRow] = []
    if args.task == "rank":
        if not 0 <= args.anchor_index < len(rows):
            raise IndexError(
                f"--anchor-index must be in [0, {len(rows) - 1}], got {args.anchor_index}."
            )

        anchor = rows[args.anchor_index]
        candidates = [row for row in rows if row.id != anchor.id]
        random.Random(args.candidate_seed).shuffle(candidates)
        if args.candidate_limit > 0:
            candidates = candidates[: args.candidate_limit]
    elif args.task in cluster_tasks:
        cluster_rows = list(rows)
        random.Random(args.candidate_seed).shuffle(cluster_rows)
        if args.candidate_limit > 0:
            cluster_rows = cluster_rows[: args.candidate_limit]
        if not cluster_rows:
            raise RuntimeError("No rows selected for clustering.")

    cache: ComparisonCache | None = None
    progress: _ProgressReporter | None = None

    try:
        if args.task in {"rank", "cluster"}:
            model_kwargs = {}
            if args.model_base_url is not None:
                model_kwargs["base_url"] = args.model_base_url
            if args.model_api_key is not None:
                model_kwargs["api_key"] = args.model_api_key

            model = load_model(args.provider, args.model_name, **model_kwargs)
            comparison_batch_size = (
                args.comparison_batch_size
                if hasattr(model, "generate_batch_async")
                else 1
            )
            cache = (
                None
                if args.no_comparison_cache
                else ComparisonCache(
                    args.comparison_cache_path,
                    sync_interval_seconds=args.comparison_cache_sync_interval,
                    flush_batch_size=args.comparison_cache_flush_size,
                )
            )
            if args.progress_interval > 0:
                progress = _ProgressReporter(
                    cache=cache,
                    interval_seconds=args.progress_interval,
                )
                print(
                    "[llm-cluster] "
                    f"task={args.task} loaded_rows={len(rows):,} "
                    f"{_task_progress_details(args.task, anchor, candidates, cluster_rows)} "
                    f"provider={args.provider} "
                    f"model={getattr(model, 'model_name', args.model_name)} "
                    f"concurrency={args.comparison_concurrency} "
                    f"batch_size={comparison_batch_size} "
                    f"cache={'disabled' if cache is None else args.comparison_cache_path} "
                    f"cache_sync_interval={args.comparison_cache_sync_interval:g}s "
                    f"cache_flush_size={args.comparison_cache_flush_size}",
                    file=sys.stderr,
                    flush=True,
                )

            comparator = LLMDistanceComparator(
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_concurrency=args.comparison_concurrency,
                max_batch_size=comparison_batch_size,
                parse_retries=args.comparison_retries,
                cache=cache,
                progress_callback=progress,
            )

            if args.task == "rank":
                if anchor is None:
                    raise RuntimeError("Ranking task did not initialize an anchor.")
                ranked_rows = sort_by_distance(
                    anchor,
                    candidates,
                    comparator,
                    seed=args.sort_seed,
                )
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
                        "comparison_batch_size": comparison_batch_size,
                        "comparison_retries": args.comparison_retries,
                        "comparison_cache_enabled": cache is not None,
                        "comparison_cache_sync_interval": (
                            args.comparison_cache_sync_interval
                            if cache is not None
                            else None
                        ),
                        "comparison_cache_flush_size": (
                            args.comparison_cache_flush_size
                            if cache is not None
                            else None
                        ),
                    },
                    "model": {
                        "provider": args.provider,
                        "model_name": args.model_name,
                        "resolved_model_name": getattr(
                            model, "model_name", args.model_name
                        ),
                    },
                    "anchor": _row_to_dict(anchor),
                    "metrics": metrics.as_dict(),
                    "ranked_rows": [_row_to_dict(row) for row in ranked_rows],
                }
            else:
                if args.cluster_count is None:
                    raise RuntimeError(
                        "Clustering task did not receive --cluster-count."
                    )
                final_center_count = (
                    None if args.no_cluster_exact_k else args.cluster_count
                )
                clustering = successive_sampling_cluster(
                    cluster_rows,
                    comparator,
                    k=args.cluster_count,
                    sample_multiplier=args.cluster_sample_multiplier,
                    cover_fraction=args.cluster_cover_fraction,
                    seed=args.cluster_seed,
                    final_center_count=final_center_count,
                )
                output = _clustering_output(
                    args=args,
                    rows=rows,
                    clustered_rows=cluster_rows,
                    comparison_batch_size=comparison_batch_size,
                    cache_enabled=cache is not None,
                    clustering=clustering,
                    model_name=getattr(model, "model_name", args.model_name),
                )
        else:
            if args.cluster_count is None:
                raise RuntimeError("Clustering task did not receive --cluster-count.")
            if args.cluster_count > len(cluster_rows):
                raise ValueError(
                    "--cluster-count must be <= selected rows for "
                    "--task embedding-cluster."
                )
            if args.progress_interval > 0:
                print(
                    "[llm-cluster] "
                    f"task={args.task} loaded_rows={len(rows):,} "
                    f"clustered_rows={len(cluster_rows):,} "
                    f"embedding_model={args.embedding_model_name} "
                    f"embedding_batch_size={args.embedding_batch_size} "
                    f"k={args.cluster_count}",
                    file=sys.stderr,
                    flush=True,
                )
            clustering = embedding_kmeans_cluster(
                cluster_rows,
                k=args.cluster_count,
                model_name=args.embedding_model_name,
                prompt=args.embedding_prompt,
                batch_size=args.embedding_batch_size,
                normalize_embeddings=args.embedding_normalize,
                seed=args.cluster_seed,
                device=args.embedding_device,
                kmeans_n_init=args.embedding_kmeans_n_init,
                kmeans_max_iter=args.embedding_kmeans_max_iter,
                show_progress_bar=args.embedding_progress,
            )
            output = _embedding_clustering_output(
                args=args,
                rows=rows,
                clustered_rows=cluster_rows,
                clustering=clustering,
            )
    finally:
        if cache is not None:
            cache.flush()
    if progress is not None:
        progress.report(force=True)
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if cache is not None:
        cache.close()
    return 0


def _row_to_dict(row: TextRow) -> dict[str, object]:
    return {
        "id": row.id,
        "text": row.text,
        "label": row.label,
        "label_name": row.label_name,
    }


def _task_progress_details(
    task: str,
    anchor: TextRow | None,
    candidates: Sequence[TextRow],
    cluster_rows: Sequence[TextRow],
) -> str:
    if task == "rank":
        if anchor is None:
            return "ranked_candidates=0"
        return (
            f"ranked_candidates={len(candidates):,} "
            f"anchor_id={anchor.id} anchor_label={anchor.label_name}"
        )
    return f"clustered_rows={len(cluster_rows):,}"


def _clustering_output(
    *,
    args: argparse.Namespace,
    rows: Sequence[TextRow],
    clustered_rows: Sequence[TextRow],
    comparison_batch_size: int,
    cache_enabled: bool,
    clustering: SuccessiveSamplingResult,
    model_name: str,
) -> dict[str, object]:
    metrics = evaluate_clustering(clustering.clusters).as_dict()
    candidate_metrics = evaluate_clustering(clustering.candidate_clusters).as_dict()
    return {
        "dataset": {
            "split": args.split,
            "config": args.config,
            "n_loaded_rows": len(rows),
            "n_clustered_rows": len(clustered_rows),
            "oos_removed": not args.include_oos,
            "candidate_seed": args.candidate_seed,
            "comparison_concurrency": args.comparison_concurrency,
            "comparison_batch_size": comparison_batch_size,
            "comparison_retries": args.comparison_retries,
            "comparison_cache_enabled": cache_enabled,
            "comparison_cache_sync_interval": (
                args.comparison_cache_sync_interval if cache_enabled else None
            ),
            "comparison_cache_flush_size": (
                args.comparison_cache_flush_size if cache_enabled else None
            ),
        },
        "model": {
            "provider": args.provider,
            "model_name": args.model_name,
            "resolved_model_name": model_name,
        },
        "clustering": {
            "algorithm": "mettu_plaxton_successive_sampling",
            "target_clusters": clustering.target_clusters,
            "sample_size": clustering.sample_size,
            "sample_multiplier": clustering.sample_multiplier,
            "cover_fraction": clustering.cover_fraction,
            "seed": clustering.seed,
            "exact_k_compression": clustering.compressed,
            "final_center_count": clustering.final_center_count,
            "n_candidate_centers": len(clustering.candidate_centers),
            "n_final_centers": len(clustering.centers),
            "rounds": [round_info.as_dict() for round_info in clustering.rounds],
        },
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_clusters": [
            _cluster_to_dict(cluster) for cluster in clustering.candidate_clusters
        ],
        "clusters": [_cluster_to_dict(cluster) for cluster in clustering.clusters],
    }


def _embedding_clustering_output(
    *,
    args: argparse.Namespace,
    rows: Sequence[TextRow],
    clustered_rows: Sequence[TextRow],
    clustering: EmbeddingClusteringResult,
) -> dict[str, object]:
    metrics = evaluate_clustering(clustering.clusters).as_dict()
    return {
        "dataset": {
            "split": args.split,
            "config": args.config,
            "n_loaded_rows": len(rows),
            "n_clustered_rows": len(clustered_rows),
            "oos_removed": not args.include_oos,
            "candidate_seed": args.candidate_seed,
        },
        "model": {
            "provider": "instructor",
            "model_name": clustering.model_name,
            "prompt": clustering.prompt,
        },
        "clustering": {
            "algorithm": "instructor_kmeans",
            "target_clusters": clustering.target_clusters,
            "seed": clustering.seed,
            "batch_size": clustering.batch_size,
            "normalize_embeddings": clustering.normalize_embeddings,
            "device": clustering.device,
            "kmeans_init": clustering.kmeans_init,
            "kmeans_n_init": clustering.kmeans_n_init,
            "kmeans_max_iter": clustering.kmeans_max_iter,
            "embedding_shape": list(clustering.embedding_shape),
            "inertia": clustering.inertia,
            "n_final_centers": len(clustering.centers),
        },
        "metrics": metrics,
        "clusters": [_cluster_to_dict(cluster) for cluster in clustering.clusters],
    }


def _cluster_to_dict(cluster: TextCluster) -> dict[str, object]:
    return {
        "center": _row_to_dict(cluster.center),
        "size": len(cluster.rows),
        "rows": [_row_to_dict(row) for row in cluster.rows],
    }


class _ProgressReporter:
    def __init__(
        self,
        *,
        cache: ComparisonCache | None,
        interval_seconds: float,
    ) -> None:
        self.cache = cache
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.last_reported_at = self.started_at
        self.resolved = 0
        self.generated = 0
        self.cached = 0

    def __call__(self, event: str) -> None:
        self.resolved += 1
        if event == "cached":
            self.cached += 1
        else:
            self.generated += 1

        now = time.monotonic()
        if now - self.last_reported_at >= self.interval_seconds:
            self.report(now=now)

    def report(self, *, force: bool = False, now: float | None = None) -> None:
        if not force and self.resolved == 0:
            return

        reported_at = time.monotonic() if now is None else now
        if not force and reported_at - self.last_reported_at < self.interval_seconds:
            return

        elapsed = _format_elapsed(reported_at - self.started_at)
        if self.cache is None:
            cache_rows = "disabled"
            cache_pending = "disabled"
        else:
            cache_rows = f"{len(self.cache):,}"
            cache_pending = f"{self.cache.pending_count:,}"
        print(
            "[llm-cluster] "
            f"resolved={self.resolved:,} generated={self.generated:,} "
            f"cached={self.cached:,} cache_rows={cache_rows} "
            f"cache_pending={cache_pending} "
            f"elapsed={elapsed}",
            file=sys.stderr,
            flush=True,
        )
        self.last_reported_at = reported_at


def _format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
