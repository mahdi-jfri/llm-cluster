from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Any, Sequence

from llm_cluster.clustering import (
    SuccessiveSamplingResult,
    TextCluster,
    successive_sampling_cluster,
)
from llm_cluster.data import TextRow, load_dataset_rows
from llm_cluster.embedding_clustering import (
    CLINC_INTENT_INSTRUCTOR_PROMPT,
    DBPEDIA_ONTOLOGY_INSTRUCTOR_PROMPT,
    DEFAULT_INSTRUCTOR_MODEL_NAME,
    EmbeddingDistanceComparator,
    EmbeddingClusteringResult,
    embedding_kmeans_cluster,
)
from llm_cluster.metrics import evaluate_clustering
from llm_cluster.weak_comparison_clustering import (
    DEFAULT_NEAREST_EDGE_STRATEGY,
    WeakComparisonAlgGResult,
    weak_comparison_alg_g_cluster,
)
from llm_cluster.ranking import evaluate_in_cluster_ranking, sort_by_distance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank or cluster text classification rows with embeddings."
    )
    parser.add_argument(
        "--task",
        choices=("rank", "cluster", "weak-comparison-cluster", "embedding-cluster"),
        default="rank",
    )
    parser.add_argument(
        "--dataset",
        default="clinc",
        help="Named dataset to load: clinc or dbpedia.",
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face dataset config. Uses the dataset default when unset.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional dataset split. Defaults to CLINC test or DBpedia train.",
    )
    parser.add_argument(
        "--include-oos",
        action="store_true",
        help="Include CLINC out-of-scope rows. Only applies to CLINC datasets.",
    )
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=20,
        help="Limit candidates for quick runs. Use 0 for all rows.",
    )
    parser.add_argument("--candidate-seed", type=int, default=None)
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help=(
            "Target k for clustering. Required with --task cluster, "
            "--task weak-comparison-cluster, and --task embedding-cluster."
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
        "--weak-comparison-correctness-probability",
        type=float,
        default=0.85,
        help=(
            "Recorded weak-comparison oracle correctness probability. "
            "The notebook implementation uses a fixed kernel/guard size."
        ),
    )
    parser.add_argument(
        "--weak-comparison-sample1-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for the notebook Alg-G S1 sample size max(6, 3k).",
    )
    parser.add_argument(
        "--weak-comparison-sample2-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for the notebook Alg-G S2 sample size max(24, 8k, 80).",
    )
    parser.add_argument(
        "--weak-comparison-window-multiplier",
        type=float,
        default=2.0,
        help=(
            "Deprecated compatibility field; the notebook implementation "
            "uses max(12, int(1.2 * floor(log2(n_i)))) unless an explicit "
            "debug window size is supplied."
        ),
    )
    parser.add_argument(
        "--weak-comparison-terminal-multiplier",
        type=float,
        default=0.01,
        help=(
            "Deprecated compatibility field; the notebook implementation "
            "stops when active rows are <= 100."
        ),
    )
    parser.add_argument(
        "--weak-comparison-sample-fraction-cap",
        type=float,
        default=0,
        help=(
            "Deprecated compatibility field. Must be 0 because the notebook "
            "implementation does not cap S1/S2 by active-row fraction."
        ),
    )
    parser.add_argument(
        "--weak-comparison-nearest-edge-strategy",
        choices=("sort", "pick-mins"),
        default=DEFAULT_NEAREST_EDGE_STRATEGY,
        help=(
            "How Alg-G chooses S1-nearest edges for filtered rows. "
            "'sort' preserves the notebook behavior by sorting all S1 x V' "
            "edges; 'pick-mins' first picks the minimum S1 edge per row and "
            "then sorts only those minima for the safe prefix."
        ),
    )
    parser.add_argument(
        "--weak-comparison-window-size",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility field. Leave unset because the notebook "
            "kernel/guard size is fixed by the round size."
        ),
    )
    parser.add_argument(
        "--weak-comparison-max-rounds",
        type=int,
        default=None,
        help="Optional maximum number of Alg-G rounds.",
    )
    parser.add_argument(
        "--no-cluster-exact-k",
        action="store_true",
        help=(
            "Return raw sampled/coreset clusters instead of compressing to "
            "exactly --cluster-count centers."
        ),
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_INSTRUCTOR_MODEL_NAME,
        help="Embedding model used for comparisons and --task embedding-cluster.",
    )
    parser.add_argument(
        "--embedding-prompt",
        default=None,
        help="INSTRUCTOR prompt prepended to each row text.",
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
        help="Disable L2 normalization before comparisons or KMeans.",
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
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--model-name",
        default="qwen/qwen3.5-9b",
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--model-api-key",
        default=None,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--comparison-concurrency",
        type=int,
        default=4,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--comparison-batch-size",
        type=int,
        default=8192,
        help=(
            "Number of weak-comparison embedding distance queries per "
            "vectorized NumPy batch."
        ),
    )
    parser.add_argument(
        "--comparison-retries",
        type=int,
        default=2,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--comparison-cache-path",
        default=".cache/llm-cluster/comparisons.sqlite",
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--comparison-cache-sync-interval",
        type=float,
        default=5.0,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--comparison-cache-flush-size",
        type=int,
        default=1000,
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--no-comparison-cache",
        action="store_true",
        help="Legacy LLM comparison option; ignored by embedding comparisons.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress lines on stderr. Use 0 to disable.",
    )
    parser.add_argument("--sort-seed", type=int, default=None)
    args = parser.parse_args(argv)
    cluster_tasks = {"cluster", "weak-comparison-cluster", "embedding-cluster"}
    if args.task in cluster_tasks:
        if args.cluster_count is None:
            raise ValueError(
                "--cluster-count is required when --task cluster, "
                "--task weak-comparison-cluster, or --task embedding-cluster."
            )
        if args.cluster_count < 1:
            raise ValueError("--cluster-count must be at least 1.")
    if args.task == "cluster":
        if args.cluster_sample_multiplier <= 0:
            raise ValueError("--cluster-sample-multiplier must be positive.")
        if not 0 < args.cluster_cover_fraction < 1:
            raise ValueError("--cluster-cover-fraction must be between 0 and 1.")
    if args.task == "weak-comparison-cluster":
        if not 0.0 <= args.weak_comparison_correctness_probability <= 1.0:
            raise ValueError(
                "--weak-comparison-correctness-probability must be in [0, 1]."
            )
        if args.weak_comparison_sample1_multiplier <= 0:
            raise ValueError(
                "--weak-comparison-sample1-multiplier must be positive."
            )
        if args.weak_comparison_sample2_multiplier <= 0:
            raise ValueError(
                "--weak-comparison-sample2-multiplier must be positive."
            )
        if args.weak_comparison_window_multiplier <= 0:
            raise ValueError("--weak-comparison-window-multiplier must be positive.")
        if args.weak_comparison_terminal_multiplier < 0:
            raise ValueError(
                "--weak-comparison-terminal-multiplier must be non-negative."
            )
        if args.weak_comparison_sample_fraction_cap != 0:
            raise ValueError(
                "--weak-comparison-sample-fraction-cap must be 0 for the "
                "notebook implementation."
            )
        if args.weak_comparison_window_size is not None:
            raise ValueError(
                "--weak-comparison-window-size is not supported by the "
                "notebook implementation."
            )
        if (
            args.weak_comparison_max_rounds is not None
            and args.weak_comparison_max_rounds < 1
        ):
            raise ValueError("--weak-comparison-max-rounds must be at least 1.")
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be at least 1.")
    if args.comparison_batch_size < 1:
        raise ValueError("--comparison-batch-size must be at least 1.")
    if args.task == "embedding-cluster":
        if args.embedding_kmeans_n_init < 1:
            raise ValueError("--embedding-kmeans-n-init must be at least 1.")
        if args.embedding_kmeans_max_iter < 1:
            raise ValueError("--embedding-kmeans-max-iter must be at least 1.")

    if args.embedding_prompt is None:
        args.embedding_prompt = _default_embedding_prompt(args.dataset)

    rows = _load_rows_from_args(args)
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
        if (
            args.task == "weak-comparison-cluster"
            and not args.no_cluster_exact_k
            and args.cluster_count is not None
            and args.cluster_count > len(cluster_rows)
        ):
            raise ValueError(
                "--cluster-count must be <= selected rows when weak comparison "
                "exact-k compression is enabled."
            )

    progress: _ProgressReporter | None = None

    if args.task in {"rank", "cluster", "weak-comparison-cluster"}:
        comparison_rows = _comparison_rows_for_task(
            args.task,
            anchor=anchor,
            candidates=candidates,
            cluster_rows=cluster_rows,
        )
        if args.progress_interval > 0:
            progress = _ProgressReporter(interval_seconds=args.progress_interval)
            batch_detail = (
                f" comparison_batch_size={args.comparison_batch_size}"
                if args.task == "weak-comparison-cluster"
                else ""
            )
            print(
                "[llm-cluster] "
                f"task={args.task} dataset={args.dataset} "
                f"loaded_rows={len(rows):,} "
                f"{_task_progress_details(args.task, anchor, candidates, cluster_rows)} "
                f"comparison_backend=embedding "
                f"embedding_model={args.embedding_model_name} "
                f"embedding_batch_size={args.embedding_batch_size}"
                f"{batch_detail}",
                file=sys.stderr,
                flush=True,
            )

        comparator = EmbeddingDistanceComparator(
            comparison_rows,
            model_name=args.embedding_model_name,
            prompt=args.embedding_prompt,
            batch_size=args.embedding_batch_size,
            normalize_embeddings=args.embedding_normalize,
            device=args.embedding_device,
            show_progress_bar=args.embedding_progress,
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
                "dataset": _dataset_output(
                    args,
                    rows,
                    n_ranked_candidates=len(ranked_rows),
                    candidate_seed=args.candidate_seed,
                    sort_seed=args.sort_seed,
                    n_embedding_source_rows=comparator.n_source_rows,
                    n_unique_embedding_texts=comparator.n_unique_texts,
                ),
                "model": _embedding_model_output(comparator),
                "comparison": _embedding_comparison_output(comparator),
                "anchor": _row_to_dict(anchor),
                "metrics": metrics.as_dict(),
                "ranked_rows": [_row_to_dict(row) for row in ranked_rows],
            }
        elif args.task == "cluster":
            if args.cluster_count is None:
                raise RuntimeError("Clustering task did not receive --cluster-count.")
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
                comparator=comparator,
                clustering=clustering,
            )
        else:
            if args.cluster_count is None:
                raise RuntimeError("Clustering task did not receive --cluster-count.")
            final_center_count = (
                None if args.no_cluster_exact_k else args.cluster_count
            )
            clustering = weak_comparison_alg_g_cluster(
                cluster_rows,
                comparator,
                k=args.cluster_count,
                correctness_probability=args.weak_comparison_correctness_probability,
                sample1_multiplier=args.weak_comparison_sample1_multiplier,
                sample2_multiplier=args.weak_comparison_sample2_multiplier,
                window_multiplier=args.weak_comparison_window_multiplier,
                terminal_multiplier=args.weak_comparison_terminal_multiplier,
                sample_fraction_cap=(
                    None
                    if args.weak_comparison_sample_fraction_cap == 0
                    else args.weak_comparison_sample_fraction_cap
                ),
                nearest_edge_strategy=args.weak_comparison_nearest_edge_strategy,
                seed=args.cluster_seed,
                final_center_count=final_center_count,
                window_size=args.weak_comparison_window_size,
                max_rounds=args.weak_comparison_max_rounds,
                comparison_batch_size=args.comparison_batch_size,
            )
            output = _weak_comparison_clustering_output(
                args=args,
                rows=rows,
                clustered_rows=cluster_rows,
                comparator=comparator,
                clustering=clustering,
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
                f"task={args.task} dataset={args.dataset} "
                f"loaded_rows={len(rows):,} "
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
    if progress is not None:
        progress.report(force=True)
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _load_rows_from_args(args: argparse.Namespace) -> list[TextRow]:
    kwargs = {
        "split": args.split,
        "config": args.config,
        "dataset_id": args.dataset_id,
    }
    if _is_clinc_dataset(args.dataset):
        return load_dataset_rows(
            args.dataset,
            **kwargs,
            remove_oos=not args.include_oos,
        )
    if args.include_oos:
        raise ValueError("--include-oos only applies to CLINC datasets.")
    return load_dataset_rows(args.dataset, **kwargs)


def _default_embedding_prompt(dataset: str) -> str:
    if _is_dbpedia_dataset(dataset):
        return DBPEDIA_ONTOLOGY_INSTRUCTOR_PROMPT
    return CLINC_INTENT_INSTRUCTOR_PROMPT


def _dataset_output(
    args: argparse.Namespace,
    rows: Sequence[TextRow],
    **extra: Any,
) -> dict[str, object]:
    metadata = dict(rows[0].metadata) if rows else {}
    output: dict[str, object] = {
        "name": args.dataset,
        "dataset_id": metadata.get("dataset_id", args.dataset_id),
        "split": metadata.get("split", args.split),
        "config": metadata.get("config", args.config),
        "n_loaded_rows": len(rows),
    }
    if _is_clinc_dataset(args.dataset):
        output["oos_removed"] = not args.include_oos
    output.update(extra)
    return output


def _is_clinc_dataset(dataset: str) -> bool:
    return _normalize_dataset_name(dataset) in {
        "clinc",
        "clinc150",
        "clinc_150",
        "clinc_oos",
    }


def _is_dbpedia_dataset(dataset: str) -> bool:
    return _normalize_dataset_name(dataset) in {
        "dbpedia",
        "dbpedia_14",
        "dbpedia_ontology",
    }


def _normalize_dataset_name(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def _row_to_dict(row: TextRow) -> dict[str, object]:
    return {
        "id": row.id,
        "text": row.text,
        "label": row.label,
        "label_name": row.label_name,
    }


def _comparison_rows_for_task(
    task: str,
    *,
    anchor: TextRow | None,
    candidates: Sequence[TextRow],
    cluster_rows: Sequence[TextRow],
) -> list[TextRow]:
    if task == "rank":
        if anchor is None:
            raise RuntimeError("Ranking task did not initialize an anchor.")
        return [anchor, *candidates]
    return list(cluster_rows)


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


def _embedding_model_output(
    comparator: EmbeddingDistanceComparator,
) -> dict[str, object]:
    return {
        "provider": "instructor",
        "model_name": comparator.model_name,
        "prompt": comparator.prompt,
    }


def _embedding_comparison_output(
    comparator: EmbeddingDistanceComparator,
) -> dict[str, object]:
    return {
        "backend": "embedding",
        "distance": "squared_l2",
        "batch_size": comparator.batch_size,
        "normalize_embeddings": comparator.normalize_embeddings,
        "device": comparator.device,
        "embedding_shape": list(comparator.embedding_shape),
    }


def _clustering_output(
    *,
    args: argparse.Namespace,
    rows: Sequence[TextRow],
    clustered_rows: Sequence[TextRow],
    comparator: EmbeddingDistanceComparator,
    clustering: SuccessiveSamplingResult,
) -> dict[str, object]:
    metrics = evaluate_clustering(clustering.clusters).as_dict()
    candidate_metrics = evaluate_clustering(clustering.candidate_clusters).as_dict()
    return {
        "dataset": _dataset_output(
            args,
            rows,
            n_clustered_rows=len(clustered_rows),
            candidate_seed=args.candidate_seed,
            n_embedding_source_rows=comparator.n_source_rows,
            n_unique_embedding_texts=comparator.n_unique_texts,
        ),
        "model": _embedding_model_output(comparator),
        "comparison": _embedding_comparison_output(comparator),
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


def _weak_comparison_clustering_output(
    *,
    args: argparse.Namespace,
    rows: Sequence[TextRow],
    clustered_rows: Sequence[TextRow],
    comparator: EmbeddingDistanceComparator,
    clustering: WeakComparisonAlgGResult,
) -> dict[str, object]:
    metrics = evaluate_clustering(clustering.clusters).as_dict()
    coreset_metrics = evaluate_clustering(clustering.coreset_clusters).as_dict()
    return {
        "dataset": _dataset_output(
            args,
            rows,
            n_clustered_rows=len(clustered_rows),
            candidate_seed=args.candidate_seed,
            n_embedding_source_rows=comparator.n_source_rows,
            n_unique_embedding_texts=comparator.n_unique_texts,
        ),
        "model": _embedding_model_output(comparator),
        "comparison": _embedding_comparison_output(comparator),
        "clustering": {
            "algorithm": "weak_comparison_alg_g_coreset_plus",
            "target_clusters": clustering.target_clusters,
            "oracle_correctness_probability": clustering.correctness_probability,
            "sample1_multiplier": clustering.sample1_multiplier,
            "sample2_multiplier": clustering.sample2_multiplier,
            "window_multiplier": clustering.window_multiplier,
            "terminal_multiplier": clustering.terminal_multiplier,
            "sample_fraction_cap": clustering.sample_fraction_cap,
            "nearest_edge_strategy": clustering.nearest_edge_strategy,
            "seed": clustering.seed,
            "exact_k_compression": clustering.compressed,
            "final_center_count": clustering.final_center_count,
            "comparison_batch_size": args.comparison_batch_size,
            "n_coreset_centers": len(clustering.coreset_centers),
            "n_final_centers": len(clustering.centers),
            "rounds": [round_info.as_dict() for round_info in clustering.rounds],
        },
        "metrics": metrics,
        "coreset_metrics": coreset_metrics,
        "coreset_clusters": [
            _cluster_to_dict(cluster) for cluster in clustering.coreset_clusters
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
        "dataset": _dataset_output(
            args,
            rows,
            n_clustered_rows=len(clustered_rows),
            candidate_seed=args.candidate_seed,
        ),
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
        interval_seconds: float,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.last_reported_at = self.started_at
        self.resolved = 0
        self.computed = 0
        self.generated = 0
        self.cached = 0

    def __call__(self, event: str) -> None:
        self.add(event, 1)

    def add(self, event: str, count: int) -> None:
        if count <= 0:
            return

        now = time.monotonic()
        self.resolved += count
        if event == "cached":
            self.cached += count
        elif event == "generated":
            self.generated += count
        else:
            self.computed += count

        if now - self.last_reported_at >= self.interval_seconds:
            self.report(now=now)

    def report(self, *, force: bool = False, now: float | None = None) -> None:
        if not force and self.resolved == 0:
            return

        reported_at = time.monotonic() if now is None else now
        if not force and reported_at - self.last_reported_at < self.interval_seconds:
            return

        elapsed = _format_elapsed(reported_at - self.started_at)
        print(
            "[llm-cluster] "
            f"resolved={self.resolved:,} computed={self.computed:,} "
            f"generated={self.generated:,} cached={self.cached:,} "
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
