from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
import math
import random
import sys
from typing import Any, Literal, TypeVar

from llm_cluster.clustering import TextCluster
from llm_cluster.comparison import ComparisonInput, ComparisonResult
from llm_cluster.data import TextRow
from llm_cluster.ranking import DistanceComparator


EdgeComparisonBatch = Callable[
    [Sequence[tuple["_ComparisonEdge", "_ComparisonEdge"]]],
    Awaitable[list[bool]],
]
EdgeDistanceMatrix = Callable[[Sequence[str], Sequence[str]], Any]
_T = TypeVar("_T")

DEFAULT_DELTA = 5
DEFAULT_PROB_SORT_D = 5
DEFAULT_ORACLE_COMPARISON_BATCH_SIZE = 8192
NOTEBOOK_RECURSION_LIMIT = 10_000_000
TERMINAL_THRESHOLD = 100
SAFE_PREFIX_DIV = 2
NearestEdgeStrategy = Literal["sort", "pick-mins"]
DEFAULT_NEAREST_EDGE_STRATEGY: NearestEdgeStrategy = "sort"
VALID_NEAREST_EDGE_STRATEGIES = frozenset(("sort", "pick-mins"))

sys.setrecursionlimit(max(sys.getrecursionlimit(), NOTEBOOK_RECURSION_LIMIT))


@dataclass(frozen=True)
class WeakComparisonAlgGRound:
    index: int
    n_active_before: int
    sample1_size: int
    sample2_size: int
    edge_sample_count: int
    window_size: int
    dislocation_bound: int
    filtered_count: int
    safe_count: int
    removed_count: int
    n_active_after: int
    sample1_center_ids: tuple[str, ...]
    sample2_center_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "n_active_before": self.n_active_before,
            "sample1_size": self.sample1_size,
            "sample2_size": self.sample2_size,
            "edge_sample_count": self.edge_sample_count,
            "window_size": self.window_size,
            "dislocation_bound": self.dislocation_bound,
            "filtered_count": self.filtered_count,
            "safe_count": self.safe_count,
            "removed_count": self.removed_count,
            "n_active_after": self.n_active_after,
            "sample1_center_ids": list(self.sample1_center_ids),
            "sample2_center_ids": list(self.sample2_center_ids),
        }


@dataclass(frozen=True)
class WeakComparisonAlgGResult:
    """Result of the weak-comparison Alg-G coreset construction."""

    target_clusters: int
    correctness_probability: float
    sample1_multiplier: float
    sample2_multiplier: float
    window_multiplier: float
    terminal_multiplier: float
    sample_fraction_cap: float | None
    nearest_edge_strategy: NearestEdgeStrategy
    seed: int | None
    final_center_count: int | None
    coreset_centers: tuple[TextRow, ...]
    coreset_clusters: tuple[TextCluster, ...]
    centers: tuple[TextRow, ...]
    clusters: tuple[TextCluster, ...]
    assignments: Mapping[str, TextRow]
    coreset_assignments: Mapping[str, TextRow]
    rounds: tuple[WeakComparisonAlgGRound, ...]

    @property
    def compressed(self) -> bool:
        return self.final_center_count is not None


@dataclass(frozen=True)
class _ComparisonEdge:
    left: TextRow
    right: TextRow

    @property
    def key(self) -> tuple[str, str]:
        return (self.left.id, self.right.id)


def _edge_distance_matrix_callback(
    comparator: DistanceComparator,
) -> EdgeDistanceMatrix | None:
    edge_distance_matrix = getattr(comparator, "edge_distance_matrix", None)
    if callable(edge_distance_matrix):
        return edge_distance_matrix
    return None


def _edge_distances_for_rows(
    left_rows: Sequence[TextRow],
    right_rows: Sequence[TextRow],
    edge_distance_matrix: EdgeDistanceMatrix,
    *,
    mask_self_edges: bool,
) -> Any:
    import numpy as np

    distances = np.asarray(
        edge_distance_matrix(
            tuple(row.text for row in left_rows),
            tuple(row.text for row in right_rows),
        )
    )
    expected_shape = (len(left_rows), len(right_rows))
    if distances.shape != expected_shape:
        raise RuntimeError(
            "edge_distance_matrix returned an unexpected shape: "
            f"{distances.shape!r}, expected {expected_shape!r}."
        )

    if not mask_self_edges:
        return distances

    right_positions_by_id: dict[str, list[int]] = {}
    for right_index, row in enumerate(right_rows):
        right_positions_by_id.setdefault(row.id, []).append(right_index)

    masked_distances = distances
    copied = False
    for left_index, row in enumerate(left_rows):
        right_positions = right_positions_by_id.get(row.id)
        if not right_positions:
            continue
        if not copied:
            masked_distances = np.array(distances, copy=True)
            copied = True
        masked_distances[left_index, right_positions] = np.inf

    return masked_distances


@dataclass(frozen=True)
class _RoundSamples:
    sample1: tuple[TextRow, ...]
    sample2: tuple[TextRow, ...]


@dataclass(frozen=True)
class _CoreStructures:
    kernels: Mapping[str, tuple[TextRow, ...]]
    guards: Mapping[str, tuple[TextRow, ...]]
    core_prime: Mapping[str, Mapping[str, tuple[TextRow, ...]]]
    edge_sample_count: int


def weak_comparison_alg_g_cluster(
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    k: int,
    correctness_probability: float = 0.85,
    sample1_multiplier: float = 1.0,
    sample2_multiplier: float = 1.0,
    window_multiplier: float = 2.0,
    terminal_multiplier: float = 0.01,
    sample_fraction_cap: float | None = None,
    nearest_edge_strategy: NearestEdgeStrategy = DEFAULT_NEAREST_EDGE_STRATEGY,
    seed: int | None = None,
    final_center_count: int | None = None,
    window_size: int | None = None,
    max_rounds: int | None = None,
    comparison_batch_size: int = DEFAULT_ORACLE_COMPARISON_BATCH_SIZE,
) -> WeakComparisonAlgGResult:
    """Run the notebook implementation of weak-comparison Alg-G clustering."""

    return asyncio.run(
        weak_comparison_alg_g_cluster_async(
            rows,
            comparator,
            k=k,
            correctness_probability=correctness_probability,
            sample1_multiplier=sample1_multiplier,
            sample2_multiplier=sample2_multiplier,
            window_multiplier=window_multiplier,
            terminal_multiplier=terminal_multiplier,
            sample_fraction_cap=sample_fraction_cap,
            nearest_edge_strategy=nearest_edge_strategy,
            seed=seed,
            final_center_count=final_center_count,
            window_size=window_size,
            max_rounds=max_rounds,
            comparison_batch_size=comparison_batch_size,
        )
    )


async def weak_comparison_alg_g_cluster_async(
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    k: int,
    correctness_probability: float = 0.85,
    sample1_multiplier: float = 1.0,
    sample2_multiplier: float = 1.0,
    window_multiplier: float = 2.0,
    terminal_multiplier: float = 0.01,
    sample_fraction_cap: float | None = None,
    nearest_edge_strategy: NearestEdgeStrategy = DEFAULT_NEAREST_EDGE_STRATEGY,
    seed: int | None = None,
    final_center_count: int | None = None,
    window_size: int | None = None,
    max_rounds: int | None = None,
    comparison_batch_size: int = DEFAULT_ORACLE_COMPARISON_BATCH_SIZE,
) -> WeakComparisonAlgGResult:
    _validate_parameters(
        rows=rows,
        k=k,
        correctness_probability=correctness_probability,
        sample1_multiplier=sample1_multiplier,
        sample2_multiplier=sample2_multiplier,
        window_multiplier=window_multiplier,
        terminal_multiplier=terminal_multiplier,
        sample_fraction_cap=sample_fraction_cap,
        nearest_edge_strategy=nearest_edge_strategy,
        final_center_count=final_center_count,
        window_size=window_size,
        max_rounds=max_rounds,
        comparison_batch_size=comparison_batch_size,
    )

    return await _weak_comparison_alg_g_cluster_async_impl(
        rows,
        comparator,
        k=k,
        correctness_probability=correctness_probability,
        sample1_multiplier=sample1_multiplier,
        sample2_multiplier=sample2_multiplier,
        window_multiplier=window_multiplier,
        terminal_multiplier=terminal_multiplier,
        sample_fraction_cap=sample_fraction_cap,
        nearest_edge_strategy=nearest_edge_strategy,
        seed=seed,
        final_center_count=final_center_count,
        window_size=window_size,
        max_rounds=max_rounds,
        comparison_batch_size=comparison_batch_size,
    )


async def _weak_comparison_alg_g_cluster_async_impl(
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    k: int,
    correctness_probability: float,
    sample1_multiplier: float,
    sample2_multiplier: float,
    window_multiplier: float,
    terminal_multiplier: float,
    sample_fraction_cap: float | None,
    nearest_edge_strategy: NearestEdgeStrategy,
    seed: int | None,
    final_center_count: int | None,
    window_size: int | None,
    max_rounds: int | None,
    comparison_batch_size: int,
) -> WeakComparisonAlgGResult:
    all_rows = list(rows)
    rng = random.Random(seed)
    round_limit = max_rounds or _log_round_limit(len(all_rows))
    active = list(all_rows)
    coreset_assignments: dict[str, TextRow] = {}
    rounds: list[WeakComparisonAlgGRound] = []

    while len(active) > TERMINAL_THRESHOLD and len(rounds) < round_limit:
        n_active_before = len(active)
        ell = max(1, int(math.log2(max(n_active_before, 2))))
        delta = DEFAULT_DELTA
        cgsize = max(12, int(1.2 * ell))
        samples = _draw_round_samples(
            active,
            k=k,
            sample1_multiplier=sample1_multiplier,
            sample2_multiplier=sample2_multiplier,
            rng=rng,
        )
        if not samples.sample1 or not samples.sample2:
            break

        sample_ids = {row.id for row in (*samples.sample1, *samples.sample2)}
        non_sample_rows = [row for row in active if row.id not in sample_ids]

        x_edges = _edge_set(samples.sample1, samples.sample2)
        pi_x = await _prob_sort_edges_async(
            x_edges,
            _DirectEdgeComparator(
                comparator,
                batch_size=comparison_batch_size,
            ).compare_edges_batch_async,
            d=DEFAULT_PROB_SORT_D,
            batch_size=comparison_batch_size,
        )
        edge_sample_count = len(x_edges)
        core_structures = _build_core_structures(
            ordered_x_edges=pi_x,
            sample1=samples.sample1,
            cgsize=cgsize,
            delta=delta,
        )

        v_prime = await _filter_candidates_by_guard_proximity_async(
            non_sample_rows,
            samples.sample1,
            core_structures.guards,
            comparator,
            batch_size=comparison_batch_size,
        )

        y_edges = _edge_set(samples.sample1, v_prime)
        alg_test = _AlgTestComparator(
            comparator=comparator,
            kernels=core_structures.kernels,
            core_prime=core_structures.core_prime,
            edge_pair_batch_size=comparison_batch_size,
            comparison_batch_size=comparison_batch_size,
        )
        ordered_nearest_edges = await _ordered_nearest_edges_async(
            y_edges,
            alg_test.compare_edges_batch_async,
            nearest_edge_strategy=nearest_edge_strategy,
            rng=rng,
            batch_size=comparison_batch_size,
        )
        safe_threshold = min(
            len(ordered_nearest_edges),
            max(1, n_active_before // SAFE_PREFIX_DIV),
        )
        safe_edges = ordered_nearest_edges[:safe_threshold]
        safe_ids = {edge.right.id for edge in safe_edges}

        for edge in safe_edges:
            coreset_assignments[edge.right.id] = edge.left
        for row in samples.sample1:
            coreset_assignments[row.id] = row

        removed_ids = safe_ids | {row.id for row in samples.sample1}
        active = [row for row in active if row.id not in removed_ids]
        rounds.append(
            WeakComparisonAlgGRound(
                index=len(rounds) + 1,
                n_active_before=n_active_before,
                sample1_size=len(samples.sample1),
                sample2_size=len(samples.sample2),
                edge_sample_count=edge_sample_count,
                window_size=cgsize,
                dislocation_bound=delta,
                filtered_count=len(v_prime),
                safe_count=len(safe_edges),
                removed_count=len(removed_ids),
                n_active_after=len(active),
                sample1_center_ids=tuple(row.id for row in samples.sample1),
                sample2_center_ids=tuple(row.id for row in samples.sample2),
            )
        )

        if len(active) >= n_active_before:
            break

    for row in active:
        coreset_assignments[row.id] = row

    coreset_clusters = _build_clusters(all_rows, coreset_assignments)
    coreset_centers = tuple(cluster.center for cluster in coreset_clusters)

    if final_center_count is None:
        centers = coreset_centers
        assignments = dict(coreset_assignments)
        clusters = coreset_clusters
    else:
        centers = _select_compression_centers(
            coreset_clusters,
            all_rows,
            final_center_count,
        )
        nearest_centers = await _nearest_centers_async(
            all_rows,
            centers,
            comparator,
            batch_size=comparison_batch_size,
        )
        assignments = {
            row.id: center for row, center in zip(all_rows, nearest_centers)
        }
        clusters = _build_clusters(all_rows, assignments, center_order=centers)

    return WeakComparisonAlgGResult(
        target_clusters=k,
        correctness_probability=correctness_probability,
        sample1_multiplier=sample1_multiplier,
        sample2_multiplier=sample2_multiplier,
        window_multiplier=window_multiplier,
        terminal_multiplier=terminal_multiplier,
        sample_fraction_cap=sample_fraction_cap,
        nearest_edge_strategy=nearest_edge_strategy,
        seed=seed,
        final_center_count=final_center_count,
        coreset_centers=coreset_centers,
        coreset_clusters=coreset_clusters,
        centers=centers,
        clusters=clusters,
        assignments=assignments,
        coreset_assignments=coreset_assignments,
        rounds=tuple(rounds),
    )


@dataclass
class _DirectEdgeComparator:
    comparator: DistanceComparator
    batch_size: int

    async def compare_edges_batch_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        answers: list[bool] = []
        for chunk in _chunks(edge_pairs, self.batch_size):
            comparisons = [
                (
                    edge1.left.text,
                    edge1.right.text,
                    edge2.left.text,
                    edge2.right.text,
                )
                for edge1, edge2 in chunk
            ]
            comparison_answers = await _compare_batch_bool_async(
                self.comparator,
                comparisons,
                batch_size=self.batch_size,
            )
            answers.extend(comparison_answers)
        return answers


@dataclass
class _AlgTestComparator:
    comparator: DistanceComparator
    kernels: Mapping[str, tuple[TextRow, ...]]
    core_prime: Mapping[str, Mapping[str, tuple[TextRow, ...]]]
    edge_pair_batch_size: int
    comparison_batch_size: int

    async def compare_edges_batch_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        answers: list[bool] = []
        for chunk in _chunks(edge_pairs, self.edge_pair_batch_size):
            answers.extend(await self._compare_edge_pair_chunk_async(chunk))
        return answers

    async def _compare_edge_pair_chunk_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        test_cases = [self._build_case(edge1, edge2) for edge1, edge2 in edge_pairs]
        comparisons: list[ComparisonInput] = []
        ranges: list[tuple[int, int]] = []

        for test_case in test_cases:
            start = len(comparisons)
            comparisons.extend(test_case.comparisons or [])
            ranges.append((start, len(comparisons)))

        results = await _compare_batch_bool_async(
            self.comparator,
            comparisons,
            batch_size=self.comparison_batch_size,
        )
        return [
            test_case.resolve(results[start:end])
            for test_case, (start, end) in zip(test_cases, ranges)
        ]

    def _build_case(
        self,
        edge1: _ComparisonEdge,
        edge2: _ComparisonEdge,
    ) -> "_AlgTestCase":
        s1 = edge1.left
        s2 = edge2.left
        v1 = edge1.right
        v2 = edge2.right

        if s1.id not in self.kernels or s2.id not in self.kernels:
            return _AlgTestCase(default=True)

        if s1.id == s2.id:
            core2 = self.kernels.get(s2.id, ())
            if not core2:
                return _AlgTestCase(default=True)
            return _AlgTestCase(
                comparisons=[
                    (s1.text, v1.text, vc.text, v2.text) for vc in core2
                ],
                invert_majority=True,
            )

        core2 = self.core_prime.get(s1.id, {}).get(s2.id, ())
        if core2:
            return _AlgTestCase(
                comparisons=[
                    (s1.text, v1.text, vc.text, v2.text) for vc in core2
                ],
                invert_majority=True,
            )

        core1 = self.core_prime.get(s2.id, {}).get(s1.id, ())
        if core1:
            return _AlgTestCase(
                comparisons=[
                    (s2.text, v2.text, vc.text, v1.text) for vc in core1
                ],
                invert_majority=False,
            )

        return _AlgTestCase(default=True)


@dataclass(frozen=True)
class _AlgTestCase:
    comparisons: list[ComparisonInput] | None = None
    invert_majority: bool = True
    default: bool | None = None

    def resolve(self, results: Sequence[bool]) -> bool:
        if self.default is not None:
            return self.default
        if self.comparisons is None:
            raise RuntimeError("AlgTest case has neither comparisons nor default.")
        if len(results) != len(self.comparisons):
            raise RuntimeError("AlgTest case received an unexpected result count.")

        not_smaller_count = sum(1 for result in results if not result)
        majority_not_smaller = not_smaller_count > len(results) // 2
        if self.invert_majority:
            return not majority_not_smaller
        return majority_not_smaller


async def _prob_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    d: int,
    batch_size: int,
) -> list[_ComparisonEdge]:
    if len(edges) <= 6 * d:
        return await _window_sort_edges_async(
            edges,
            compare_batch_async,
            batch_size=batch_size,
        )

    mid = len(edges) // 2
    left_sorted = await _prob_sort_edges_async(
        edges[:mid],
        compare_batch_async,
        d=d,
        batch_size=batch_size,
    )
    right_sorted = await _prob_sort_edges_async(
        edges[mid:],
        compare_batch_async,
        d=d,
        batch_size=batch_size,
    )

    merged: list[_ComparisonEdge] = []
    left = deque(left_sorted)
    right = deque(right_sorted)
    prefix_size = 3 * d
    while len(left) + len(right) > 6 * d:
        window = [
            *_deque_prefix(left, prefix_size),
            *_deque_prefix(right, prefix_size),
        ]
        window_sorted = await _window_sort_edges_async(
            window,
            compare_batch_async,
            batch_size=batch_size,
        )
        promoted = window_sorted[:d]
        promoted_keys = {edge.key for edge in promoted}
        merged.extend(promoted)
        _remove_keys_from_deque_prefix(left, promoted_keys, prefix_size)
        _remove_keys_from_deque_prefix(right, promoted_keys, prefix_size)

    tail = [*left, *right]
    merged.extend(
        await _window_sort_edges_async(
            tail,
            compare_batch_async,
            batch_size=batch_size,
        )
    )
    return merged


def _deque_prefix(
    edges: deque[_ComparisonEdge],
    size: int,
) -> list[_ComparisonEdge]:
    return list(islice(edges, size))


def _remove_keys_from_deque_prefix(
    edges: deque[_ComparisonEdge],
    removed_keys: set[tuple[str, str]],
    size: int,
) -> None:
    if not edges or not removed_keys:
        return

    scanned = min(size, len(edges))
    kept: list[_ComparisonEdge] = []
    for _ in range(scanned):
        edge = edges.popleft()
        if edge.key not in removed_keys:
            kept.append(edge)

    edges.extendleft(reversed(kept))


async def _window_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    batch_size: int,
) -> list[_ComparisonEdge]:
    return await _window_quick_sort_edges_async(
        edges,
        compare_batch_async,
        batch_size=batch_size,
    )


async def _window_quick_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    batch_size: int,
) -> list[_ComparisonEdge]:
    if len(edges) <= 1:
        return list(edges)

    pivot = edges[0]
    less: list[_ComparisonEdge] = []
    equal: list[_ComparisonEdge] = [pivot]
    greater: list[_ComparisonEdge] = []

    partition_chunk: list[_ComparisonEdge] = []
    for edge in edges[1:]:
        partition_chunk.append(edge)
        if len(partition_chunk) >= batch_size:
            await _partition_window_against_pivot_async(
                partition_chunk,
                pivot,
                compare_batch_async,
                less,
                equal,
                greater,
            )
            partition_chunk.clear()

    if partition_chunk:
        await _partition_window_against_pivot_async(
            partition_chunk,
            pivot,
            compare_batch_async,
            less,
            equal,
            greater,
        )

    less_sorted = await _window_quick_sort_edges_async(
        less,
        compare_batch_async,
        batch_size=batch_size,
    )
    greater_sorted = await _window_quick_sort_edges_async(
        greater,
        compare_batch_async,
        batch_size=batch_size,
    )
    return [*less_sorted, *equal, *greater_sorted]


async def _partition_window_against_pivot_async(
    edges: Sequence[_ComparisonEdge],
    pivot: _ComparisonEdge,
    compare_batch_async: EdgeComparisonBatch,
    less: list[_ComparisonEdge],
    equal: list[_ComparisonEdge],
    greater: list[_ComparisonEdge],
) -> None:
    forward_results = await compare_batch_async([(edge, pivot) for edge in edges])
    unresolved_edges = [
        edge for edge, edge_is_less in zip(edges, forward_results) if not edge_is_less
    ]
    reverse_results = await compare_batch_async(
        [(pivot, edge) for edge in unresolved_edges]
    )

    reverse_by_key = {
        edge.key: pivot_is_less
        for edge, pivot_is_less in zip(unresolved_edges, reverse_results)
    }
    for edge, edge_is_less in zip(edges, forward_results):
        if edge_is_less:
            less.append(edge)
        elif reverse_by_key.get(edge.key, False):
            greater.append(edge)
        else:
            equal.append(edge)


async def _adv_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    rng: random.Random,
    *,
    batch_size: int,
) -> list[_ComparisonEdge]:
    if len(edges) <= 1:
        return list(edges)

    pivot_index = rng.randrange(len(edges))
    pivot = edges[pivot_index]
    less: list[_ComparisonEdge] = []
    greater: list[_ComparisonEdge] = []

    partition_chunk: list[_ComparisonEdge] = []
    for index, edge in enumerate(edges):
        if index == pivot_index:
            continue

        partition_chunk.append(edge)
        if len(partition_chunk) >= batch_size:
            await _partition_against_pivot_async(
                partition_chunk,
                pivot,
                compare_batch_async,
                less,
                greater,
            )
            partition_chunk.clear()

    if partition_chunk:
        await _partition_against_pivot_async(
            partition_chunk,
            pivot,
            compare_batch_async,
            less,
            greater,
        )

    left_rng, right_rng = _branch_rngs(rng)
    less_sorted = await _adv_sort_edges_async(
        less,
        compare_batch_async,
        left_rng,
        batch_size=batch_size,
    )
    greater_sorted = await _adv_sort_edges_async(
        greater,
        compare_batch_async,
        right_rng,
        batch_size=batch_size,
    )
    return [*less_sorted, pivot, *greater_sorted]


async def _ordered_nearest_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    nearest_edge_strategy: NearestEdgeStrategy,
    rng: random.Random,
    batch_size: int,
) -> list[_ComparisonEdge]:
    if nearest_edge_strategy == "sort":
        ordered_edges = await _adv_sort_edges_async(
            edges,
            compare_batch_async,
            rng,
            batch_size=batch_size,
        )
        nearest_edges = _first_edges_by_right_vertex(ordered_edges)
        return [
            edge for edge in ordered_edges if nearest_edges.get(edge.right.id) == edge
        ]

    if nearest_edge_strategy == "pick-mins":
        nearest_edges = await _pick_min_edges_by_right_vertex_async(
            edges,
            compare_batch_async,
        )
        return await _adv_sort_edges_async(
            nearest_edges,
            compare_batch_async,
            rng,
            batch_size=batch_size,
        )

    raise ValueError(f"Unknown nearest_edge_strategy: {nearest_edge_strategy!r}.")


async def _pick_min_edges_by_right_vertex_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
) -> list[_ComparisonEdge]:
    active_groups = _edges_grouped_by_right_vertex(edges)
    if not active_groups:
        return []

    while any(len(group) > 1 for group in active_groups):
        next_groups: list[list[_ComparisonEdge]] = [[] for _ in active_groups]
        comparisons: list[tuple[_ComparisonEdge, _ComparisonEdge]] = []
        comparison_groups: list[int] = []

        for group_index, group in enumerate(active_groups):
            for edge_index in range(0, len(group) - 1, 2):
                comparisons.append((group[edge_index], group[edge_index + 1]))
                comparison_groups.append(group_index)
            if len(group) % 2:
                next_groups[group_index].append(group[-1])

        if not comparisons:
            break

        results = await compare_batch_async(comparisons)
        for group_index, (left, right), is_left_less in zip(
            comparison_groups,
            comparisons,
            results,
        ):
            next_groups[group_index].append(left if is_left_less else right)

        active_groups = next_groups

    return [group[0] for group in active_groups if group]


def _edges_grouped_by_right_vertex(
    edges: Sequence[_ComparisonEdge],
) -> list[list[_ComparisonEdge]]:
    groups_by_right_id: dict[str, list[_ComparisonEdge]] = {}
    ordered_right_ids: list[str] = []
    for edge in edges:
        if edge.right.id not in groups_by_right_id:
            groups_by_right_id[edge.right.id] = []
            ordered_right_ids.append(edge.right.id)
        groups_by_right_id[edge.right.id].append(edge)

    return [groups_by_right_id[right_id] for right_id in ordered_right_ids]


async def _partition_against_pivot_async(
    edges: Sequence[_ComparisonEdge],
    pivot: _ComparisonEdge,
    compare_batch_async: EdgeComparisonBatch,
    less: list[_ComparisonEdge],
    greater: list[_ComparisonEdge],
) -> None:
    comparisons = [(edge, pivot) for edge in edges]
    edge_is_less = await compare_batch_async(comparisons)
    for edge, is_less in zip(edges, edge_is_less):
        if is_less:
            less.append(edge)
        else:
            greater.append(edge)


def _build_core_structures(
    *,
    ordered_x_edges: Sequence[_ComparisonEdge],
    sample1: Sequence[TextRow],
    cgsize: int,
    delta: int,
) -> _CoreStructures:
    x_rank = {edge.key: rank for rank, edge in enumerate(ordered_x_edges)}
    edges_by_left_id: dict[str, list[_ComparisonEdge]] = {
        row.id: [] for row in sample1
    }
    for edge in ordered_x_edges:
        edges_by_left_id.setdefault(edge.left.id, []).append(edge)

    kernels: dict[str, tuple[TextRow, ...]] = {}
    guards: dict[str, tuple[TextRow, ...]] = {}
    core_edges_by_sample: dict[str, tuple[_ComparisonEdge, ...]] = {}
    core_edge_keys: dict[str, set[tuple[str, str]]] = {}

    for sample in sample1:
        sample_edges = edges_by_left_id.get(sample.id, [])
        core_edges = sample_edges[:cgsize]
        kernels[sample.id] = tuple(edge.right for edge in core_edges)
        core_edges_by_sample[sample.id] = tuple(core_edges)
        core_edge_keys[sample.id] = {edge.key for edge in core_edges}

        edge_count = len(sample_edges)
        guard_start = min(edge_count, cgsize + delta)
        guard_end = min(edge_count, cgsize + delta + max(1, cgsize // 3))
        guards[sample.id] = tuple(
            edge.right for edge in sample_edges[guard_start:guard_end]
        )

    core_prime: dict[str, dict[str, tuple[TextRow, ...]]] = {
        row.id: {} for row in sample1
    }
    for index, sample_a in enumerate(sample1):
        for sample_b in sample1[index + 1 :]:
            pi_z = sorted(
                [
                    *core_edges_by_sample.get(sample_a.id, ()),
                    *core_edges_by_sample.get(sample_b.id, ()),
                ],
                key=lambda edge: x_rank[edge.key],
            )
            if not pi_z:
                core_prime[sample_a.id][sample_b.id] = ()
                core_prime[sample_b.id][sample_a.id] = ()
                continue

            pi_z_rank = {edge.key: rank for rank, edge in enumerate(pi_z, start=1)}
            low_rank = 2 * cgsize - delta
            high_rank = 2 * cgsize
            e_star = pi_z[-1]
            if e_star.key in core_edge_keys.get(sample_a.id, set()):
                core_prime[sample_a.id][sample_b.id] = tuple(
                    vertex
                    for vertex in kernels.get(sample_b.id, ())
                    if not _rank_in_window(
                        pi_z_rank,
                        (sample_b.id, vertex.id),
                        low_rank,
                        high_rank,
                    )
                )
                core_prime[sample_b.id][sample_a.id] = ()
            else:
                core_prime[sample_b.id][sample_a.id] = tuple(
                    vertex
                    for vertex in kernels.get(sample_a.id, ())
                    if not _rank_in_window(
                        pi_z_rank,
                        (sample_a.id, vertex.id),
                        low_rank,
                        high_rank,
                    )
                )
                core_prime[sample_a.id][sample_b.id] = ()

    return _CoreStructures(
        kernels=kernels,
        guards=guards,
        core_prime=core_prime,
        edge_sample_count=len(ordered_x_edges),
    )


async def _filter_candidates_by_guard_proximity_async(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    comparator: DistanceComparator,
    *,
    batch_size: int,
) -> list[TextRow]:
    if not candidates:
        return []

    edge_distance_matrix = _edge_distance_matrix_callback(comparator)
    if edge_distance_matrix is not None:
        return _filter_candidates_by_guard_proximity_matrix(
            candidates,
            sample1,
            guards,
            edge_distance_matrix,
        )

    guard_proximity_counts_async = getattr(
        comparator,
        "guard_proximity_counts_async",
        None,
    )
    if callable(guard_proximity_counts_async):
        return await _filter_candidates_by_guard_proximity_fast_async(
            candidates,
            sample1,
            guards,
            guard_proximity_counts_async,
        )

    guard_proximity_counts = getattr(comparator, "guard_proximity_counts", None)
    if callable(guard_proximity_counts):
        return _filter_candidates_by_guard_proximity_fast(
            candidates,
            sample1,
            guards,
            guard_proximity_counts,
        )

    prox_max = {row.id: 0 for row in candidates}
    for sample in sample1:
        guard_rows = guards.get(sample.id, ())
        if not guard_rows:
            continue

        counts = {row.id: 0 for row in candidates}
        comparisons: list[ComparisonInput] = []
        compared_row_ids: list[str] = []
        for row in candidates:
            for guard in guard_rows:
                comparisons.append(
                    (sample.text, row.text, sample.text, guard.text)
                )
                compared_row_ids.append(row.id)
                if len(comparisons) >= batch_size:
                    await _add_proximity_counts_async(
                        comparator,
                        comparisons,
                        compared_row_ids,
                        counts,
                        batch_size=batch_size,
                    )
                    comparisons.clear()
                    compared_row_ids.clear()

        if comparisons:
            await _add_proximity_counts_async(
                comparator,
                comparisons,
                compared_row_ids,
                counts,
                batch_size=batch_size,
            )
        for row_id, count in counts.items():
            prox_max[row_id] = max(prox_max[row_id], count)

    return _filter_candidates_by_proximity(candidates, prox_max)


def _filter_candidates_by_guard_proximity_matrix(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    edge_distance_matrix: EdgeDistanceMatrix,
) -> list[TextRow]:
    import numpy as np

    candidate_ids = tuple(row.id for row in candidates)
    prox_max = np.zeros(len(candidates), dtype=np.int64)

    active_samples = tuple(row for row in sample1 if guards.get(row.id))
    if active_samples:
        all_guard_texts = _unique_texts(
            guard.text
            for sample in active_samples
            for guard in guards.get(sample.id, ())
        )
        guard_position_by_text = {
            text: index for index, text in enumerate(all_guard_texts)
        }

        candidate_distances = _edge_distances_for_rows(
            active_samples,
            candidates,
            edge_distance_matrix,
            mask_self_edges=False,
        )
        guard_distances = np.asarray(
            edge_distance_matrix(
                tuple(sample.text for sample in active_samples),
                all_guard_texts,
            )
        )
        expected_guard_shape = (len(active_samples), len(all_guard_texts))
        if guard_distances.shape != expected_guard_shape:
            raise RuntimeError(
                "edge_distance_matrix returned an unexpected guard shape: "
                f"{guard_distances.shape!r}, expected {expected_guard_shape!r}."
            )

        for sample_index, sample in enumerate(active_samples):
            guard_indexes = [
                guard_position_by_text[guard.text]
                for guard in guards.get(sample.id, ())
            ]
            counts = np.sum(
                candidate_distances[sample_index, :, None]
                < guard_distances[sample_index, guard_indexes][None, :],
                axis=1,
            )
            prox_max = np.maximum(prox_max, counts)

    return _filter_candidates_by_proximity(
        candidates,
        {
            row_id: int(count)
            for row_id, count in zip(candidate_ids, prox_max)
        },
    )


def _filter_candidates_by_guard_proximity_fast(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    guard_proximity_counts: Callable[
        [str, Sequence[str], Sequence[str]],
        Sequence[int],
    ],
) -> list[TextRow]:
    candidate_ids = tuple(row.id for row in candidates)
    candidate_texts = tuple(row.text for row in candidates)
    prox_max = {row_id: 0 for row_id in candidate_ids}

    for sample in sample1:
        guard_rows = guards.get(sample.id, ())
        if not guard_rows:
            continue

        counts = guard_proximity_counts(
            sample.text,
            candidate_texts,
            tuple(guard.text for guard in guard_rows),
        )
        if len(counts) != len(candidate_ids):
            raise RuntimeError(
                "guard_proximity_counts returned an unexpected number of counts."
            )

        for row_id, count in zip(candidate_ids, counts):
            prox_max[row_id] = max(prox_max[row_id], int(count))

    return _filter_candidates_by_proximity(candidates, prox_max)


async def _filter_candidates_by_guard_proximity_fast_async(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    guard_proximity_counts_async: Callable[
        [str, Sequence[str], Sequence[str]],
        Awaitable[Sequence[int]],
    ],
) -> list[TextRow]:
    candidate_ids = tuple(row.id for row in candidates)
    candidate_texts = tuple(row.text for row in candidates)
    prox_max = {row_id: 0 for row_id in candidate_ids}
    tasks = []

    for sample in sample1:
        guard_rows = guards.get(sample.id, ())
        if not guard_rows:
            continue
        tasks.append(
            guard_proximity_counts_async(
                sample.text,
                candidate_texts,
                tuple(guard.text for guard in guard_rows),
            )
        )

    for counts in await asyncio.gather(*tasks):
        if len(counts) != len(candidate_ids):
            raise RuntimeError(
                "guard_proximity_counts returned an unexpected number of counts."
            )
        for row_id, count in zip(candidate_ids, counts):
            prox_max[row_id] = max(prox_max[row_id], int(count))

    return _filter_candidates_by_proximity(candidates, prox_max)


def _filter_candidates_by_proximity(
    candidates: Sequence[TextRow],
    prox_max: Mapping[str, int],
) -> list[TextRow]:
    candidate_positions = {row.id: index for index, row in enumerate(candidates)}
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (prox_max[row.id], candidate_positions[row.id]),
    )
    cutoff = int(len(sorted_candidates) * 0.75)
    return sorted_candidates[:cutoff]


async def _add_proximity_counts_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
    compared_row_ids: list[str],
    counts: dict[str, int],
    *,
    batch_size: int,
) -> None:
    results = await _compare_batch_async(
        comparator,
        comparisons,
        batch_size=batch_size,
    )
    for row_id, result in zip(compared_row_ids, results):
        if result.is_ab_less_than_cd:
            counts[row_id] += 1


def _edge_set(
    left_rows: Sequence[TextRow],
    right_rows: Sequence[TextRow],
) -> list[_ComparisonEdge]:
    return [
        _ComparisonEdge(left=left, right=right)
        for left in left_rows
        for right in right_rows
        if left.id != right.id
    ]


def _unique_texts(texts: Sequence[str] | Iterator[str]) -> tuple[str, ...]:
    by_text: dict[str, None] = {}
    for text in texts:
        by_text.setdefault(text, None)
    return tuple(by_text)


def _rank_in_window(
    rank_by_key: Mapping[tuple[str, str], int],
    edge_key: tuple[str, str],
    low_rank: int,
    high_rank: int,
) -> bool:
    rank = rank_by_key.get(edge_key, math.inf)
    return low_rank <= rank <= high_rank


def _first_edges_by_right_vertex(
    ordered_edges: Sequence[_ComparisonEdge],
) -> dict[str, _ComparisonEdge]:
    nearest: dict[str, _ComparisonEdge] = {}
    for edge in ordered_edges:
        nearest.setdefault(edge.right.id, edge)
    return nearest


async def _nearest_centers_async(
    rows: Sequence[TextRow],
    centers: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    batch_size: int,
) -> list[TextRow]:
    if not centers:
        raise ValueError("centers must contain at least one item.")

    center_by_id = {center.id: center for center in centers}
    nearest: list[TextRow] = []
    fixed: list[bool] = []
    first_center = centers[0]

    for row in rows:
        own_center = center_by_id.get(row.id)
        if own_center is None:
            nearest.append(first_center)
            fixed.append(False)
        else:
            nearest.append(own_center)
            fixed.append(True)

    for candidate in centers[1:]:
        comparisons: list[ComparisonInput] = []
        comparison_indexes: list[int] = []
        for index, row in enumerate(rows):
            if fixed[index]:
                continue

            current = nearest[index]
            if candidate.id == current.id:
                continue

            comparisons.append((row.text, candidate.text, row.text, current.text))
            comparison_indexes.append(index)

        results = await _compare_batch_async(
            comparator,
            comparisons,
            batch_size=batch_size,
        )
        for index, result in zip(comparison_indexes, results):
            if result.is_ab_less_than_cd:
                nearest[index] = candidate

    return nearest


async def _compare_batch_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
    *,
    batch_size: int = DEFAULT_ORACLE_COMPARISON_BATCH_SIZE,
) -> list[ComparisonResult]:
    if not comparisons:
        return []
    if len(comparisons) > batch_size:
        results: list[ComparisonResult] = []
        for chunk in _chunks(comparisons, batch_size):
            results.extend(await _compare_batch_raw_async(comparator, list(chunk)))
        return results

    return await _compare_batch_raw_async(comparator, comparisons)


async def _compare_batch_bool_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
    *,
    batch_size: int = DEFAULT_ORACLE_COMPARISON_BATCH_SIZE,
) -> list[bool]:
    if not comparisons:
        return []
    if len(comparisons) > batch_size:
        answers: list[bool] = []
        for chunk in _chunks(comparisons, batch_size):
            answers.extend(
                await _compare_batch_bool_raw_async(comparator, list(chunk))
            )
        return answers

    return await _compare_batch_bool_raw_async(comparator, comparisons)


async def _compare_batch_bool_raw_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
) -> list[bool]:
    if not comparisons:
        return []

    compare_batch_bool_async = getattr(comparator, "compare_batch_bool_async", None)
    if compare_batch_bool_async is not None:
        return [
            bool(answer)
            for answer in await compare_batch_bool_async(comparisons)
        ]

    compare_batch_bool = getattr(comparator, "compare_batch_bool", None)
    if compare_batch_bool is not None:
        return [bool(answer) for answer in compare_batch_bool(comparisons)]

    results = await _compare_batch_raw_async(comparator, comparisons)
    return [result.is_ab_less_than_cd for result in results]


async def _compare_batch_raw_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
) -> list[ComparisonResult]:
    if not comparisons:
        return []

    compare_batch_async = getattr(comparator, "compare_batch_async", None)
    if compare_batch_async is not None:
        return await compare_batch_async(comparisons)

    compare_batch = getattr(comparator, "compare_batch", None)
    if compare_batch is not None:
        return compare_batch(comparisons)

    return [comparator.compare(*comparison) for comparison in comparisons]


def _draw_round_samples(
    active: Sequence[TextRow],
    *,
    k: int,
    sample1_multiplier: float,
    sample2_multiplier: float,
    rng: random.Random,
) -> _RoundSamples:
    target1 = max(1, math.ceil(sample1_multiplier * max(6, 3 * k)))
    target2 = max(1, math.ceil(sample2_multiplier * max(24, 8 * k, 80)))

    sample1_count = min(len(active), target1)
    sample2_count = min(len(active), target2)
    sample1 = tuple(rng.sample(list(active), sample1_count))
    sample2 = tuple(rng.sample(list(active), sample2_count))
    return _RoundSamples(sample1=sample1, sample2=sample2)


def _build_clusters(
    rows: Sequence[TextRow],
    assignments: Mapping[str, TextRow],
    *,
    center_order: Sequence[TextRow] | None = None,
) -> tuple[TextCluster, ...]:
    rows_by_center_id: dict[str, list[TextRow]] = {}
    centers_by_id: dict[str, TextRow] = {}
    for row in rows:
        center = assignments[row.id]
        centers_by_id[center.id] = center
        rows_by_center_id.setdefault(center.id, []).append(row)

    if center_order is None:
        ordered_center_ids = [
            center_id
            for center_id, _ in sorted(
                rows_by_center_id.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]
    else:
        ordered_center_ids = [
            center.id for center in center_order if center.id in rows_by_center_id
        ]
        ordered_center_id_set = set(ordered_center_ids)
        ordered_center_ids.extend(
            center_id
            for center_id in sorted(rows_by_center_id)
            if center_id not in ordered_center_id_set
        )

    return tuple(
        TextCluster(
            center=centers_by_id[center_id],
            rows=tuple(rows_by_center_id[center_id]),
        )
        for center_id in ordered_center_ids
    )


def _select_compression_centers(
    coreset_clusters: Sequence[TextCluster],
    rows: Sequence[TextRow],
    count: int,
) -> tuple[TextRow, ...]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    if count > len(rows):
        raise ValueError(
            "final_center_count must be <= the number of clustered rows "
            "for exact-k compression."
        )

    selected: list[TextRow] = []
    selected_ids: set[str] = set()
    ordered_clusters = sorted(
        coreset_clusters,
        key=lambda cluster: (-len(cluster.rows), cluster.center.id),
    )

    for cluster in ordered_clusters:
        if cluster.center.id in selected_ids:
            continue
        selected.append(cluster.center)
        selected_ids.add(cluster.center.id)
        if len(selected) == count:
            return tuple(selected)

    for cluster in ordered_clusters:
        for row in sorted(cluster.rows, key=lambda item: item.id):
            if row.id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row.id)
            if len(selected) == count:
                return tuple(selected)

    for row in sorted(rows, key=lambda item: item.id):
        if row.id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row.id)
        if len(selected) == count:
            return tuple(selected)

    raise RuntimeError("Could not select enough compression centers.")


def _validate_parameters(
    *,
    rows: Sequence[TextRow],
    k: int,
    correctness_probability: float,
    sample1_multiplier: float,
    sample2_multiplier: float,
    window_multiplier: float,
    terminal_multiplier: float,
    sample_fraction_cap: float | None,
    nearest_edge_strategy: str,
    final_center_count: int | None,
    window_size: int | None,
    max_rounds: int | None,
    comparison_batch_size: int,
) -> None:
    if not rows:
        raise ValueError("rows must contain at least one item.")
    row_ids = [row.id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("rows must have unique ids.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not 0.0 <= correctness_probability <= 1.0:
        raise ValueError(
            "correctness_probability must be in [0, 1]."
        )
    if sample1_multiplier <= 0:
        raise ValueError("sample1_multiplier must be positive.")
    if sample2_multiplier <= 0:
        raise ValueError("sample2_multiplier must be positive.")
    if window_multiplier <= 0:
        raise ValueError("window_multiplier must be positive.")
    if terminal_multiplier < 0:
        raise ValueError("terminal_multiplier must be non-negative.")
    if sample_fraction_cap is not None:
        raise ValueError(
            "sample_fraction_cap is not part of the ICLR.ipynb implementation."
        )
    if nearest_edge_strategy not in VALID_NEAREST_EDGE_STRATEGIES:
        raise ValueError(
            "nearest_edge_strategy must be one of: "
            f"{', '.join(sorted(VALID_NEAREST_EDGE_STRATEGIES))}."
        )
    if final_center_count is not None and final_center_count < 1:
        raise ValueError("final_center_count must be at least 1 when provided.")
    if final_center_count is not None and final_center_count > len(rows):
        raise ValueError(
            "final_center_count must be <= the number of rows when provided."
        )
    if window_size is not None:
        raise ValueError("window_size is not part of the ICLR.ipynb implementation.")
    if max_rounds is not None and max_rounds < 1:
        raise ValueError("max_rounds must be at least 1 when provided.")
    if comparison_batch_size < 1:
        raise ValueError("comparison_batch_size must be at least 1.")


def _log_round_limit(n_rows: int) -> int:
    return int(math.log2(max(n_rows, 2))) + 1


def _branch_rngs(rng: random.Random) -> tuple[random.Random, random.Random]:
    return random.Random(rng.getrandbits(64)), random.Random(rng.getrandbits(64))


def _chunks(
    values: Sequence[_T],
    size: int,
) -> Iterator[Sequence[_T]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
