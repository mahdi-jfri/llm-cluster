from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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
IndexComparisonInput = tuple[int, int, int, int]
IndexAlgTestSpec = tuple[int, int, tuple[int, ...], int, bool]
_T = TypeVar("_T")

DEFAULT_DELTA = 5
DEFAULT_ORACLE_COMPARISON_BATCH_SIZE = 8192
NOTEBOOK_RECURSION_LIMIT = 10_000_000
TERMINAL_THRESHOLD = 100
SAFE_PREFIX_DIVISOR = 2
NearestEdgeStrategy = Literal["sort"]
DEFAULT_NEAREST_EDGE_STRATEGY: NearestEdgeStrategy = "sort"
VALID_NEAREST_EDGE_STRATEGIES = frozenset(("sort",))

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

    return distances


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

    _print_weak_comparison_stage(
        "start",
        rows=len(all_rows),
        k=k,
        round_limit=round_limit,
        final_center_count=final_center_count,
    )

    while len(active) > TERMINAL_THRESHOLD and len(rounds) < round_limit:
        round_index = len(rounds) + 1
        n_active_before = len(active)
        ell = max(1, int(math.log2(max(n_active_before, 2))))
        delta = DEFAULT_DELTA
        cgsize = max(12, int(1.2 * ell))
        _print_weak_comparison_stage(
            "draw_round_samples",
            round=round_index,
            active=n_active_before,
        )
        samples = _draw_round_samples(
            active,
            k=k,
            sample1_multiplier=sample1_multiplier,
            sample2_multiplier=sample2_multiplier,
            rng=rng,
        )

        sample_ids = {row.id for row in (*samples.sample1, *samples.sample2)}
        non_sample_rows = [row for row in active if row.id not in sample_ids]

        x_edges = _edge_set(samples.sample1, samples.sample2)
        _print_weak_comparison_stage(
            "sort_sample_edges",
            round=round_index,
            sample1=len(samples.sample1),
            sample2=len(samples.sample2),
            edges=len(x_edges),
        )
        pi_x = await _quick_sort_edges_async(
            x_edges,
            _DirectEdgeComparator(
                comparator,
                batch_size=comparison_batch_size,
            ).compare_edges_batch_async,
            batch_size=comparison_batch_size,
        )
        edge_sample_count = len(x_edges)
        _print_weak_comparison_stage(
            "build_core_structures",
            round=round_index,
            ordered_edges=len(pi_x),
            window_size=cgsize,
            dislocation_bound=delta,
        )
        core_structures = _build_core_structures(
            ordered_x_edges=pi_x,
            sample1=samples.sample1,
            cgsize=cgsize,
            delta=delta,
        )

        _print_weak_comparison_stage(
            "filter_candidates",
            round=round_index,
            candidates=len(non_sample_rows),
            guards=len(core_structures.guards),
        )
        v_prime = await _filter_candidates_by_guard_proximity_async(
            non_sample_rows,
            samples.sample1,
            core_structures.guards,
            comparator,
            batch_size=comparison_batch_size,
        )

        y_edges = _edge_set(samples.sample1, v_prime)
        _print_weak_comparison_stage(
            "order_nearest_edges",
            round=round_index,
            filtered=len(v_prime),
            edges=len(y_edges),
            strategy=nearest_edge_strategy,
        )
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
        _print_weak_comparison_stage(
            "remove_safe_prefix",
            round=round_index,
            ordered_nearest_edges=len(ordered_nearest_edges),
        )
        safe_threshold = min(
            len(ordered_nearest_edges),
            max(1, n_active_before // SAFE_PREFIX_DIVISOR),
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

    _print_weak_comparison_stage("assign_terminal_rows", active=len(active))
    for row in active:
        coreset_assignments[row.id] = row

    _print_weak_comparison_stage(
        "build_coreset_clusters",
        assignments=len(coreset_assignments),
    )
    coreset_clusters = _build_clusters(all_rows, coreset_assignments)
    coreset_centers = tuple(cluster.center for cluster in coreset_clusters)

    if final_center_count is None:
        centers = coreset_centers
        assignments = dict(coreset_assignments)
        clusters = coreset_clusters
    else:
        _print_weak_comparison_stage(
            "compress_exact_k",
            coreset_centers=len(coreset_centers),
            final_center_count=final_center_count,
        )
        centers = _select_compression_centers(
            coreset_clusters,
            all_rows,
            final_center_count,
        )
        _print_weak_comparison_stage(
            "assign_exact_k_centers",
            rows=len(all_rows),
            centers=len(centers),
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


def _print_weak_comparison_stage(stage: str, **details: object) -> None:
    detail_text = " ".join(
        f"{key}={value}" for key, value in details.items() if value is not None
    )
    suffix = f" {detail_text}" if detail_text else ""
    print(
        f"[llm-cluster] weak-comparison stage={stage}{suffix}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class _DirectEdgeComparator:
    comparator: DistanceComparator
    batch_size: int
    _comparison_index_by_text: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    async def compare_edges_batch_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        if _has_index_batch_bool_comparator(self.comparator):
            return await self._compare_edges_batch_by_index_async(edge_pairs)

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

    async def _compare_edges_batch_by_index_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        import numpy as np

        answers: list[bool] = []
        for chunk in _chunks(edge_pairs, self.batch_size):
            comparison_array = np.empty((len(chunk), 4), dtype=np.int64)
            for index, (edge1, edge2) in enumerate(chunk):
                comparison_array[index, 0] = self._row_index(edge1.left)
                comparison_array[index, 1] = self._row_index(edge1.right)
                comparison_array[index, 2] = self._row_index(edge2.left)
                comparison_array[index, 3] = self._row_index(edge2.right)

            comparison_answers = await _compare_index_array_bool_async(
                self.comparator,
                comparison_array,
                batch_size=self.batch_size,
            )
            answers.extend(bool(answer) for answer in comparison_answers)
        return answers

    def _row_index(self, row: TextRow) -> int:
        try:
            return self._comparison_index_by_text[row.text]
        except KeyError:
            index_for_text = getattr(self.comparator, "comparison_index_for_text")
            index = int(index_for_text(row.text))
            self._comparison_index_by_text[row.text] = index
            return index


@dataclass
class _AlgTestComparator:
    comparator: DistanceComparator
    kernels: Mapping[str, tuple[TextRow, ...]]
    core_prime: Mapping[str, Mapping[str, tuple[TextRow, ...]]]
    edge_pair_batch_size: int
    comparison_batch_size: int
    _comparison_index_by_text: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _kernel_index_by_id: dict[str, tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _core_prime_index_by_id_pair: dict[tuple[str, str], tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    async def compare_edges_batch_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        answers: list[bool] = []
        for chunk in self._edge_pair_chunks(edge_pairs):
            answers.extend(await self._compare_edge_pair_chunk_async(chunk))
        return answers

    async def _compare_edge_pair_chunk_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        if _has_index_grouped_majority_comparator(
            self.comparator
        ) or _has_index_batch_bool_comparator(self.comparator):
            return await self._compare_edge_pair_chunk_by_index_async(edge_pairs)

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

    async def _compare_edge_pair_chunk_by_index_async(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> list[bool]:
        import numpy as np

        answers = [True] * len(edge_pairs)
        specs: list[tuple[int, int, int, tuple[int, ...], int, bool]] = []
        comparison_count = 0

        for pair_index, (edge1, edge2) in enumerate(edge_pairs):
            spec = self._build_index_case_spec(edge1, edge2)
            if spec is None:
                continue

            ab_left, ab_right, cd_left_values, cd_right, invert_majority = spec
            if not cd_left_values:
                continue

            specs.append(
                (
                    pair_index,
                    ab_left,
                    ab_right,
                    cd_left_values,
                    cd_right,
                    invert_majority,
                )
            )
            comparison_count += len(cd_left_values)

        if comparison_count == 0:
            return answers

        if _has_index_grouped_majority_comparator(self.comparator):
            pair_indexes = np.empty(len(specs), dtype=np.intp)
            ab_left_indexes = np.empty(len(specs), dtype=np.int64)
            ab_right_indexes = np.empty(len(specs), dtype=np.int64)
            cd_right_indexes = np.empty(len(specs), dtype=np.int64)
            invert_majority = np.empty(len(specs), dtype=bool)
            cd_left_indexes = np.empty(comparison_count, dtype=np.int64)
            offsets = np.empty(len(specs) + 1, dtype=np.int64)
            offsets[0] = 0

            offset = 0
            for spec_index, (
                pair_index,
                ab_left,
                ab_right,
                cd_left_values,
                cd_right,
                invert,
            ) in enumerate(specs):
                end = offset + len(cd_left_values)
                pair_indexes[spec_index] = pair_index
                ab_left_indexes[spec_index] = ab_left
                ab_right_indexes[spec_index] = ab_right
                cd_right_indexes[spec_index] = cd_right
                invert_majority[spec_index] = invert
                cd_left_indexes[offset:end] = cd_left_values
                offsets[spec_index + 1] = end
                offset = end

            grouped_results = await _compare_index_grouped_majority_bool_async(
                self.comparator,
                ab_left_indexes=ab_left_indexes,
                ab_right_indexes=ab_right_indexes,
                cd_left_indexes=cd_left_indexes,
                cd_right_indexes=cd_right_indexes,
                offsets=offsets,
                invert_majority=invert_majority,
            )
            if len(grouped_results) != len(specs):
                raise RuntimeError(
                    "Indexed comparator returned an unexpected result count."
                )
            for pair_index, answer in zip(pair_indexes, grouped_results):
                answers[int(pair_index)] = bool(answer)
            return answers

        comparisons = np.empty((comparison_count, 4), dtype=np.int64)
        ranges: list[tuple[int, int, int, bool]] = []
        offset = 0
        for (
            pair_index,
            ab_left,
            ab_right,
            cd_left_values,
            cd_right,
            invert_majority,
        ) in specs:
            start = offset
            end = start + len(cd_left_values)
            comparisons[start:end, 0] = ab_left
            comparisons[start:end, 1] = ab_right
            comparisons[start:end, 2] = cd_left_values
            comparisons[start:end, 3] = cd_right
            ranges.append((pair_index, start, end, invert_majority))
            offset = end

        results = await _compare_index_array_bool_async(
            self.comparator,
            comparisons,
            batch_size=self.comparison_batch_size,
        )
        result_array = np.asarray(results, dtype=bool)
        if len(result_array) != comparison_count:
            raise RuntimeError(
                "Indexed comparator returned an unexpected result count."
            )

        not_smaller_prefix = np.empty(comparison_count + 1, dtype=np.int64)
        not_smaller_prefix[0] = 0
        np.cumsum(~result_array, out=not_smaller_prefix[1:])
        for pair_index, start, end, invert_majority in ranges:
            answers[pair_index] = _resolve_alg_test_count(
                not_smaller_count=int(
                    not_smaller_prefix[end] - not_smaller_prefix[start]
                ),
                result_count=end - start,
                invert_majority=invert_majority,
            )

        return answers

    def _edge_pair_chunks(
        self,
        edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    ) -> Iterator[list[tuple[_ComparisonEdge, _ComparisonEdge]]]:
        chunk: list[tuple[_ComparisonEdge, _ComparisonEdge]] = []
        comparison_count = 0
        max_edge_pairs = max(1, self.edge_pair_batch_size)
        max_comparisons = max(1, self.comparison_batch_size)

        for edge_pair in edge_pairs:
            pair_comparison_count = self._edge_pair_comparison_count(*edge_pair)
            if chunk and (
                len(chunk) >= max_edge_pairs
                or comparison_count + pair_comparison_count > max_comparisons
            ):
                yield chunk
                chunk = []
                comparison_count = 0

            chunk.append(edge_pair)
            comparison_count += pair_comparison_count

        if chunk:
            yield chunk

    def _edge_pair_comparison_count(
        self,
        edge1: _ComparisonEdge,
        edge2: _ComparisonEdge,
    ) -> int:
        s1 = edge1.left
        s2 = edge2.left
        if s1.id not in self.kernels or s2.id not in self.kernels:
            return 0

        if s1.id == s2.id:
            return len(self.kernels.get(s2.id, ()))

        core2 = self.core_prime.get(s1.id, {}).get(s2.id, ())
        if core2:
            return len(core2)

        core1 = self.core_prime.get(s2.id, {}).get(s1.id, ())
        return len(core1)

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

    def _build_index_case_spec(
        self,
        edge1: _ComparisonEdge,
        edge2: _ComparisonEdge,
    ) -> IndexAlgTestSpec | None:
        s1 = edge1.left
        s2 = edge2.left
        v1 = edge1.right
        v2 = edge2.right

        if s1.id not in self.kernels or s2.id not in self.kernels:
            return None

        if s1.id == s2.id:
            core2 = self._kernel_indexes(s2.id)
            if not core2:
                return None
            return (
                self._row_index(s1),
                self._row_index(v1),
                core2,
                self._row_index(v2),
                True,
            )

        core2 = self._core_prime_indexes(s1.id, s2.id)
        if core2:
            return (
                self._row_index(s1),
                self._row_index(v1),
                core2,
                self._row_index(v2),
                True,
            )

        core1 = self._core_prime_indexes(s2.id, s1.id)
        if core1:
            return (
                self._row_index(s2),
                self._row_index(v2),
                core1,
                self._row_index(v1),
                False,
            )

        return None

    def _row_index(self, row: TextRow) -> int:
        try:
            return self._comparison_index_by_text[row.text]
        except KeyError:
            index_for_text = getattr(self.comparator, "comparison_index_for_text")
            index = int(index_for_text(row.text))
            self._comparison_index_by_text[row.text] = index
            return index

    def _kernel_indexes(self, row_id: str) -> tuple[int, ...]:
        try:
            return self._kernel_index_by_id[row_id]
        except KeyError:
            indexes = tuple(
                self._row_index(row) for row in self.kernels.get(row_id, ())
            )
            self._kernel_index_by_id[row_id] = indexes
            return indexes

    def _core_prime_indexes(
        self,
        left_id: str,
        right_id: str,
    ) -> tuple[int, ...]:
        key = (left_id, right_id)
        try:
            return self._core_prime_index_by_id_pair[key]
        except KeyError:
            indexes = tuple(
                self._row_index(row)
                for row in self.core_prime.get(left_id, {}).get(right_id, ())
            )
            self._core_prime_index_by_id_pair[key] = indexes
            return indexes


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


def _resolve_alg_test_count(
    *,
    not_smaller_count: int,
    result_count: int,
    invert_majority: bool,
) -> bool:
    majority_not_smaller = not_smaller_count > result_count // 2
    if invert_majority:
        return not majority_not_smaller
    return majority_not_smaller


@dataclass
class _FlattenedEdgeSortNode:
    edges: list[_ComparisonEdge]
    middle: list[_ComparisonEdge] | None = None
    less_child: "_FlattenedEdgeSortNode | None" = None
    greater_child: "_FlattenedEdgeSortNode | None" = None


@dataclass
class _QuickSortPartition:
    node: _FlattenedEdgeSortNode
    pivot: _ComparisonEdge
    less: list[_ComparisonEdge]
    equal: list[_ComparisonEdge]
    greater: list[_ComparisonEdge]


@dataclass
class _AdvSortPartition:
    node: _FlattenedEdgeSortNode
    pivot: _ComparisonEdge
    less: list[_ComparisonEdge]
    greater: list[_ComparisonEdge]
    left_rng: random.Random
    right_rng: random.Random


async def _quick_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    batch_size: int,
) -> list[_ComparisonEdge]:
    root = _FlattenedEdgeSortNode(list(edges))
    active = [root]
    height = 0

    while active:
        _print_sort_level(
            "quick_sort_edges",
            height,
            [len(node.edges) for node in active],
        )
        partitions: list[_QuickSortPartition] = []
        forward_pairs: list[tuple[_ComparisonEdge, _ComparisonEdge]] = []
        forward_meta: list[tuple[_QuickSortPartition, _ComparisonEdge]] = []

        for node in active:
            if len(node.edges) <= 1:
                continue

            pivot = node.edges[0]
            partition = _QuickSortPartition(
                node=node,
                pivot=pivot,
                less=[],
                equal=[pivot],
                greater=[],
            )
            partitions.append(partition)
            for edge in node.edges[1:]:
                forward_pairs.append((edge, pivot))
                forward_meta.append((partition, edge))

        if not partitions:
            break

        forward_results = await _compare_edge_pairs_in_batches_async(
            forward_pairs,
            compare_batch_async,
            batch_size=batch_size,
        )
        reverse_pairs: list[tuple[_ComparisonEdge, _ComparisonEdge]] = []
        reverse_meta: list[tuple[_QuickSortPartition, _ComparisonEdge]] = []
        for (partition, edge), edge_is_less in zip(forward_meta, forward_results):
            if edge_is_less:
                partition.less.append(edge)
            else:
                reverse_pairs.append((partition.pivot, edge))
                reverse_meta.append((partition, edge))

        reverse_results = await _compare_edge_pairs_in_batches_async(
            reverse_pairs,
            compare_batch_async,
            batch_size=batch_size,
        )
        for (partition, edge), pivot_is_less in zip(reverse_meta, reverse_results):
            if pivot_is_less:
                partition.greater.append(edge)
            else:
                partition.equal.append(edge)

        next_active: list[_FlattenedEdgeSortNode] = []
        for partition in partitions:
            less_child = _FlattenedEdgeSortNode(partition.less)
            greater_child = _FlattenedEdgeSortNode(partition.greater)
            partition.node.middle = partition.equal
            partition.node.less_child = less_child
            partition.node.greater_child = greater_child
            partition.node.edges = []
            next_active.extend((less_child, greater_child))
        active = next_active
        height += 1

    return _flatten_edge_sort_tree(root)


async def _adv_sort_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    rng: random.Random,
    *,
    batch_size: int,
) -> list[_ComparisonEdge]:
    root = _FlattenedEdgeSortNode(list(edges))
    active: list[tuple[_FlattenedEdgeSortNode, random.Random]] = [(root, rng)]
    height = 0

    while active:
        _print_sort_level(
            "adv_sort_edges",
            height,
            [len(node.edges) for node, _ in active],
        )
        partitions: list[_AdvSortPartition] = []
        edge_pairs: list[tuple[_ComparisonEdge, _ComparisonEdge]] = []
        pair_meta: list[tuple[_AdvSortPartition, _ComparisonEdge]] = []

        for node, node_rng in active:
            if len(node.edges) <= 1:
                continue

            pivot_index = node_rng.randrange(len(node.edges))
            pivot = node.edges[pivot_index]
            left_rng, right_rng = _branch_rngs(node_rng)
            partition = _AdvSortPartition(
                node=node,
                pivot=pivot,
                less=[],
                greater=[],
                left_rng=left_rng,
                right_rng=right_rng,
            )
            partitions.append(partition)

            for index, edge in enumerate(node.edges):
                if index == pivot_index:
                    continue
                edge_pairs.append((edge, pivot))
                pair_meta.append((partition, edge))

        if not partitions:
            break

        edge_is_less = await _compare_edge_pairs_in_batches_async(
            edge_pairs,
            compare_batch_async,
            batch_size=batch_size,
        )
        for (partition, edge), is_less in zip(pair_meta, edge_is_less):
            if is_less:
                partition.less.append(edge)
            else:
                partition.greater.append(edge)

        next_active: list[tuple[_FlattenedEdgeSortNode, random.Random]] = []
        for partition in partitions:
            less_child = _FlattenedEdgeSortNode(partition.less)
            greater_child = _FlattenedEdgeSortNode(partition.greater)
            partition.node.middle = [partition.pivot]
            partition.node.less_child = less_child
            partition.node.greater_child = greater_child
            partition.node.edges = []
            next_active.extend(
                (
                    (less_child, partition.left_rng),
                    (greater_child, partition.right_rng),
                )
            )
        active = next_active
        height += 1

    return _flatten_edge_sort_tree(root)


def _print_sort_level(
    stage: str,
    height: int,
    sizes: Sequence[int],
) -> None:
    _print_weak_comparison_stage(
        stage,
        height=height,
        partitions=len(sizes),
        active_edges=sum(sizes),
        max_size=max(sizes, default=0),
    )


async def _compare_edge_pairs_in_batches_async(
    edge_pairs: Sequence[tuple[_ComparisonEdge, _ComparisonEdge]],
    compare_batch_async: EdgeComparisonBatch,
    *,
    batch_size: int,
) -> list[bool]:
    if not edge_pairs:
        return []

    answers: list[bool] = []
    for chunk in _chunks(edge_pairs, batch_size):
        answers.extend(await compare_batch_async(chunk))
    return answers


def _flatten_edge_sort_tree(root: _FlattenedEdgeSortNode) -> list[_ComparisonEdge]:
    sorted_edges: list[_ComparisonEdge] = []
    stack: list[_FlattenedEdgeSortNode | list[_ComparisonEdge]] = [root]

    while stack:
        item = stack.pop()
        if isinstance(item, list):
            sorted_edges.extend(item)
            continue

        node = item
        if node.middle is None:
            sorted_edges.extend(node.edges)
            continue

        if node.greater_child is not None:
            stack.append(node.greater_child)
        stack.append(node.middle)
        if node.less_child is not None:
            stack.append(node.less_child)

    return sorted_edges


async def _ordered_nearest_edges_async(
    edges: Sequence[_ComparisonEdge],
    compare_batch_async: EdgeComparisonBatch,
    *,
    nearest_edge_strategy: NearestEdgeStrategy,
    rng: random.Random,
    batch_size: int,
) -> list[_ComparisonEdge]:
    if nearest_edge_strategy not in VALID_NEAREST_EDGE_STRATEGIES:
        raise ValueError(f"Unknown nearest_edge_strategy: {nearest_edge_strategy!r}.")

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

    if getattr(comparator, "prefer_guard_proximity_counts", False):
        fast_result = await _filter_candidates_by_preferred_guard_counts_async(
            candidates,
            sample1,
            guards,
            comparator,
        )
        if fast_result is not None:
            return fast_result
        if _has_index_batch_bool_comparator(comparator):
            return await _filter_candidates_by_flattened_guard_index_comparisons_async(
                candidates,
                sample1,
                guards,
                comparator,
                batch_size=batch_size,
            )
        if _has_batch_bool_comparator(comparator):
            return await _filter_candidates_by_flattened_guard_comparisons_async(
                candidates,
                sample1,
                guards,
                comparator,
                batch_size=batch_size,
            )

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

    if _has_index_batch_bool_comparator(comparator):
        return await _filter_candidates_by_flattened_guard_index_comparisons_async(
            candidates,
            sample1,
            guards,
            comparator,
            batch_size=batch_size,
        )

    return await _filter_candidates_by_flattened_guard_comparisons_async(
        candidates,
        sample1,
        guards,
        comparator,
        batch_size=batch_size,
    )


def _has_batch_bool_comparator(comparator: DistanceComparator) -> bool:
    return callable(getattr(comparator, "compare_batch_bool_async", None)) or callable(
        getattr(comparator, "compare_batch_bool", None)
    )


def _has_index_batch_bool_comparator(comparator: DistanceComparator) -> bool:
    return callable(getattr(comparator, "comparison_index_for_text", None)) and (
        callable(getattr(comparator, "compare_index_array_bool_async", None))
        or callable(getattr(comparator, "compare_index_array_bool", None))
        or callable(getattr(comparator, "compare_index_batch_bool_async", None))
        or callable(getattr(comparator, "compare_index_batch_bool", None))
    )


def _has_index_grouped_majority_comparator(comparator: DistanceComparator) -> bool:
    return callable(getattr(comparator, "comparison_index_for_text", None)) and (
        callable(getattr(comparator, "compare_index_grouped_majority_bool_async", None))
        or callable(getattr(comparator, "compare_index_grouped_majority_bool", None))
    )


async def _compare_index_grouped_majority_bool_async(
    comparator: DistanceComparator,
    *,
    ab_left_indexes: Any,
    ab_right_indexes: Any,
    cd_left_indexes: Any,
    cd_right_indexes: Any,
    offsets: Any,
    invert_majority: Any,
) -> Any:
    compare_async = getattr(
        comparator,
        "compare_index_grouped_majority_bool_async",
        None,
    )
    if compare_async is not None:
        return await compare_async(
            ab_left_indexes=ab_left_indexes,
            ab_right_indexes=ab_right_indexes,
            cd_left_indexes=cd_left_indexes,
            cd_right_indexes=cd_right_indexes,
            offsets=offsets,
            invert_majority=invert_majority,
        )

    compare = getattr(comparator, "compare_index_grouped_majority_bool", None)
    if compare is not None:
        return compare(
            ab_left_indexes=ab_left_indexes,
            ab_right_indexes=ab_right_indexes,
            cd_left_indexes=cd_left_indexes,
            cd_right_indexes=cd_right_indexes,
            offsets=offsets,
            invert_majority=invert_majority,
        )

    raise RuntimeError("Comparator does not support indexed grouped comparisons.")


def _index_array_to_comparison_list(comparisons: Any) -> list[IndexComparisonInput]:
    return [
        (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in comparisons
    ]


async def _compare_index_array_bool_async(
    comparator: DistanceComparator,
    comparisons: Any,
    *,
    batch_size: int = DEFAULT_ORACLE_COMPARISON_BATCH_SIZE,
) -> Any:
    if len(comparisons) == 0:
        return []
    if len(comparisons) > batch_size:
        import numpy as np

        answer_chunks = []
        for chunk in _chunks(comparisons, batch_size):
            answer_chunks.append(
                await _compare_index_array_bool_raw_async(comparator, chunk)
            )
        return np.concatenate(
            [np.asarray(chunk, dtype=bool) for chunk in answer_chunks],
        )

    return await _compare_index_array_bool_raw_async(comparator, comparisons)


async def _compare_index_array_bool_raw_async(
    comparator: DistanceComparator,
    comparisons: Any,
) -> Any:
    if len(comparisons) == 0:
        return []

    compare_index_array_bool_async = getattr(
        comparator,
        "compare_index_array_bool_async",
        None,
    )
    if compare_index_array_bool_async is not None:
        return await compare_index_array_bool_async(comparisons)

    compare_index_array_bool = getattr(comparator, "compare_index_array_bool", None)
    if compare_index_array_bool is not None:
        return compare_index_array_bool(comparisons)

    comparison_list = _index_array_to_comparison_list(comparisons)
    compare_index_batch_bool_async = getattr(
        comparator,
        "compare_index_batch_bool_async",
        None,
    )
    if compare_index_batch_bool_async is not None:
        return await compare_index_batch_bool_async(comparison_list)

    compare_index_batch_bool = getattr(comparator, "compare_index_batch_bool", None)
    if compare_index_batch_bool is not None:
        return compare_index_batch_bool(comparison_list)

    raise RuntimeError("Comparator does not support indexed batch comparisons.")


async def _filter_candidates_by_flattened_guard_index_comparisons_async(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    comparator: DistanceComparator,
    *,
    batch_size: int,
) -> list[TextRow]:
    import numpy as np

    candidate_ids = tuple(row.id for row in candidates)
    active_samples = tuple(
        (sample, guards.get(sample.id, ()))
        for sample in sample1
        if guards.get(sample.id)
    )
    if not active_samples:
        return _filter_candidates_by_proximity(
            candidates,
            {row_id: 0 for row_id in candidate_ids},
        )

    index_by_text: dict[str, int] = {}
    index_for_text = getattr(comparator, "comparison_index_for_text")

    def row_index(row: TextRow) -> int:
        try:
            return index_by_text[row.text]
        except KeyError:
            index = int(index_for_text(row.text))
            index_by_text[row.text] = index
            return index

    candidate_index_values = np.fromiter(
        (row_index(candidate) for candidate in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    indexed_active_samples = tuple(
        (
            sample_index,
            row_index(sample),
            np.fromiter(
                (row_index(guard) for guard in guard_rows),
                dtype=np.int64,
                count=len(guard_rows),
            ),
        )
        for sample_index, (sample, guard_rows) in enumerate(active_samples)
    )
    counts = np.zeros((len(active_samples), len(candidates)), dtype=np.int32)
    max_batch_size = max(1, batch_size)
    comparisons = np.empty((max_batch_size, 4), dtype=np.int64)
    comparison_sample_indexes = np.empty(max_batch_size, dtype=np.intp)
    comparison_candidate_indexes = np.empty(max_batch_size, dtype=np.intp)
    comparison_count = 0

    async def flush() -> None:
        nonlocal comparison_count

        if comparison_count == 0:
            return

        answers = await _compare_index_array_bool_async(
            comparator,
            comparisons[:comparison_count],
            batch_size=batch_size,
        )
        answer_array = np.asarray(answers, dtype=bool)
        if len(answer_array) != comparison_count:
            raise RuntimeError(
                "Indexed comparator returned an unexpected result count."
            )

        if np.any(answer_array):
            np.add.at(
                counts,
                (
                    comparison_sample_indexes[:comparison_count][answer_array],
                    comparison_candidate_indexes[:comparison_count][answer_array],
                ),
                1,
            )

        comparison_count = 0

    for sample_index, sample_index_value, guard_index_values in indexed_active_samples:
        for candidate_index, candidate_index_value in enumerate(candidate_index_values):
            guard_offset = 0
            while guard_offset < len(guard_index_values):
                remaining_capacity = max_batch_size - comparison_count
                if remaining_capacity == 0:
                    await flush()
                    remaining_capacity = max_batch_size

                copy_count = min(
                    remaining_capacity,
                    len(guard_index_values) - guard_offset,
                )
                target = slice(comparison_count, comparison_count + copy_count)
                source = slice(guard_offset, guard_offset + copy_count)
                comparisons[target, 0] = sample_index_value
                comparisons[target, 1] = int(candidate_index_value)
                comparisons[target, 2] = sample_index_value
                comparisons[target, 3] = guard_index_values[source]
                comparison_sample_indexes[target] = sample_index
                comparison_candidate_indexes[target] = candidate_index
                comparison_count += copy_count
                guard_offset += copy_count

    await flush()
    prox_max = counts.max(axis=0)
    return _filter_candidates_by_proximity(
        candidates,
        {
            row_id: int(count)
            for row_id, count in zip(candidate_ids, prox_max)
        },
    )


async def _filter_candidates_by_flattened_guard_comparisons_async(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    comparator: DistanceComparator,
    *,
    batch_size: int,
) -> list[TextRow]:
    import numpy as np

    candidate_ids = tuple(row.id for row in candidates)
    active_samples = tuple(
        (sample, guards.get(sample.id, ()))
        for sample in sample1
        if guards.get(sample.id)
    )
    if not active_samples:
        return _filter_candidates_by_proximity(
            candidates,
            {row_id: 0 for row_id in candidate_ids},
        )

    counts = np.zeros((len(active_samples), len(candidates)), dtype=np.int32)
    comparisons: list[ComparisonInput] = []
    comparison_sample_indexes: list[int] = []
    comparison_candidate_indexes: list[int] = []

    async def flush() -> None:
        if not comparisons:
            return

        answers = await _compare_batch_bool_async(
            comparator,
            comparisons,
            batch_size=batch_size,
        )
        for sample_index, candidate_index, answer in zip(
            comparison_sample_indexes,
            comparison_candidate_indexes,
            answers,
        ):
            if answer:
                counts[sample_index, candidate_index] += 1

        comparisons.clear()
        comparison_sample_indexes.clear()
        comparison_candidate_indexes.clear()

    for sample_index, (sample, guard_rows) in enumerate(active_samples):
        for candidate_index, candidate in enumerate(candidates):
            for guard in guard_rows:
                comparisons.append(
                    (sample.text, candidate.text, sample.text, guard.text)
                )
                comparison_sample_indexes.append(sample_index)
                comparison_candidate_indexes.append(candidate_index)
                if len(comparisons) >= batch_size:
                    await flush()

    await flush()
    prox_max = counts.max(axis=0)
    return _filter_candidates_by_proximity(
        candidates,
        {
            row_id: int(count)
            for row_id, count in zip(candidate_ids, prox_max)
        },
    )


async def _filter_candidates_by_preferred_guard_counts_async(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    comparator: DistanceComparator,
) -> list[TextRow] | None:
    guard_proximity_max_counts = getattr(
        comparator,
        "guard_proximity_max_counts",
        None,
    )
    if callable(guard_proximity_max_counts):
        return _filter_candidates_by_guard_proximity_max_counts(
            candidates,
            sample1,
            guards,
            guard_proximity_max_counts,
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

    return None


def _filter_candidates_by_guard_proximity_max_counts(
    candidates: Sequence[TextRow],
    sample1: Sequence[TextRow],
    guards: Mapping[str, tuple[TextRow, ...]],
    guard_proximity_max_counts: Callable[
        [Sequence[str], Sequence[str], Sequence[Sequence[str]]],
        Sequence[int],
    ],
) -> list[TextRow]:
    candidate_ids = tuple(row.id for row in candidates)
    active_samples = tuple(
        (sample, guards.get(sample.id, ()))
        for sample in sample1
        if guards.get(sample.id)
    )
    if not active_samples:
        return _filter_candidates_by_proximity(
            candidates,
            {row_id: 0 for row_id in candidate_ids},
        )

    counts = guard_proximity_max_counts(
        tuple(sample.text for sample, _ in active_samples),
        tuple(candidate.text for candidate in candidates),
        tuple(
            tuple(guard.text for guard in guard_rows)
            for _, guard_rows in active_samples
        ),
    )
    if len(counts) != len(candidate_ids):
        raise RuntimeError(
            "guard_proximity_max_counts returned an unexpected number of counts."
        )

    return _filter_candidates_by_proximity(
        candidates,
        {
            row_id: int(count)
            for row_id, count in zip(candidate_ids, counts)
        },
    )


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
