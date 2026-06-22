from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from llm_cluster.comparison import ComparisonInput, ComparisonResult
from llm_cluster.data import TextRow
from llm_cluster.ranking import DistanceComparator


@dataclass(frozen=True)
class TextCluster:
    center: TextRow
    rows: tuple[TextRow, ...]

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class SuccessiveSamplingRound:
    index: int
    n_remaining_before: int
    sample_size: int
    sample_center_ids: tuple[str, ...]
    covered_count: int
    n_remaining_after: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "n_remaining_before": self.n_remaining_before,
            "sample_size": self.sample_size,
            "sample_center_ids": list(self.sample_center_ids),
            "covered_count": self.covered_count,
            "n_remaining_after": self.n_remaining_after,
        }


@dataclass(frozen=True)
class SuccessiveSamplingResult:
    target_clusters: int
    sample_size: int
    sample_multiplier: float
    cover_fraction: float
    seed: int | None
    final_center_count: int | None
    candidate_centers: tuple[TextRow, ...]
    candidate_clusters: tuple[TextCluster, ...]
    centers: tuple[TextRow, ...]
    clusters: tuple[TextCluster, ...]
    assignments: Mapping[str, TextRow]
    candidate_assignments: Mapping[str, TextRow]
    rounds: tuple[SuccessiveSamplingRound, ...]

    @property
    def compressed(self) -> bool:
        return self.final_center_count is not None


@dataclass(frozen=True)
class _AssignedRow:
    row: TextRow
    center: TextRow


def successive_sampling_cluster(
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    k: int,
    sample_multiplier: float = 1.0,
    cover_fraction: float = 0.5,
    seed: int | None = None,
    final_center_count: int | None = None,
) -> SuccessiveSamplingResult:
    """Cluster rows with Mettu-Plaxton-style successive sampling.

    The algorithm repeatedly samples O(k) candidate centers from the active rows,
    assigns every active row to its nearest sampled center using only comparative
    distance queries, sorts active rows by those assigned distances, and removes
    the closest constant fraction. The raw result has O(k log(n/k)) candidate
    centers. Set ``final_center_count`` to reassign all rows to the largest
    sampled clusters' centers as a pragmatic exact-k compression step.
    """

    return asyncio.run(
        successive_sampling_cluster_async(
            rows,
            comparator,
            k=k,
            sample_multiplier=sample_multiplier,
            cover_fraction=cover_fraction,
            seed=seed,
            final_center_count=final_center_count,
        )
    )


async def successive_sampling_cluster_async(
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    k: int,
    sample_multiplier: float = 1.0,
    cover_fraction: float = 0.5,
    seed: int | None = None,
    final_center_count: int | None = None,
) -> SuccessiveSamplingResult:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if sample_multiplier <= 0:
        raise ValueError("sample_multiplier must be positive.")
    if not 0 < cover_fraction < 1:
        raise ValueError("cover_fraction must be between 0 and 1.")
    if final_center_count is not None and final_center_count < 1:
        raise ValueError("final_center_count must be at least 1 when provided.")

    all_rows = list(rows)
    if not all_rows:
        raise ValueError("rows must contain at least one item.")

    rng = random.Random(seed)
    sample_size = min(len(all_rows), max(1, math.ceil(k * sample_multiplier)))
    remaining = list(all_rows)
    candidate_assignments: dict[str, TextRow] = {}
    rounds: list[SuccessiveSamplingRound] = []

    while len(remaining) > sample_size:
        sample = rng.sample(remaining, sample_size)
        nearest_centers = await _nearest_centers_async(remaining, sample, comparator)
        assigned_rows = [
            _AssignedRow(row=row, center=center)
            for row, center in zip(remaining, nearest_centers)
        ]
        sorted_rows = await _sort_by_assigned_distance_async(
            assigned_rows,
            comparator,
            rng,
        )

        covered_count = min(
            len(sorted_rows),
            max(1, math.ceil(cover_fraction * len(sorted_rows))),
        )
        covered = sorted_rows[:covered_count]
        uncovered = sorted_rows[covered_count:]
        for assigned in covered:
            candidate_assignments[assigned.row.id] = assigned.center

        rounds.append(
            SuccessiveSamplingRound(
                index=len(rounds),
                n_remaining_before=len(remaining),
                sample_size=len(sample),
                sample_center_ids=tuple(row.id for row in sample),
                covered_count=len(covered),
                n_remaining_after=len(uncovered),
            )
        )
        remaining = [assigned.row for assigned in uncovered]

    for row in remaining:
        candidate_assignments[row.id] = row

    candidate_clusters = _build_clusters(all_rows, candidate_assignments)
    candidate_centers = tuple(cluster.center for cluster in candidate_clusters)

    if final_center_count is None:
        centers = candidate_centers
        assignments = dict(candidate_assignments)
        clusters = candidate_clusters
    else:
        center_count = min(final_center_count, len(candidate_centers))
        centers = _largest_cluster_centers(candidate_clusters, center_count)
        final_nearest = await _nearest_centers_async(all_rows, centers, comparator)
        assignments = {row.id: center for row, center in zip(all_rows, final_nearest)}
        clusters = _build_clusters(all_rows, assignments, center_order=centers)

    return SuccessiveSamplingResult(
        target_clusters=k,
        sample_size=sample_size,
        sample_multiplier=sample_multiplier,
        cover_fraction=cover_fraction,
        seed=seed,
        final_center_count=final_center_count,
        candidate_centers=candidate_centers,
        candidate_clusters=candidate_clusters,
        centers=centers,
        clusters=clusters,
        assignments=assignments,
        candidate_assignments=candidate_assignments,
        rounds=tuple(rounds),
    )


async def _nearest_centers_async(
    rows: Sequence[TextRow],
    centers: Sequence[TextRow],
    comparator: DistanceComparator,
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

        results = await _compare_batch_async(comparator, comparisons)
        for index, result in zip(comparison_indexes, results):
            if result.is_ab_less_than_cd:
                nearest[index] = candidate

    return nearest


async def _sort_by_assigned_distance_async(
    assigned_rows: Sequence[_AssignedRow],
    comparator: DistanceComparator,
    rng: random.Random,
) -> list[_AssignedRow]:
    if len(assigned_rows) <= 1:
        return list(assigned_rows)

    pivot_index = rng.randrange(len(assigned_rows))
    pivot = assigned_rows[pivot_index]
    pivot_zero = _is_zero_distance(pivot)
    left: list[_AssignedRow] = []
    right: list[_AssignedRow] = []
    pending: list[_AssignedRow] = []
    comparisons: list[ComparisonInput] = []

    for index, assigned in enumerate(assigned_rows):
        if index == pivot_index:
            continue

        assigned_zero = _is_zero_distance(assigned)
        if assigned_zero and not pivot_zero:
            left.append(assigned)
        elif pivot_zero:
            right.append(assigned)
        else:
            pending.append(assigned)
            comparisons.append(
                (
                    assigned.row.text,
                    assigned.center.text,
                    pivot.row.text,
                    pivot.center.text,
                )
            )

    results = await _compare_batch_async(comparator, comparisons)
    for assigned, result in zip(pending, results):
        if result.is_ab_less_than_cd:
            left.append(assigned)
        else:
            right.append(assigned)

    left_rng, right_rng = _branch_rngs(rng)
    left_sorted, right_sorted = await asyncio.gather(
        _sort_by_assigned_distance_async(left, comparator, left_rng),
        _sort_by_assigned_distance_async(right, comparator, right_rng),
    )
    return [*left_sorted, pivot, *right_sorted]


async def _compare_batch_async(
    comparator: DistanceComparator,
    comparisons: list[ComparisonInput],
) -> list[ComparisonResult]:
    if not comparisons:
        return []

    compare_batch_async = getattr(comparator, "compare_batch_async", None)
    if compare_batch_async is not None:
        return await compare_batch_async(comparisons)
    return comparator.compare_batch(comparisons)


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


def _largest_cluster_centers(
    clusters: Sequence[TextCluster],
    count: int,
) -> tuple[TextRow, ...]:
    return tuple(
        cluster.center
        for cluster in sorted(
            clusters,
            key=lambda cluster: (-len(cluster.rows), cluster.center.id),
        )[:count]
    )


def _is_zero_distance(assigned: _AssignedRow) -> bool:
    return assigned.row.id == assigned.center.id


def _branch_rngs(rng: random.Random) -> tuple[random.Random, random.Random]:
    return random.Random(rng.getrandbits(64)), random.Random(rng.getrandbits(64))
