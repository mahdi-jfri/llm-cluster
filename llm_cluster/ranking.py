from __future__ import annotations

import asyncio
from dataclasses import dataclass
import random
from typing import Protocol, Sequence

from llm_cluster.comparison import ComparisonResult
from llm_cluster.data import TextRow


class DistanceComparator(Protocol):
    def compare(self, a: str, b: str, c: str, d: str) -> ComparisonResult:
        """Return whether d(a, b) < d(c, d)."""

    def compare_batch(
        self,
        comparisons: list[tuple[str, str, str, str]],
    ) -> list[ComparisonResult]:
        """Return comparison results in the same order as inputs."""

    async def compare_batch_async(
        self,
        comparisons: list[tuple[str, str, str, str]],
    ) -> list[ComparisonResult]:
        """Return comparison results asynchronously in input order."""


@dataclass(frozen=True)
class RankingMetrics:
    n_items: int
    n_in_cluster: int
    n_out_cluster: int
    inversions: int
    max_inversions: int
    inversion_rate: float
    in_cluster_auc: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "n_items": self.n_items,
            "n_in_cluster": self.n_in_cluster,
            "n_out_cluster": self.n_out_cluster,
            "inversions": self.inversions,
            "max_inversions": self.max_inversions,
            "inversion_rate": self.inversion_rate,
            "in_cluster_auc": self.in_cluster_auc,
        }


def sort_by_distance(
    anchor: TextRow,
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    seed: int | None = None,
) -> list[TextRow]:
    """Sort rows from nearest to farthest with randomized quicksort."""

    if hasattr(comparator, "compare_batch_async"):
        return asyncio.run(sort_by_distance_async(anchor, rows, comparator, seed=seed))

    candidates = [row for row in rows if row.id != anchor.id]
    rng = random.Random(seed)
    return _quicksort_by_distance(anchor, candidates, comparator, rng)


async def sort_by_distance_async(
    anchor: TextRow,
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    *,
    seed: int | None = None,
) -> list[TextRow]:
    candidates = [row for row in rows if row.id != anchor.id]
    rng = random.Random(seed)
    return await _quicksort_by_distance_async(anchor, candidates, comparator, rng)


def _quicksort_by_distance(
    anchor: TextRow,
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    rng: random.Random,
) -> list[TextRow]:
    if len(rows) <= 1:
        return list(rows)

    pivot_index = rng.randrange(len(rows))
    pivot = rows[pivot_index]
    left: list[TextRow] = []
    right: list[TextRow] = []
    partition_rows: list[TextRow] = []
    comparison_inputs: list[tuple[str, str, str, str]] = []

    for index, row in enumerate(rows):
        if index == pivot_index:
            continue
        partition_rows.append(row)
        comparison_inputs.append((anchor.text, row.text, anchor.text, pivot.text))

    results = comparator.compare_batch(comparison_inputs)
    for row, result in zip(partition_rows, results):
        if result.is_ab_less_than_cd:
            left.append(row)
        else:
            right.append(row)

    return [
        *_quicksort_by_distance(anchor, left, comparator, rng),
        pivot,
        *_quicksort_by_distance(anchor, right, comparator, rng),
    ]


async def _quicksort_by_distance_async(
    anchor: TextRow,
    rows: Sequence[TextRow],
    comparator: DistanceComparator,
    rng: random.Random,
) -> list[TextRow]:
    if len(rows) <= 1:
        return list(rows)

    pivot_index = rng.randrange(len(rows))
    pivot = rows[pivot_index]
    left: list[TextRow] = []
    right: list[TextRow] = []
    partition_rows: list[TextRow] = []
    comparison_inputs: list[tuple[str, str, str, str]] = []

    for index, row in enumerate(rows):
        if index == pivot_index:
            continue
        partition_rows.append(row)
        comparison_inputs.append((anchor.text, row.text, anchor.text, pivot.text))

    results = await comparator.compare_batch_async(comparison_inputs)
    for row, result in zip(partition_rows, results):
        if result.is_ab_less_than_cd:
            left.append(row)
        else:
            right.append(row)

    left_sorted = await _quicksort_by_distance_async(anchor, left, comparator, rng)
    right_sorted = await _quicksort_by_distance_async(anchor, right, comparator, rng)
    return [*left_sorted, pivot, *right_sorted]


def evaluate_in_cluster_ranking(anchor: TextRow, ranked_rows: Sequence[TextRow]) -> RankingMetrics:
    """Evaluate a ranking by counting out-of-cluster-before-in-cluster inversions."""

    relevance = [
        1 if row.label_name == anchor.label_name else 0
        for row in ranked_rows
        if row.id != anchor.id
    ]
    return inversion_metrics(relevance)


def inversion_metrics(binary_relevance: Sequence[int]) -> RankingMetrics:
    """Count inversions where an out-cluster row appears before an in-cluster row."""

    inversions = 0
    negatives_seen = 0
    positives = 0
    negatives = 0

    for value in binary_relevance:
        if value not in {0, 1}:
            raise ValueError("binary_relevance must contain only 0/1 values.")

        if value == 1:
            positives += 1
            inversions += negatives_seen
        else:
            negatives += 1
            negatives_seen += 1

    max_inversions = positives * negatives
    inversion_rate = inversions / max_inversions if max_inversions else 0.0
    in_cluster_auc = 1.0 - inversion_rate if max_inversions else 1.0

    return RankingMetrics(
        n_items=len(binary_relevance),
        n_in_cluster=positives,
        n_out_cluster=negatives,
        inversions=inversions,
        max_inversions=max_inversions,
        inversion_rate=inversion_rate,
        in_cluster_auc=in_cluster_auc,
    )
