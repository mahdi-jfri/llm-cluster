from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from llm_cluster.clustering import TextCluster


@dataclass(frozen=True)
class ClusteringMetrics:
    n_items: int
    n_clusters: int
    n_labels: int
    purity: float
    inverse_purity: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    rand_index: float
    adjusted_rand_index: float
    mutual_information: float
    normalized_mutual_information: float
    homogeneity: float
    completeness: float
    v_measure: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "n_items": self.n_items,
            "n_clusters": self.n_clusters,
            "n_labels": self.n_labels,
            "purity": self.purity,
            "inverse_purity": self.inverse_purity,
            "pairwise_precision": self.pairwise_precision,
            "pairwise_recall": self.pairwise_recall,
            "pairwise_f1": self.pairwise_f1,
            "rand_index": self.rand_index,
            "adjusted_rand_index": self.adjusted_rand_index,
            "mutual_information": self.mutual_information,
            "normalized_mutual_information": self.normalized_mutual_information,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "v_measure": self.v_measure,
        }


def evaluate_clustering(clusters: Sequence[TextCluster]) -> ClusteringMetrics:
    """Evaluate predicted clusters against each row's gold label_name."""

    contingency: list[Counter[str]] = []
    label_counts: Counter[str] = Counter()
    seen_row_ids: set[str] = set()

    for cluster in clusters:
        cluster_label_counts: Counter[str] = Counter()
        for row in cluster.rows:
            if row.id in seen_row_ids:
                raise ValueError(f"Row {row.id!r} appears in more than one cluster.")
            seen_row_ids.add(row.id)
            cluster_label_counts[row.label_name] += 1
            label_counts[row.label_name] += 1
        if cluster_label_counts:
            contingency.append(cluster_label_counts)

    n_items = sum(label_counts.values())
    if n_items == 0:
        raise ValueError("clusters must contain at least one row.")

    cluster_sizes = [sum(cluster_counts.values()) for cluster_counts in contingency]
    n_clusters = len(cluster_sizes)
    n_labels = len(label_counts)

    majority_label_count = sum(
        max(cluster_counts.values()) for cluster_counts in contingency
    )
    purity = majority_label_count / n_items
    inverse_purity = _inverse_purity(contingency, label_counts, n_items)

    true_positive_pairs = sum(
        _comb2(count)
        for cluster_counts in contingency
        for count in cluster_counts.values()
    )
    predicted_positive_pairs = sum(_comb2(size) for size in cluster_sizes)
    gold_positive_pairs = sum(_comb2(count) for count in label_counts.values())
    total_pairs = _comb2(n_items)
    false_positive_pairs = predicted_positive_pairs - true_positive_pairs
    false_negative_pairs = gold_positive_pairs - true_positive_pairs
    true_negative_pairs = (
        total_pairs - true_positive_pairs - false_positive_pairs - false_negative_pairs
    )

    pairwise_precision = (
        true_positive_pairs / predicted_positive_pairs
        if predicted_positive_pairs
        else 1.0
    )
    pairwise_recall = (
        true_positive_pairs / gold_positive_pairs if gold_positive_pairs else 1.0
    )
    pairwise_f1 = _harmonic_mean(pairwise_precision, pairwise_recall)
    rand_index = (
        (true_positive_pairs + true_negative_pairs) / total_pairs
        if total_pairs
        else 1.0
    )
    adjusted_rand_index = _adjusted_rand_index(
        true_positive_pairs=true_positive_pairs,
        predicted_positive_pairs=predicted_positive_pairs,
        gold_positive_pairs=gold_positive_pairs,
        total_pairs=total_pairs,
    )

    label_entropy = _entropy(label_counts.values(), n_items)
    cluster_entropy = _entropy(cluster_sizes, n_items)
    mutual_information = _mutual_information(contingency, label_counts, n_items)
    normalized_mutual_information = (
        2.0 * mutual_information / (label_entropy + cluster_entropy)
        if label_entropy + cluster_entropy
        else 1.0
    )
    homogeneity = mutual_information / label_entropy if label_entropy else 1.0
    completeness = mutual_information / cluster_entropy if cluster_entropy else 1.0
    v_measure = _harmonic_mean(homogeneity, completeness)

    return ClusteringMetrics(
        n_items=n_items,
        n_clusters=n_clusters,
        n_labels=n_labels,
        purity=_clamp01(purity),
        inverse_purity=_clamp01(inverse_purity),
        pairwise_precision=_clamp01(pairwise_precision),
        pairwise_recall=_clamp01(pairwise_recall),
        pairwise_f1=_clamp01(pairwise_f1),
        rand_index=_clamp01(rand_index),
        adjusted_rand_index=adjusted_rand_index,
        mutual_information=max(0.0, mutual_information),
        normalized_mutual_information=_clamp01(normalized_mutual_information),
        homogeneity=_clamp01(homogeneity),
        completeness=_clamp01(completeness),
        v_measure=_clamp01(v_measure),
    )


def _inverse_purity(
    contingency: Sequence[Counter[str]],
    label_counts: Counter[str],
    n_items: int,
) -> float:
    best_cluster_counts = (
        max(cluster_counts.get(label, 0) for cluster_counts in contingency)
        for label in label_counts
    )
    return sum(best_cluster_counts) / n_items


def _adjusted_rand_index(
    *,
    true_positive_pairs: int,
    predicted_positive_pairs: int,
    gold_positive_pairs: int,
    total_pairs: int,
) -> float:
    if total_pairs == 0:
        return 1.0

    expected_index = predicted_positive_pairs * gold_positive_pairs / total_pairs
    max_index = 0.5 * (predicted_positive_pairs + gold_positive_pairs)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    return (true_positive_pairs - expected_index) / denominator


def _mutual_information(
    contingency: Sequence[Counter[str]],
    label_counts: Counter[str],
    n_items: int,
) -> float:
    mutual_information = 0.0
    for cluster_counts in contingency:
        cluster_size = sum(cluster_counts.values())
        for label, joint_count in cluster_counts.items():
            mutual_information += (joint_count / n_items) * math.log(
                (joint_count * n_items) / (cluster_size * label_counts[label])
            )
    return mutual_information


def _entropy(counts: Iterable[int], n_items: int) -> float:
    entropy = 0.0
    for count in counts:
        probability = count / n_items
        entropy -= probability * math.log(probability)
    return entropy


def _comb2(value: int) -> int:
    return value * (value - 1) // 2


def _harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b) if a + b else 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
