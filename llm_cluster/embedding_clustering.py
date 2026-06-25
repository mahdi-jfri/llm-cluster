from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llm_cluster.clustering import TextCluster
from llm_cluster.comparison import (
    ComparisonInput,
    ComparisonProgressCallback,
    ComparisonResult,
)
from llm_cluster.data import TextRow

DEFAULT_INSTRUCTOR_MODEL_NAME = "hkunlp/instructor-large"
CLINC_INTENT_INSTRUCTOR_PROMPT = "Represent utterances for intent classification: "
DBPEDIA_ONTOLOGY_INSTRUCTOR_PROMPT = (
    "Represent Wikipedia articles for ontology classification: "
)
DEFAULT_KMEANS_INIT = "k-means++"


@dataclass(frozen=True)
class EmbeddingClusteringResult:
    target_clusters: int
    model_name: str
    prompt: str
    seed: int | None
    batch_size: int
    normalize_embeddings: bool
    device: str | None
    kmeans_init: str
    kmeans_n_init: int
    kmeans_max_iter: int
    embedding_shape: tuple[int, int]
    inertia: float
    centers: tuple[TextRow, ...]
    clusters: tuple[TextCluster, ...]
    assignments: Mapping[str, TextRow]


@dataclass
class EmbeddingDistanceComparator:
    """Embedding-backed comparator for deciding whether d(a, b) < d(c, d)."""

    rows: Sequence[TextRow]
    model_name: str = DEFAULT_INSTRUCTOR_MODEL_NAME
    prompt: str = CLINC_INTENT_INSTRUCTOR_PROMPT
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None
    show_progress_bar: bool = False
    progress_callback: ComparisonProgressCallback | None = None
    n_source_rows: int = field(init=False)
    n_unique_texts: int = field(init=False)
    embedding_shape: tuple[int, int] = field(init=False)
    _text_to_index: dict[str, int] = field(init=False, repr=False)
    _embeddings: Any = field(init=False, repr=False)
    _embedding_norms: Any = field(init=False, repr=False)
    _distance_matrix: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        source_rows = list(self.rows)
        if not source_rows:
            raise ValueError("rows must contain at least one item.")

        unique_rows = _unique_rows_by_text(source_rows)
        model = _load_instructor_model(self.model_name, device=self.device)
        embeddings = _encode_instructor_embeddings(
            model,
            unique_rows,
            prompt=self.prompt,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
        )

        import numpy as np

        embeddings = np.ascontiguousarray(embeddings)

        self.n_source_rows = len(source_rows)
        self.n_unique_texts = len(unique_rows)
        self.embedding_shape = (int(embeddings.shape[0]), int(embeddings.shape[1]))
        self._text_to_index = {
            row.text: index for index, row in enumerate(unique_rows)
        }
        self._embeddings = embeddings
        self._embedding_norms = np.einsum("ij,ij->i", embeddings, embeddings)
        self._distance_matrix = _squared_l2_distance_matrix(
            embeddings,
            self._embedding_norms,
        )

    def compare(self, a: str, b: str, c: str, d: str) -> ComparisonResult:
        return self.compare_batch([(a, b, c, d)])[0]

    def compare_batch(
        self,
        comparisons: list[ComparisonInput],
    ) -> list[ComparisonResult]:
        if not comparisons:
            return []

        ab_distances, cd_distances = self._comparison_distances(comparisons)

        results = [
            ComparisonResult(
                reasoning=(
                    "embedding_sq_l2_ab="
                    f"{float(ab_distance):.6g}; embedding_sq_l2_cd="
                    f"{float(cd_distance):.6g}"
                ),
                is_ab_less_than_cd=bool(ab_distance < cd_distance),
            )
            for ab_distance, cd_distance in zip(ab_distances, cd_distances)
        ]
        self._notify_progress(len(results))
        return results

    def compare_batch_bool(
        self,
        comparisons: list[ComparisonInput],
    ) -> Any:
        """Return boolean comparison answers without allocating result objects."""

        if not comparisons:
            import numpy as np

            return np.empty(0, dtype=bool)

        ab_distances, cd_distances = self._comparison_distances(comparisons)
        self._notify_progress(len(comparisons))
        return ab_distances < cd_distances

    def pair_distances(
        self,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
    ) -> Any:
        """Return squared embedding distances for aligned text pairs."""

        if len(left_texts) != len(right_texts):
            raise ValueError("left_texts and right_texts must have the same length.")

        left_indexes = self._indexes_for_texts(left_texts)
        right_indexes = self._indexes_for_texts(right_texts)
        distances = self._pair_distances_by_index(left_indexes, right_indexes)
        self._notify_progress(len(left_indexes))
        return distances

    def edge_distance_matrix(
        self,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
    ) -> Any:
        """Return the full squared-distance matrix for two text sets."""

        left_indexes = self._indexes_for_texts(left_texts)
        right_indexes = self._indexes_for_texts(right_texts)
        distances = self._distance_matrix_by_index(left_indexes, right_indexes)
        self._notify_progress(len(left_indexes) * len(right_indexes))
        return distances

    def guard_proximity_counts(
        self,
        sample_text: str,
        candidate_texts: Sequence[str],
        guard_texts: Sequence[str],
    ) -> list[int]:
        """Count guards farther from a sample than each candidate.

        This is equivalent to evaluating
        `(sample, candidate, sample, guard)` for every candidate/guard pair
        and summing the true comparison results by candidate.
        """

        if not candidate_texts:
            return []
        if not guard_texts:
            return [0] * len(candidate_texts)

        import numpy as np

        candidate_distances = self.edge_distance_matrix(
            (sample_text,),
            candidate_texts,
        )[0]
        guard_distances = self.edge_distance_matrix(
            (sample_text,),
            guard_texts,
        )[0]
        counts = np.sum(
            candidate_distances[:, None] < guard_distances[None, :],
            axis=1,
        )

        return [int(count) for count in counts]

    async def compare_batch_async(
        self,
        comparisons: list[ComparisonInput],
    ) -> list[ComparisonResult]:
        return self.compare_batch(comparisons)

    def _index_for_text(self, text: str) -> int:
        try:
            return self._text_to_index[text]
        except KeyError as exc:
            preview = text.replace("\n", " ")[:80]
            raise KeyError(
                "No embedding is available for comparison text: "
                f"{preview!r}. Build the comparator with every row that can "
                "appear in comparisons."
            ) from exc

    def _indexes_for_texts(self, texts: Sequence[str]) -> Any:
        import numpy as np

        return np.asarray(
            [self._index_for_text(text) for text in texts],
            dtype=np.int64,
        )

    def _comparison_distances(
        self,
        comparisons: list[ComparisonInput],
    ) -> tuple[Any, Any]:
        import numpy as np

        indexes = np.asarray(
            [
                [self._index_for_text(text) for text in comparison]
                for comparison in comparisons
            ],
            dtype=np.int64,
        )
        ab_distances = self._pair_distances_by_index(indexes[:, 0], indexes[:, 1])
        cd_distances = self._pair_distances_by_index(indexes[:, 2], indexes[:, 3])
        return ab_distances, cd_distances

    def _pair_distances_by_index(self, left_indexes: Any, right_indexes: Any) -> Any:
        import numpy as np

        if len(left_indexes) == 0:
            return np.empty(0, dtype=self._embeddings.dtype)

        return self._distance_matrix[left_indexes, right_indexes]

    def _distance_matrix_by_index(self, left_indexes: Any, right_indexes: Any) -> Any:
        import numpy as np

        left_count = len(left_indexes)
        right_count = len(right_indexes)
        if left_count == 0 or right_count == 0:
            return np.empty(
                (left_count, right_count),
                dtype=self._embeddings.dtype,
            )

        return self._distance_matrix[np.ix_(left_indexes, right_indexes)]

    def _notify_progress(self, count: int) -> None:
        if self.progress_callback is None:
            return
        add_progress = getattr(self.progress_callback, "add", None)
        if add_progress is not None:
            add_progress("computed", count)
            return
        for _ in range(count):
            self.progress_callback("computed")


def embedding_kmeans_cluster(
    rows: Sequence[TextRow],
    *,
    k: int,
    model_name: str = DEFAULT_INSTRUCTOR_MODEL_NAME,
    prompt: str = CLINC_INTENT_INSTRUCTOR_PROMPT,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    seed: int | None = None,
    device: str | None = None,
    kmeans_init: str = DEFAULT_KMEANS_INIT,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    show_progress_bar: bool = False,
) -> EmbeddingClusteringResult:
    """Cluster text rows by INSTRUCTOR embeddings and KMeans.

    Centers are represented by the row nearest to each learned centroid, so the
    result can reuse the same `TextCluster` representation as the comparison
    oracle clustering path.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if kmeans_n_init < 1:
        raise ValueError("kmeans_n_init must be at least 1.")
    if kmeans_max_iter < 1:
        raise ValueError("kmeans_max_iter must be at least 1.")

    all_rows = list(rows)
    if not all_rows:
        raise ValueError("rows must contain at least one item.")
    if k > len(all_rows):
        raise ValueError(f"k must be <= number of rows ({len(all_rows)}).")

    model = _load_instructor_model(model_name, device=device)
    embeddings = _encode_instructor_embeddings(
        model,
        all_rows,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
    )
    labels, centroids, inertia = _fit_kmeans(
        embeddings,
        k=k,
        seed=seed,
        init=kmeans_init,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
    )
    center_by_label = _nearest_rows_to_centroids(
        all_rows,
        embeddings,
        labels,
        centroids,
    )
    assignments = {
        row.id: center_by_label[int(label)] for row, label in zip(all_rows, labels)
    }
    clusters = _build_clusters_from_labels(all_rows, labels, center_by_label)

    return EmbeddingClusteringResult(
        target_clusters=k,
        model_name=model_name,
        prompt=prompt,
        seed=seed,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        kmeans_init=kmeans_init,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        embedding_shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
        inertia=inertia,
        centers=tuple(cluster.center for cluster in clusters),
        clusters=clusters,
        assignments=assignments,
    )


def _unique_rows_by_text(rows: Sequence[TextRow]) -> list[TextRow]:
    rows_by_text: dict[str, TextRow] = {}
    for row in rows:
        rows_by_text.setdefault(row.text, row)
    return list(rows_by_text.values())


def _squared_l2_distance_matrix(embeddings: Any, norms: Any) -> Any:
    import numpy as np

    distances = norms[:, None] + norms[None, :] - 2.0 * (embeddings @ embeddings.T)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return np.ascontiguousarray(distances)


def _load_instructor_model(model_name: str, *, device: str | None) -> Any:
    try:
        from InstructorEmbedding import INSTRUCTOR
    except ImportError as exc:
        raise RuntimeError(
            "Embedding clustering requires InstructorEmbedding. Install the "
            "project dependencies with `.venv/bin/python -m pip install -e .`."
        ) from exc

    if device is None:
        model = INSTRUCTOR(model_name)
    else:
        model = INSTRUCTOR(model_name, device=device)
    _patch_instructor_sentence_transformers_compatibility(model)
    return model


def _patch_instructor_sentence_transformers_compatibility(model: Any) -> None:
    if not hasattr(model, "_text_length") and hasattr(model, "_input_length"):
        model._text_length = model._input_length


def _encode_instructor_embeddings(
    model: Any,
    rows: Sequence[TextRow],
    *,
    prompt: str,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress_bar: bool,
) -> Any:
    import numpy as np

    inputs = [[prompt, row.text] for row in rows]
    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": True,
        "normalize_embeddings": normalize_embeddings,
    }
    try:
        embeddings = model.encode(inputs, **encode_kwargs)
    except TypeError as exc:
        if "normalize_embeddings" not in str(exc):
            raise
        encode_kwargs.pop("normalize_embeddings")
        embeddings = model.encode(inputs, **encode_kwargs)

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2D embedding array, got shape {array.shape}.")
    if array.shape[0] != len(rows):
        raise RuntimeError(
            "Embedding model returned a different number of vectors than inputs: "
            f"{array.shape[0]} != {len(rows)}."
        )
    if normalize_embeddings:
        array = _l2_normalize(array)
    return array


def _fit_kmeans(
    embeddings: Any,
    *,
    k: int,
    seed: int | None,
    init: str,
    n_init: int,
    max_iter: int,
) -> tuple[Any, Any, float]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "Embedding clustering requires scikit-learn. Install the embedding "
            "dependencies with `.venv/bin/python -m pip install -e .`."
        ) from exc

    estimator = KMeans(
        n_clusters=k,
        init=init,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = estimator.fit_predict(embeddings)
    return labels, estimator.cluster_centers_, float(estimator.inertia_)


def _nearest_rows_to_centroids(
    rows: Sequence[TextRow],
    embeddings: Any,
    labels: Any,
    centroids: Any,
) -> dict[int, TextRow]:
    import numpy as np

    labels_array = np.asarray(labels)
    center_by_label: dict[int, TextRow] = {}
    for label in sorted(int(value) for value in set(labels_array.tolist())):
        indexes = np.flatnonzero(labels_array == label)
        distances = np.sum((embeddings[indexes] - centroids[label]) ** 2, axis=1)
        best_index = int(indexes[int(np.argmin(distances))])
        center_by_label[label] = rows[best_index]
    return center_by_label


def _build_clusters_from_labels(
    rows: Sequence[TextRow],
    labels: Any,
    center_by_label: Mapping[int, TextRow],
) -> tuple[TextCluster, ...]:
    rows_by_label: dict[int, list[TextRow]] = {}
    for row, label in zip(rows, labels):
        rows_by_label.setdefault(int(label), []).append(row)

    clusters = [
        TextCluster(center=center_by_label[label], rows=tuple(label_rows))
        for label, label_rows in rows_by_label.items()
    ]
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (-len(cluster.rows), cluster.center.id),
        )
    )


def _l2_normalize(embeddings: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms
