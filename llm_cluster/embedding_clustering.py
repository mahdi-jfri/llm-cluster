from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from llm_cluster.clustering import TextCluster
from llm_cluster.data import TextRow

DEFAULT_INSTRUCTOR_MODEL_NAME = "hkunlp/instructor-large"
CLINC_INTENT_INSTRUCTOR_PROMPT = "Represent utterances for intent classification: "
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
