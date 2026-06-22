from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

CLINC_DATASET_IDS = (
    "clinc/clinc_oos",
    "clinc_oos",
    "DeepPavlov/clinc_oos",
)
CLINC_DEFAULT_CONFIG = "plus"


@dataclass(frozen=True)
class TextRow:
    """One text item with a gold clustering label."""

    id: str
    text: str
    label: int | str
    label_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_dataset_rows(name: str, **kwargs: Any) -> list[TextRow]:
    """Load a named dataset into normalized text rows."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"clinc", "clinc150", "clinc_150", "clinc_oos"}:
        return load_clinc(**kwargs)

    raise ValueError(f"Unsupported dataset: {name!r}")


def load_clinc(
    *,
    split: str = "test",
    config: str | None = CLINC_DEFAULT_CONFIG,
    dataset_id: str | None = None,
    remove_oos: bool = True,
) -> list[TextRow]:
    """Load CLINC150 rows, optionally dropping out-of-scope examples.

    The default uses the CLINC OOS `plus` test split. After removing OOS, this
    should yield the 4,500 in-scope test queries used by prior clustering work.
    """

    remove_labels = {"oos", "out_of_scope", "out-of-scope"} if remove_oos else set()
    dataset_ids = (dataset_id,) if dataset_id else CLINC_DATASET_IDS
    errors: list[str] = []

    for candidate_id in dataset_ids:
        try:
            return load_hf_text_classification_dataset(
                dataset_id=candidate_id,
                config=config,
                split=split,
                text_field=None,
                label_field=None,
                remove_label_names=remove_labels,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - fallback path depends on HF state.
            errors.append(f"{candidate_id}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to load CLINC dataset.\n{joined_errors}")


def load_hf_text_classification_dataset(
    *,
    dataset_id: str,
    split: str,
    config: str | None = None,
    text_field: str | None = None,
    label_field: str | None = None,
    remove_label_names: Iterable[str] = (),
) -> list[TextRow]:
    """Load a Hugging Face text classification dataset as `TextRow` objects."""

    from datasets import load_dataset as hf_load_dataset

    raw_dataset = hf_load_dataset(dataset_id, config, split=split)
    features = raw_dataset.features

    resolved_text_field = text_field or _infer_field(
        features.keys(), ("text", "sentence", "query", "utterance"), "text"
    )
    resolved_label_field = label_field or _infer_field(
        features.keys(), ("intent", "label", "category"), "label"
    )

    label_feature = features[resolved_label_field]
    label_names = tuple(getattr(label_feature, "names", ()) or ())
    remove_labels = {_normalize_label_name(label) for label in remove_label_names}

    rows: list[TextRow] = []
    for index, item in enumerate(raw_dataset):
        text = str(item[resolved_text_field])
        label = item[resolved_label_field]
        label_name = _resolve_label_name(label, label_names)
        if _normalize_label_name(label_name) in remove_labels:
            continue

        metadata = {
            key: value
            for key, value in item.items()
            if key not in {resolved_text_field, resolved_label_field}
        }
        metadata.update(
            {
                "dataset_id": dataset_id,
                "split": split,
                "text_field": resolved_text_field,
                "label_field": resolved_label_field,
            }
        )

        rows.append(
            TextRow(
                id=str(item.get("id", index)),
                text=text,
                label=label,
                label_name=label_name,
                metadata=metadata,
            )
        )

    return rows


def _infer_field(fields: Sequence[str], candidates: Sequence[str], kind: str) -> str:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    field_list = ", ".join(fields)
    raise ValueError(
        f"Could not infer {kind} field from available fields: {field_list}"
    )


def _resolve_label_name(label: Any, label_names: Sequence[str]) -> str:
    if isinstance(label, int) and 0 <= label < len(label_names):
        return label_names[label]
    return str(label)


def _normalize_label_name(label_name: str) -> str:
    return label_name.lower().strip().replace("-", "_").replace(" ", "_")
