from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

CLINC_DATASET_IDS = (
    "clinc/clinc_oos",
    "clinc_oos",
    "DeepPavlov/clinc_oos",
)
CLINC_DEFAULT_CONFIG = "plus"

DBPEDIA_DATASET_IDS = ("fancyzhx/dbpedia_14",)
DBPEDIA_DEFAULT_SPLIT = "train"


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
    if normalized in {"dbpedia", "dbpedia_14", "dbpedia_ontology"}:
        return load_dbpedia(**kwargs)

    raise ValueError(f"Unsupported dataset: {name!r}")


def load_clinc(
    *,
    split: str | None = "test",
    config: str | None = CLINC_DEFAULT_CONFIG,
    dataset_id: str | None = None,
    remove_oos: bool = True,
) -> list[TextRow]:
    """Load CLINC150 rows, optionally dropping out-of-scope examples.

    The default uses the CLINC OOS `plus` test split. After removing OOS, this
    should yield the 4,500 in-scope test queries used by prior clustering work.
    """

    split = split or "test"
    config = CLINC_DEFAULT_CONFIG if config is None else config
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
        except Exception as exc:  # pragma: no cover
            # Fallback path depends on Hugging Face dataset availability.
            errors.append(f"{candidate_id}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to load CLINC dataset.\n{joined_errors}")


def load_dbpedia(
    *,
    split: str | None = DBPEDIA_DEFAULT_SPLIT,
    config: str | None = None,
    dataset_id: str | None = None,
) -> list[TextRow]:
    """Load DBpedia Ontology rows from the 14-class DBpedia article dataset."""

    split = split or DBPEDIA_DEFAULT_SPLIT
    dataset_ids = (dataset_id,) if dataset_id else DBPEDIA_DATASET_IDS
    errors: list[str] = []

    for candidate_id in dataset_ids:
        try:
            return load_hf_text_classification_dataset(
                dataset_id=candidate_id,
                config=config,
                split=split,
                text_fields=("title", "content"),
                label_field="label",
            )
        except Exception as exc:  # pragma: no cover
            # Fallback path depends on Hugging Face dataset availability.
            errors.append(f"{candidate_id}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to load DBpedia dataset.\n{joined_errors}")


def load_hf_text_classification_dataset(
    *,
    dataset_id: str,
    split: str,
    config: str | None = None,
    text_field: str | None = None,
    text_fields: Sequence[str] | None = None,
    label_field: str | None = None,
    remove_label_names: Iterable[str] = (),
) -> list[TextRow]:
    """Load a Hugging Face text classification dataset as `TextRow` objects."""

    from datasets import load_dataset as hf_load_dataset

    if text_field is not None and text_fields is not None:
        raise ValueError("Specify only one of text_field or text_fields.")

    raw_dataset = hf_load_dataset(dataset_id, config, split=split)
    features = raw_dataset.features

    if text_fields is not None:
        resolved_text_fields = tuple(text_fields)
        if not resolved_text_fields:
            raise ValueError("text_fields must contain at least one field.")
        missing_text_fields = [
            field for field in resolved_text_fields if field not in features
        ]
        if missing_text_fields:
            field_list = ", ".join(features.keys())
            missing = ", ".join(missing_text_fields)
            raise ValueError(
                f"Missing text field(s) {missing} from available fields: {field_list}"
            )
    else:
        resolved_text_fields = (
            text_field
            or _infer_field(
                features.keys(), ("text", "sentence", "query", "utterance"), "text"
            ),
        )
    resolved_label_field = label_field or _infer_field(
        features.keys(), ("intent", "label", "category"), "label"
    )

    label_feature = features[resolved_label_field]
    label_names = tuple(getattr(label_feature, "names", ()) or ())
    remove_labels = {_normalize_label_name(label) for label in remove_label_names}

    rows: list[TextRow] = []
    for index, item in enumerate(raw_dataset):
        text = _join_text_fields(item, resolved_text_fields)
        label = item[resolved_label_field]
        label_name = _resolve_label_name(label, label_names)
        if _normalize_label_name(label_name) in remove_labels:
            continue

        metadata = {
            key: value
            for key, value in item.items()
            if key not in {*resolved_text_fields, resolved_label_field}
        }
        if len(resolved_text_fields) == 1:
            text_metadata = {"text_field": resolved_text_fields[0]}
        else:
            text_metadata = {"text_fields": list(resolved_text_fields)}
        metadata.update(
            {
                "dataset_id": dataset_id,
                "config": config,
                "split": split,
                "label_field": resolved_label_field,
                **text_metadata,
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


def _join_text_fields(item: Mapping[str, Any], fields: Sequence[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = item.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


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
