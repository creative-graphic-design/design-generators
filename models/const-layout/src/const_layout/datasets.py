from __future__ import annotations

RICO_LABELS: tuple[str, ...] = (
    "Toolbar",
    "Image",
    "Text",
    "Icon",
    "Text Button",
    "Input",
    "List Item",
    "Advertisement",
    "Pager Indicator",
    "Web View",
    "Background Image",
    "Drawer",
    "Modal",
)

PUBLAYNET_LABELS: tuple[str, ...] = ("text", "title", "list", "table", "figure")

MAGAZINE_LABELS: tuple[str, ...] = (
    "text",
    "image",
    "headline",
    "text-over-image",
    "headline-over-image",
)

DATASET_METADATA: dict[str, dict[str, object]] = {
    "rico": {"name": "rico", "labels": RICO_LABELS, "max_elements": 9},
    "publaynet": {
        "name": "publaynet",
        "labels": PUBLAYNET_LABELS,
        "max_elements": 9,
    },
    "magazine": {
        "name": "magazine",
        "labels": MAGAZINE_LABELS,
        "max_elements": 33,
    },
}

_ALIASES = {
    "rico": "rico",
    "rico13": "rico",
    "publaynet": "publaynet",
    "pub_laynet": "publaynet",
    "magazine": "magazine",
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown const-layout dataset_name: {dataset_name}") from exc


def dataset_metadata(dataset_name: str) -> dict[str, object]:
    return DATASET_METADATA[normalize_dataset_name(dataset_name)]


def labels_for_dataset(dataset_name: str) -> tuple[str, ...]:
    return tuple(dataset_metadata(dataset_name)["labels"])  # type: ignore[arg-type]


def id2label_for_dataset(dataset_name: str) -> dict[int, str]:
    return dict(enumerate(labels_for_dataset(dataset_name)))


def label2id_for_dataset(dataset_name: str) -> dict[str, int]:
    return {label: i for i, label in id2label_for_dataset(dataset_name).items()}
