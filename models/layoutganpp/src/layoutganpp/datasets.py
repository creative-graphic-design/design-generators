"""Dataset metadata and label helpers for LayoutGAN++ checkpoints."""

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
    """Normalize a LayoutGAN++ dataset name or alias.

    Args:
        dataset_name: Dataset key such as `rico`, `publaynet`, or `magazine`.

    Returns:
        The canonical dataset key.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> normalize_dataset_name("pub-laynet")
        'publaynet'
    """
    key = dataset_name.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown layoutganpp dataset_name: {dataset_name}") from exc


def dataset_metadata(dataset_name: str) -> dict[str, object]:
    """Return metadata for a LayoutGAN++ dataset.

    Args:
        dataset_name: Dataset key or alias.

    Returns:
        Metadata containing the canonical name, labels, and maximum element count.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> dataset_metadata("rico")["max_elements"]
        9
    """
    return DATASET_METADATA[normalize_dataset_name(dataset_name)]


def labels_for_dataset(dataset_name: str) -> tuple[str, ...]:
    """Return labels for a LayoutGAN++ dataset.

    Args:
        dataset_name: Dataset key or alias.

    Returns:
        Label names in checkpoint order.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> labels_for_dataset("publaynet")[0]
        'text'
    """
    return tuple(dataset_metadata(dataset_name)["labels"])  # type: ignore[arg-type]


def id2label_for_dataset(dataset_name: str) -> dict[int, str]:
    """Return an ID-to-label mapping for a dataset.

    Args:
        dataset_name: Dataset key or alias.

    Returns:
        Dictionary mapping integer IDs to label names.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> id2label_for_dataset("magazine")[1]
        'image'
    """
    return dict(enumerate(labels_for_dataset(dataset_name)))


def label2id_for_dataset(dataset_name: str) -> dict[str, int]:
    """Return a label-to-ID mapping for a dataset.

    Args:
        dataset_name: Dataset key or alias.

    Returns:
        Dictionary mapping label names to integer IDs.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> label2id_for_dataset("rico")["Toolbar"]
        0
    """
    return {label: i for i, label in id2label_for_dataset(dataset_name).items()}
