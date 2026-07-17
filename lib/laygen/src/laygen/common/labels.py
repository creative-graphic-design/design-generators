"""Dataset label registries shared by layout generation packages."""

from __future__ import annotations


RICO25_LABELS: tuple[str, ...] = (
    "Text",
    "Image",
    "Icon",
    "Text Button",
    "List Item",
    "Input",
    "Background Image",
    "Card",
    "Web View",
    "Radio Button",
    "Drawer",
    "Checkbox",
    "Advertisement",
    "Modal",
    "Pager Indicator",
    "Slider",
    "On/Off Switch",
    "Button Bar",
    "Toolbar",
    "Number Stepper",
    "Multi-Tab",
    "Date Picker",
    "Map View",
    "Video",
    "Bottom Navigation",
)

PUBLAYNET_LABELS: tuple[str, ...] = ("text", "title", "list", "table", "figure")

MAGAZINE_LABELS: tuple[str, ...] = (
    "text",
    "image",
    "headline",
    "text-over-image",
    "headline-over-image",
)

CRELLO_LABELS: tuple[str, ...] = (
    "coloredBackground",
    "imageElement",
    "maskElement",
    "svgElement",
    "textElement",
)

_ALIASES = {
    "rico": "rico25",
    "rico25": "rico25",
    "rico25_max25": "rico25",
    "publaynet": "publaynet",
    "publaynet_max25": "publaynet",
    "magazine": "magazine",
    "crello": "crello",
}

_LABELS = {
    "rico25": RICO25_LABELS,
    "publaynet": PUBLAYNET_LABELS,
    "magazine": MAGAZINE_LABELS,
    "crello": CRELLO_LABELS,
}


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize common dataset aliases to canonical registry names.

    Args:
        dataset_name: User-facing dataset name or vendor alias.

    Returns:
        Canonical dataset name used by the shared label registry.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> normalize_dataset_name("rico25_max25")
        'rico25'
    """
    key = dataset_name.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_name: {dataset_name}") from exc


def labels_for_dataset(dataset_name: str) -> tuple[str, ...]:
    """Return the ordered label vocabulary for a dataset."""
    return _LABELS[normalize_dataset_name(dataset_name)]


def id2label_for_dataset(dataset_name: str) -> dict[int, str]:
    """Return an integer-id to label-name mapping for a dataset."""
    return dict(enumerate(labels_for_dataset(dataset_name)))


def label2id_for_dataset(dataset_name: str) -> dict[str, int]:
    """Return a label-name to integer-id mapping for a dataset."""
    return {label: i for i, label in id2label_for_dataset(dataset_name).items()}
