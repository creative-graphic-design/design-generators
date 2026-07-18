"""LayoutDiffusion label vocabulary compatibility helpers."""

from __future__ import annotations

from typing import Final

from laygen.common.labels import DatasetName, normalize_dataset_name

VENDOR_RICO25_LABELS: Final[tuple[str, ...]] = (
    "Text",
    "Image",
    "Icon",
    "List_Item",
    "Text_Button",
    "Toolbar",
    "Web_View",
    "Input",
    "Card",
    "Advertisement",
    "Background_Image",
    "Drawer",
    "Radio_Button",
    "Checkbox",
    "Multi_Tab",
    "Pager_Indicator",
    "Modal",
    "On_Off_Switch",
    "Slider",
    "Map_View",
    "Button_Bar",
    "Video",
    "Bottom_Navigation",
    "Number_Stepper",
    "Date_Picker",
)
VENDOR_PUBLAYNET_LABELS: Final[tuple[str, ...]] = (
    "text",
    "title",
    "list",
    "table",
    "figure",
)


def vendor_labels_for_dataset(dataset_name: DatasetName | str) -> tuple[str, ...]:
    """Return LayoutDiffusion vendor label strings in checkpoint order.

    Args:
        dataset_name: Dataset name or alias.

    Returns:
        Ordered vendor label names.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> vendor_labels_for_dataset("publaynet")[0]
        'text'
    """
    dataset = normalize_dataset_name(dataset_name)
    if dataset is DatasetName.rico25:
        return VENDOR_RICO25_LABELS
    if dataset is DatasetName.publaynet:
        return VENDOR_PUBLAYNET_LABELS
    raise ValueError(f"Unsupported LayoutDiffusion dataset_name: {dataset_name}")


def default_id2label(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return the public id-to-label mapping for LayoutDiffusion."""
    return dict(enumerate(vendor_labels_for_dataset(dataset_name)))


def normalize_vendor_label(label: str) -> str:
    """Normalize public spelling to the vendor vocabulary spelling.

    Args:
        label: Label spelling from a public dataset or checkpoint.

    Returns:
        LayoutDiffusion vendor spelling.
    """
    return label.replace(" ", "_")


def vendor_label_to_public_id(dataset_name: DatasetName | str, label: str) -> int:
    """Map a vendor label string to a dataset-local public id."""
    labels = vendor_labels_for_dataset(dataset_name)
    normalized = normalize_vendor_label(label)
    return labels.index(normalized)


def public_id_to_vendor_label(dataset_name: DatasetName | str, label_id: int) -> str:
    """Map a public dataset-local label id to a vendor label string."""
    return vendor_labels_for_dataset(dataset_name)[int(label_id)]
