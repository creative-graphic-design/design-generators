"""LayoutDiffusion label vocabulary compatibility helpers."""

from __future__ import annotations

from typing import Final

from laygen.common.labels import DatasetName, normalize_dataset_name

LAYOUTDIFFUSION_RICO25_LABELS: Final[tuple[str, ...]] = (
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
LAYOUTDIFFUSION_PUBLAYNET_LABELS: Final[tuple[str, ...]] = (
    "text",
    "title",
    "list",
    "table",
    "figure",
)


def layoutdiffusion_labels_for_dataset(
    dataset_name: DatasetName | str,
) -> tuple[str, ...]:
    """Return LayoutDiffusion label strings in checkpoint order.

    Args:
        dataset_name: Dataset name or alias.

    Returns:
        Ordered checkpoint label names.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> layoutdiffusion_labels_for_dataset("publaynet")[0]
        'text'
    """
    dataset = normalize_dataset_name(dataset_name)
    if dataset is DatasetName.rico25:
        return LAYOUTDIFFUSION_RICO25_LABELS
    if dataset is DatasetName.publaynet:
        return LAYOUTDIFFUSION_PUBLAYNET_LABELS
    raise ValueError(f"Unsupported LayoutDiffusion dataset_name: {dataset_name}")


def default_id2label(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return the public id-to-label mapping for LayoutDiffusion."""
    return dict(enumerate(layoutdiffusion_labels_for_dataset(dataset_name)))


def normalize_layoutdiffusion_label(label: str) -> str:
    """Normalize public spelling to the internal vocabulary spelling.

    Args:
        label: Label spelling from a public dataset or checkpoint.

    Returns:
        LayoutDiffusion checkpoint spelling.
    """
    return label.replace(" ", "_")


def label_to_public_id(dataset_name: DatasetName | str, label: str) -> int:
    """Map a checkpoint label string to a dataset-local public id."""
    labels = layoutdiffusion_labels_for_dataset(dataset_name)
    normalized = normalize_layoutdiffusion_label(label)
    return labels.index(normalized)


def public_id_to_label(dataset_name: DatasetName | str, label_id: int) -> str:
    """Map a public dataset-local label id to a checkpoint label string."""
    return layoutdiffusion_labels_for_dataset(dataset_name)[int(label_id)]
