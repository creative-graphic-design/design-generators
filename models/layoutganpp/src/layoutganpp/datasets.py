"""Dataset metadata and label helpers for LayoutGAN++ checkpoints."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, TypedDict

from laygen.common.labels import DatasetName


class RicoLabel(StrEnum):
    """RICO label names in LayoutGAN++ checkpoint order."""

    toolbar = "Toolbar"
    image = "Image"
    text = "Text"
    icon = "Icon"
    text_button = "Text Button"
    input = "Input"
    list_item = "List Item"
    advertisement = "Advertisement"
    pager_indicator = "Pager Indicator"
    web_view = "Web View"
    background_image = "Background Image"
    drawer = "Drawer"
    modal = "Modal"


class PubLayNetLabel(StrEnum):
    """PubLayNet label names in LayoutGAN++ checkpoint order."""

    text = "text"
    title = "title"
    list = "list"
    table = "table"
    figure = "figure"


class MagazineLabel(StrEnum):
    """Magazine label names in LayoutGAN++ checkpoint order."""

    text = "text"
    image = "image"
    headline = "headline"
    text_over_image = "text-over-image"
    headline_over_image = "headline-over-image"


RICO_LABELS: Final[tuple[RicoLabel, ...]] = tuple(RicoLabel)
PUBLAYNET_LABELS: Final[tuple[PubLayNetLabel, ...]] = tuple(PubLayNetLabel)
MAGAZINE_LABELS: Final[tuple[MagazineLabel, ...]] = tuple(MagazineLabel)


class DatasetMetadata(TypedDict):
    """Metadata for a LayoutGAN++ checkpoint dataset."""

    name: DatasetName
    labels: tuple[StrEnum, ...]
    max_elements: int


DATASET_METADATA: Final[dict[DatasetName, DatasetMetadata]] = {
    DatasetName.rico13: {
        "name": DatasetName.rico13,
        "labels": RICO_LABELS,
        "max_elements": 9,
    },
    DatasetName.publaynet: {
        "name": DatasetName.publaynet,
        "labels": PUBLAYNET_LABELS,
        "max_elements": 9,
    },
    DatasetName.magazine: {
        "name": DatasetName.magazine,
        "labels": MAGAZINE_LABELS,
        "max_elements": 33,
    },
}

_ALIASES: Final[dict[str, DatasetName]] = {
    "rico": DatasetName.rico13,
    "rico13": DatasetName.rico13,
    "publaynet": DatasetName.publaynet,
    "pub_laynet": DatasetName.publaynet,
    "magazine": DatasetName.magazine,
}


def normalize_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize a LayoutGAN++ dataset name or alias.

    Args:
        dataset_name: Dataset key such as `rico`, `publaynet`, or `magazine`.

    Returns:
        The canonical dataset key.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> str(normalize_dataset_name("pub-laynet"))
        'publaynet'
    """
    if isinstance(dataset_name, DatasetName):
        return dataset_name
    key = dataset_name.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown layoutganpp dataset_name: {dataset_name}") from exc


def dataset_metadata(dataset_name: DatasetName | str) -> DatasetMetadata:
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


def labels_for_dataset(dataset_name: DatasetName | str) -> tuple[StrEnum, ...]:
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
    return dataset_metadata(dataset_name)["labels"]


def id2label_for_dataset(dataset_name: DatasetName | str) -> dict[int, str]:
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
    return {i: str(label) for i, label in enumerate(labels_for_dataset(dataset_name))}


def label2id_for_dataset(dataset_name: DatasetName | str) -> dict[str, int]:
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
