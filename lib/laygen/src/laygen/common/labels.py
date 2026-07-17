"""Dataset label registries shared by layout generation packages."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DatasetName(StrEnum):
    """Canonical dataset names supported by the shared label registry."""

    rico13 = "rico13"
    rico25 = "rico25"
    publaynet = "publaynet"
    magazine = "magazine"
    crello = "crello"


class Rico13Label(StrEnum):
    """RICO13 label names in dataset id order."""

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


class Rico25Label(StrEnum):
    """RICO25 label names in dataset id order."""

    text = "Text"
    image = "Image"
    icon = "Icon"
    text_button = "Text Button"
    list_item = "List Item"
    input = "Input"
    background_image = "Background Image"
    card = "Card"
    web_view = "Web View"
    radio_button = "Radio Button"
    drawer = "Drawer"
    checkbox = "Checkbox"
    advertisement = "Advertisement"
    modal = "Modal"
    pager_indicator = "Pager Indicator"
    slider = "Slider"
    on_off_switch = "On/Off Switch"
    button_bar = "Button Bar"
    toolbar = "Toolbar"
    number_stepper = "Number Stepper"
    multi_tab = "Multi-Tab"
    date_picker = "Date Picker"
    map_view = "Map View"
    video = "Video"
    bottom_navigation = "Bottom Navigation"


class PubLayNetLabel(StrEnum):
    """PubLayNet label names in dataset id order."""

    text = "text"
    title = "title"
    list = "list"
    table = "table"
    figure = "figure"


class MagazineLabel(StrEnum):
    """Magazine label names in dataset id order."""

    text = "text"
    image = "image"
    headline = "headline"
    text_over_image = "text-over-image"
    headline_over_image = "headline-over-image"


class CrelloLabel(StrEnum):
    """Crello label names in dataset id order."""

    colored_background = "coloredBackground"
    image_element = "imageElement"
    mask_element = "maskElement"
    svg_element = "svgElement"
    text_element = "textElement"


RICO13_LABELS: Final[tuple[Rico13Label, ...]] = tuple(Rico13Label)
RICO25_LABELS: Final[tuple[Rico25Label, ...]] = tuple(Rico25Label)
PUBLAYNET_LABELS: Final[tuple[PubLayNetLabel, ...]] = tuple(PubLayNetLabel)
MAGAZINE_LABELS: Final[tuple[MagazineLabel, ...]] = tuple(MagazineLabel)
CRELLO_LABELS: Final[tuple[CrelloLabel, ...]] = tuple(CrelloLabel)

_ALIASES: Final[dict[str, DatasetName]] = {
    "rico": DatasetName.rico25,
    "rico13": DatasetName.rico13,
    "rico13_max25": DatasetName.rico13,
    "rico25": DatasetName.rico25,
    "rico25_max25": DatasetName.rico25,
    "publaynet": DatasetName.publaynet,
    "publaynet_max25": DatasetName.publaynet,
    "magazine": DatasetName.magazine,
    "crello": DatasetName.crello,
}

_LABELS: Final[dict[DatasetName, tuple[StrEnum, ...]]] = {
    DatasetName.rico13: RICO13_LABELS,
    DatasetName.rico25: RICO25_LABELS,
    DatasetName.publaynet: PUBLAYNET_LABELS,
    DatasetName.magazine: MAGAZINE_LABELS,
    DatasetName.crello: CRELLO_LABELS,
}


def normalize_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize common dataset aliases to canonical registry names.

    Args:
        dataset_name: User-facing dataset name or vendor alias.

    Returns:
        Canonical dataset name used by the shared label registry.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> str(normalize_dataset_name("rico13_max25"))
        'rico13'
    """
    if isinstance(dataset_name, DatasetName):
        return dataset_name
    key = dataset_name.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_name: {dataset_name}") from exc


def labels_for_dataset(dataset_name: DatasetName | str) -> tuple[str, ...]:
    """Return the ordered label vocabulary for a dataset."""
    return tuple(str(label) for label in _LABELS[normalize_dataset_name(dataset_name)])


def id2label_for_dataset(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return an integer-id to label-name mapping for a dataset."""
    return dict(enumerate(labels_for_dataset(dataset_name)))


def label2id_for_dataset(dataset_name: DatasetName | str) -> dict[str, int]:
    """Return a label-name to integer-id mapping for a dataset."""
    return {label: i for i, label in id2label_for_dataset(dataset_name).items()}
