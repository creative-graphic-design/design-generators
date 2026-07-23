"""Dataset label registries shared by position-generation packages."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DatasetName(StrEnum):
    """Canonical poster/content dataset names supported by posgen."""

    cgl = "cgl"
    cgl_v2 = "cgl_v2"
    pku_posterlayout = "pku_posterlayout"
    posterlayout = "posterlayout"
    smarttext_demo = "smarttext-demo"
    crello = "crello"


class CGLLabel(StrEnum):
    """CGL label names in dataset id order."""

    logo = "logo"
    text = "text"
    underlay = "underlay"
    embellishment = "embellishment"
    highlighted_text = "highlighted text"


class PKUPosterLayoutLabel(StrEnum):
    """PKU-PosterLayout label names in dataset id order."""

    text = "text"
    logo = "logo"
    underlay = "underlay"
    invalid = "INVALID"


class CrelloLabel(StrEnum):
    """Crello label names in dataset id order."""

    colored_background = "coloredBackground"
    image_element = "imageElement"
    mask_element = "maskElement"
    svg_element = "svgElement"
    text_element = "textElement"


CGL_LABELS: Final[tuple[CGLLabel, ...]] = tuple(CGLLabel)
PKU_POSTERLAYOUT_LABELS: Final[tuple[PKUPosterLayoutLabel, ...]] = tuple(
    PKUPosterLayoutLabel
)
CRELLO_LABELS: Final[tuple[CrelloLabel, ...]] = tuple(CrelloLabel)

_ALIASES: Final[dict[str, DatasetName]] = {
    "cgl": DatasetName.cgl,
    "cgl_dataset": DatasetName.cgl,
    "cgl_v2": DatasetName.cgl_v2,
    "cgl_dataset_v2": DatasetName.cgl_v2,
    "pku": DatasetName.pku_posterlayout,
    "pku_posterlayout": DatasetName.pku_posterlayout,
    "pku_poster_layout": DatasetName.pku_posterlayout,
    "posterlayout": DatasetName.posterlayout,
    "poster_layout": DatasetName.posterlayout,
    "smarttext_demo": DatasetName.smarttext_demo,
    "smarttext-demo": DatasetName.smarttext_demo,
    "crello": DatasetName.crello,
}

_LABELS: Final[dict[DatasetName, tuple[StrEnum, ...]]] = {
    DatasetName.cgl: CGL_LABELS,
    DatasetName.cgl_v2: CGL_LABELS,
    DatasetName.pku_posterlayout: PKU_POSTERLAYOUT_LABELS,
    DatasetName.posterlayout: PKU_POSTERLAYOUT_LABELS,
    DatasetName.smarttext_demo: (PKUPosterLayoutLabel.text,),
    DatasetName.crello: CRELLO_LABELS,
}


def normalize_label(label: str) -> str:
    """Normalize a position-generation label key."""
    return label.strip().lower().replace("-", "_")


def normalize_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize poster/content dataset aliases to canonical names.

    Args:
        dataset_name: Dataset enum or public/release alias.

    Returns:
        Canonical posgen dataset enum.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if isinstance(dataset_name, DatasetName):
        return dataset_name
    key = normalize_label(dataset_name)
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown posgen dataset_name: {dataset_name}") from exc


def labels_for_dataset(dataset_name: DatasetName | str) -> tuple[str, ...]:
    """Return the ordered label vocabulary for a posgen dataset."""
    return tuple(str(label) for label in _LABELS[normalize_dataset_name(dataset_name)])


def id2label_for_dataset(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return an integer-id to label-name mapping for a posgen dataset."""
    return dict(enumerate(labels_for_dataset(dataset_name)))


def label2id_for_dataset(dataset_name: DatasetName | str) -> dict[str, int]:
    """Return a label-name to integer-id mapping for a posgen dataset."""
    return {label: i for i, label in id2label_for_dataset(dataset_name).items()}
