"""Dataset metadata for Parse-Then-Place checkpoints."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final, TypedDict

from laygen.common import RICO25_INTERACTION_LABEL_NAMES, WEBUI_BASE_LABEL_NAMES


class ParseThenPlaceDatasetName(StrEnum):
    """Datasets supported by the original Parse-Then-Place release."""

    rico = auto()
    web = auto()


class Stage2Mode(StrEnum):
    """Released stage-2 checkpoint modes."""

    pretrain = auto()
    finetune = auto()


class DatasetMetadata(TypedDict):
    """Static conversion metadata for one dataset."""

    id2label: dict[int, str]
    canvas_size: tuple[int, int]
    max_elements: int


RICO_LABELS: Final[tuple[str, ...]] = (
    "text button",
    "background image",
    "icon",
    "list item",
    "text",
    "toolbar",
    "web view",
    "input",
    "card",
    "advertisement",
    "image",
    *RICO25_INTERACTION_LABEL_NAMES,
)
WEB_LABELS: Final[tuple[str, ...]] = (
    *WEBUI_BASE_LABEL_NAMES,
    "select",
    "textarea",
)

DATASET_METADATA: Final[dict[ParseThenPlaceDatasetName, DatasetMetadata]] = {
    ParseThenPlaceDatasetName.rico: {
        "id2label": dict(enumerate(RICO_LABELS)),
        "canvas_size": (144, 256),
        "max_elements": 20,
    },
    ParseThenPlaceDatasetName.web: {
        "id2label": dict(enumerate(WEB_LABELS)),
        "canvas_size": (120, 120),
        "max_elements": 76,
    },
}


def normalize_dataset_name(
    dataset_name: ParseThenPlaceDatasetName | str,
) -> ParseThenPlaceDatasetName:
    """Normalize Parse-Then-Place dataset names.

    Args:
        dataset_name: Dataset enum value or public/vendor string.

    Returns:
        Canonical Parse-Then-Place dataset name.

    Raises:
        ValueError: If the dataset is unsupported.
    """
    if isinstance(dataset_name, ParseThenPlaceDatasetName):
        return dataset_name
    key = dataset_name.lower().replace("-", "_")
    if key == "webui":
        key = "web"
    try:
        return ParseThenPlaceDatasetName(key)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported Parse-Then-Place dataset: {dataset_name}"
        ) from exc


def normalize_stage2_mode(stage2_mode: Stage2Mode | str) -> Stage2Mode:
    """Normalize a released stage-2 checkpoint mode."""
    if isinstance(stage2_mode, Stage2Mode):
        return stage2_mode
    try:
        return Stage2Mode(stage2_mode.lower().replace("-", "_"))
    except ValueError as exc:
        raise ValueError(f"Unsupported stage2_mode: {stage2_mode}") from exc


def dataset_metadata(
    dataset_name: ParseThenPlaceDatasetName | str,
) -> DatasetMetadata:
    """Return static metadata for a Parse-Then-Place dataset."""
    return DATASET_METADATA[normalize_dataset_name(dataset_name)]


def id2label_for_dataset(
    dataset_name: ParseThenPlaceDatasetName | str,
) -> dict[int, str]:
    """Return the dataset-local integer-id label map."""
    return dict(dataset_metadata(dataset_name)["id2label"])


def label2id_for_dataset(
    dataset_name: ParseThenPlaceDatasetName | str,
) -> dict[str, int]:
    """Return lower-case label names mapped to dataset-local ids."""
    return {
        label.lower(): idx for idx, label in id2label_for_dataset(dataset_name).items()
    }


def canvas_size_for_dataset(
    dataset_name: ParseThenPlaceDatasetName | str,
) -> tuple[int, int]:
    """Return the vendor canvas size as ``(width, height)``."""
    return dataset_metadata(dataset_name)["canvas_size"]
