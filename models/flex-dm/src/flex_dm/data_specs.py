"""Built-in Flex-DM dataset schema helpers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Final, cast

from laygen.common.labels import id2label_for_dataset as laygen_id2label_for_dataset
from posgen.common.labels import id2label_for_dataset as posgen_id2label_for_dataset

from .configuration_flex_dm import FlexDmColumnSpec, FlexDmDatasetName


class FlexDmFeatureGroup(StrEnum):
    """Vendor Flex-DM feature groups."""

    random = auto()
    elem = auto()
    type = auto()
    pos = auto()
    attr = auto()
    img = auto()
    txt = auto()


CRELLO_TYPE_VOCABULARY: Final[tuple[str, ...]] = (
    "coloredBackground",
    "imageElement",
    "maskElement",
    "svgElement",
    "textElement",
)
CRELLO_MODEL_TYPE_VOCABULARY: Final[tuple[str, ...]] = ("", *CRELLO_TYPE_VOCABULARY)
RICO_TYPE_VOCABULARY: Final[tuple[str, ...]] = tuple(
    laygen_id2label_for_dataset("rico25").values()
)


def _normalize_dataset(dataset_name: FlexDmDatasetName | str) -> FlexDmDatasetName:
    if isinstance(dataset_name, FlexDmDatasetName):
        return dataset_name
    try:
        return FlexDmDatasetName(dataset_name.lower().replace("-", "_"))
    except ValueError as exc:
        raise ValueError(f"Unsupported Flex-DM dataset_name: {dataset_name}") from exc


def load_builtin_spec(dataset_name: FlexDmDatasetName | str) -> dict[str, object]:
    """Return a lightweight copy of the vendor column schema.

    Args:
        dataset_name: ``crello`` or ``rico``.

    Returns:
        Dictionary with a ``columns`` mapping.

    Raises:
        ValueError: If the dataset name is unsupported.

    Examples:
        >>> load_builtin_spec("crello")["name"]
        'crello'
    """
    dataset = _normalize_dataset(dataset_name)
    if dataset is FlexDmDatasetName.crello:
        return {
            "name": "crello",
            "columns": {
                "length": {"dtype": "int64", "shape": (1,), "is_sequence": False},
                "group": {"dtype": "string", "shape": (1,), "is_sequence": False},
                "format": {"dtype": "string", "shape": (1,), "is_sequence": False},
                "canvas_width": {
                    "dtype": "int64",
                    "shape": (1,),
                    "is_sequence": False,
                },
                "canvas_height": {
                    "dtype": "int64",
                    "shape": (1,),
                    "is_sequence": False,
                },
                "category": {
                    "dtype": "string",
                    "shape": (1,),
                    "is_sequence": False,
                },
                "type": {"dtype": "string", "shape": (1,), "is_sequence": True},
                "left": {"dtype": "float32", "shape": (1,), "is_sequence": True},
                "top": {"dtype": "float32", "shape": (1,), "is_sequence": True},
                "width": {"dtype": "float32", "shape": (1,), "is_sequence": True},
                "height": {"dtype": "float32", "shape": (1,), "is_sequence": True},
                "opacity": {"dtype": "float32", "shape": (1,), "is_sequence": True},
                "color": {"dtype": "int64", "shape": (3,), "is_sequence": True},
                "image_embedding": {
                    "dtype": "float32",
                    "shape": (512,),
                    "is_sequence": True,
                },
                "text_embedding": {
                    "dtype": "float32",
                    "shape": (512,),
                    "is_sequence": True,
                },
                "font_family": {
                    "dtype": "string",
                    "shape": (1,),
                    "is_sequence": True,
                },
            },
        }
    return {
        "name": "rico",
        "columns": {
            "length": {"dtype": "int64", "shape": (1,), "is_sequence": False},
            "left": {"dtype": "float32", "shape": (1,), "is_sequence": True},
            "top": {"dtype": "float32", "shape": (1,), "is_sequence": True},
            "width": {"dtype": "float32", "shape": (1,), "is_sequence": True},
            "height": {"dtype": "float32", "shape": (1,), "is_sequence": True},
            "clickable": {"dtype": "int64", "shape": (1,), "is_sequence": True},
            "type": {"dtype": "string", "shape": (1,), "is_sequence": True},
            "icon": {"dtype": "string", "shape": (1,), "is_sequence": True},
            "text_button": {
                "dtype": "string",
                "shape": (1,),
                "is_sequence": True,
            },
        },
    }


def attribute_groups_for_dataset(
    dataset_name: FlexDmDatasetName | str,
) -> dict[str, tuple[str, ...]]:
    """Return vendor attribute groups for a dataset."""
    dataset = _normalize_dataset(dataset_name)
    if dataset is FlexDmDatasetName.crello:
        return {
            "type": ("type",),
            "pos": ("left", "top", "width", "height"),
            "attr": ("opacity", "color", "font_family"),
            "img": ("image_embedding",),
            "txt": ("text_embedding",),
        }
    return {
        "type": ("type",),
        "pos": ("left", "top", "width", "height"),
        "attr": ("clickable", "icon", "text_button"),
    }


def _vocabulary_size(
    key: str,
    *,
    dataset_name: FlexDmDatasetName,
    vocabulary: Mapping[str, object],
) -> int:
    if key in ("left", "top", "width", "height"):
        return 64
    if key == "opacity":
        return 8
    if key == "color":
        return 16
    if key == "clickable":
        return 2
    if key == "type" and dataset_name is FlexDmDatasetName.crello:
        return len(CRELLO_MODEL_TYPE_VOCABULARY)
    raw = vocabulary.get(key)
    if key in ("font_family", "icon", "text_button") and isinstance(raw, Mapping):
        return sum(1 for count in raw.values() if int(cast(int, count)) >= 500) + 1
    if key == "type" and isinstance(raw, Mapping):
        return len(raw) + 1
    if isinstance(raw, Mapping):
        return len(raw)
    if isinstance(raw, list | tuple):
        return len(raw)
    if key == "type":
        return len(RICO_TYPE_VOCABULARY)
    if key in ("font_family", "icon", "text_button"):
        return 2
    return 1


def build_column_specs(
    *,
    dataset_name: FlexDmDatasetName | str,
    vocabulary: Mapping[str, object],
) -> dict[str, FlexDmColumnSpec]:
    """Build Flex-DM model column specs from vendor vocabulary metadata."""
    dataset = _normalize_dataset(dataset_name)
    spec = load_builtin_spec(dataset)
    columns = cast(Mapping[str, Mapping[str, object]], spec["columns"])
    input_columns: dict[str, FlexDmColumnSpec] = {}
    for key, column in columns.items():
        shape = tuple(cast(tuple[int, ...], column.get("shape", (1,))))
        dtype = str(column["dtype"])
        is_sequence = bool(column["is_sequence"])
        is_numerical = key in ("image_embedding", "text_embedding")
        if is_numerical:
            item = cast(
                FlexDmColumnSpec,
                {
                    "type": "numerical",
                    "input_dim": None,
                    "shape": shape,
                    "is_sequence": is_sequence,
                    "primary_label": None,
                },
            )
        else:
            input_dim = (
                50
                if key == "length"
                else _vocabulary_size(key, dataset_name=dataset, vocabulary=vocabulary)
            )
            item = cast(
                FlexDmColumnSpec,
                {
                    "type": "categorical",
                    "input_dim": input_dim,
                    "shape": shape,
                    "is_sequence": is_sequence,
                    "primary_label": 0 if key == "type" else None,
                },
            )
        if dataset is FlexDmDatasetName.crello and key in {
            "color",
            "image_embedding",
            "text_embedding",
            "font_family",
        }:
            allowed = {
                "color": {"textElement", "coloredBackground"},
                "image_embedding": {"svgElement", "imageElement", "maskElement"},
                "text_embedding": {"textElement"},
                "font_family": {"textElement"},
            }[key]
            item["loss_condition"] = {
                "key": "type",
                "mask": tuple(
                    label in allowed for label in CRELLO_MODEL_TYPE_VOCABULARY
                ),
            }
        _ = dtype
        input_columns[key] = item
    return input_columns


def id2label_from_vocabulary(
    dataset_name: FlexDmDatasetName | str,
    vocabulary: Mapping[str, object],
) -> dict[int, str]:
    """Resolve the public type-label mapping from vendor vocabulary data."""
    dataset = _normalize_dataset(dataset_name)
    raw = vocabulary.get("type")
    if isinstance(raw, Mapping):
        ordered = sorted(raw.items(), key=lambda item: int(cast(int, item[1])))
        return {idx: str(label) for idx, (label, _count) in enumerate(ordered)}
    if isinstance(raw, list | tuple):
        return {idx: str(label) for idx, label in enumerate(raw)}
    if dataset is FlexDmDatasetName.crello:
        return posgen_id2label_for_dataset("crello")
    return laygen_id2label_for_dataset("rico25")
