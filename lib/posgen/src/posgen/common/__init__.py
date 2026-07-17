"""Common building blocks for future position-generation packages."""

from .content import PositionContent
from .labels import (
    DatasetName,
    id2label_for_dataset,
    label2id_for_dataset,
    labels_for_dataset,
    normalize_dataset_name,
    normalize_label,
)
from .testing import assert_position_content_schema
from .visualization import render_position_summary

__all__ = [
    "DatasetName",
    "PositionContent",
    "assert_position_content_schema",
    "id2label_for_dataset",
    "label2id_for_dataset",
    "labels_for_dataset",
    "normalize_dataset_name",
    "normalize_label",
    "render_position_summary",
]
