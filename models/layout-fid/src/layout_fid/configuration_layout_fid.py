"""Configuration for layout FID feature encoders."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Final

from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)


class LayoutFIDArchitecture(StrEnum):
    """Closed set of supported layout FID encoder architectures."""

    fidnet_v3 = auto()
    layoutnet = auto()


class LayoutFIDSource(StrEnum):
    """Closed set of supported released artifact families."""

    layoutdm = auto()
    layoutflow = auto()


class LayoutFIDStatsSplit(StrEnum):
    """Closed set of bundled reference-statistics splits."""

    val = auto()
    test = auto()


DEFAULT_REFERENCE_STATS: Final[dict[str, str]] = {
    "val": "reference_stats/val.npz",
    "test": "reference_stats/test.npz",
}


def normalize_architecture(
    architecture: LayoutFIDArchitecture | str,
) -> LayoutFIDArchitecture:
    """Normalize a public architecture value.

    Args:
        architecture: Architecture enum or string value.

    Returns:
        Normalized architecture enum.

    Raises:
        ValueError: If the architecture is unsupported.

    Examples:
        >>> str(normalize_architecture("layoutnet"))
        'layoutnet'
    """
    if isinstance(architecture, LayoutFIDArchitecture):
        return architecture
    try:
        return LayoutFIDArchitecture(architecture)
    except ValueError as exc:
        raise ValueError(f"Unsupported architecture: {architecture}") from exc


def normalize_source(source: LayoutFIDSource | str) -> LayoutFIDSource:
    """Normalize a public artifact-source value."""
    if isinstance(source, LayoutFIDSource):
        return source
    try:
        return LayoutFIDSource(source)
    except ValueError as exc:
        raise ValueError(f"Unsupported source: {source}") from exc


def normalize_stats_split(split: LayoutFIDStatsSplit | str) -> LayoutFIDStatsSplit:
    """Normalize a public reference-statistics split value."""
    if isinstance(split, LayoutFIDStatsSplit):
        return split
    try:
        return LayoutFIDStatsSplit(split)
    except ValueError as exc:
        raise ValueError(f"Unsupported reference statistics split: {split}") from exc


class LayoutFIDConfig(PretrainedConfig):
    """Configuration saved with layout FID checkpoints.

    Args:
        dataset_name: Canonical layout dataset name.
        id2label: Dataset-local id-to-label metadata.
        architecture: Encoder architecture selected by the checkpoint.
        source: Released artifact family selected by the checkpoint.
        num_public_labels: Number of public dataset labels.
        num_label_embeddings: Number of model label embeddings.
        max_length: Maximum element count accepted by the checkpoint.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        bbox_format_for_model: Internal bbox format consumed by the encoder.
        label_id_offset: Offset applied before model label embedding lookup.
        pad_label_id: Label id used only in padded model tensor positions.
        reference_stats: Relative reference-statistics paths by split.
        kwargs: Extra Hugging Face config fields.

    Raises:
        ValueError: If label counts or enum values are invalid.

    Examples:
        >>> cfg = LayoutFIDConfig(
        ...     dataset_name="publaynet",
        ...     architecture="layoutnet",
        ...     source="layoutflow",
        ...     num_public_labels=5,
        ...     num_label_embeddings=6,
        ...     max_length=20,
        ... )
        >>> cfg.reference_stats["test"]
        'reference_stats/test.npz'
    """

    model_type = "layout-fid"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        dataset_name: DatasetName | str,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        architecture: LayoutFIDArchitecture | str,
        source: LayoutFIDSource | str,
        num_public_labels: int,
        num_label_embeddings: int,
        max_length: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        bbox_format_for_model: BoxFormat | str = "ltrb",
        label_id_offset: int = 0,
        pad_label_id: int = 0,
        reference_stats: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a layout FID checkpoint configuration."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        dataset = normalize_dataset_name(dataset_name)
        arch = normalize_architecture(architecture)
        src = normalize_source(source)
        box_format = normalize_box_format(bbox_format_for_model)
        if num_public_labels <= 0:
            raise ValueError("num_public_labels must be positive")
        if num_label_embeddings < num_public_labels:
            raise ValueError("num_label_embeddings must cover public labels")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if pad_label_id < 0 or pad_label_id >= num_label_embeddings:
            raise ValueError("pad_label_id must be inside the embedding table")

        raw_id2label = id2label or id2label_for_dataset(dataset)
        self.dataset_name = str(dataset)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.architecture = str(arch)
        self.source = str(src)
        self.num_public_labels = num_public_labels
        self.num_label_embeddings = num_label_embeddings
        self.max_length = max_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.bbox_format_for_model = str(box_format)
        self.label_id_offset = label_id_offset
        self.pad_label_id = pad_label_id
        self.reference_stats = dict(reference_stats or DEFAULT_REFERENCE_STATS)

    @property
    def feature_dim(self) -> int:
        """Return the layout FID feature dimension."""
        return self.d_model

    def _get_generation_parameters(self) -> dict[str, object]:
        """Return no generation parameters for this evaluator config."""
        return {}
