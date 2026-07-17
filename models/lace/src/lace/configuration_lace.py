"""Dataset configuration helpers for LACE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypedDict

from laygen.common.labels import (
    PUBLAYNET_LABELS,
    RICO13_LABELS,
    RICO25_LABELS,
)


class LaceDatasetName(StrEnum):
    """Canonical dataset names supported by LACE."""

    publaynet = "publaynet"
    rico13 = "rico13"
    rico25 = "rico25"


@dataclass(frozen=True)
class LaceDatasetSpec:
    """Static dataset metadata used to configure a LACE checkpoint.

    Attributes:
        dataset: Canonical dataset name.
        labels: Ordered category labels without the padding class.
        max_seq_length: Maximum number of layout elements.
        dim_transformer: Transformer hidden size used by the original model.
        nhead: Number of attention heads.
        num_layers: Number of transformer blocks.
        dim_feedforward: Feed-forward hidden size.
    """

    dataset: LaceDatasetName
    labels: tuple[str, ...]
    max_seq_length: int = 25
    dim_transformer: int = 512
    nhead: int = 16
    num_layers: int = 4
    dim_feedforward: int = 2048

    @property
    def pad_label_id(self) -> int:
        """Return the integer id reserved for padding."""
        return len(self.labels)

    @property
    def num_classes_with_pad(self) -> int:
        """Return the number of category channels including padding."""
        return len(self.labels) + 1

    @property
    def seq_dim(self) -> int:
        """Return the latent per-element feature size."""
        return self.num_classes_with_pad + 4

    @property
    def id2label(self) -> dict[int, str]:
        """Return the category id to label mapping."""
        return dict(enumerate(self.labels))


class LaceModelConfigKwargs(TypedDict):
    """Keyword arguments accepted by ``LaceTransformerModel``."""

    seq_dim: int
    max_seq_length: int
    num_layers: int
    dim_transformer: int
    nhead: int
    dim_feedforward: int
    diffusion_step: int
    timestep_type: str


DATASET_SPECS: Final[dict[LaceDatasetName, LaceDatasetSpec]] = {
    LaceDatasetName.publaynet: LaceDatasetSpec(
        dataset=LaceDatasetName.publaynet,
        labels=tuple(str(label) for label in PUBLAYNET_LABELS),
        dim_transformer=1024,
    ),
    LaceDatasetName.rico13: LaceDatasetSpec(
        dataset=LaceDatasetName.rico13,
        labels=tuple(str(label) for label in RICO13_LABELS),
    ),
    LaceDatasetName.rico25: LaceDatasetSpec(
        dataset=LaceDatasetName.rico25,
        labels=tuple(str(label) for label in RICO25_LABELS),
    ),
}

_ALIASES: Final[dict[str, LaceDatasetName]] = {
    "publaynet": LaceDatasetName.publaynet,
    "publaynet_max25": LaceDatasetName.publaynet,
    "rico13": LaceDatasetName.rico13,
    "rico13_max25": LaceDatasetName.rico13,
    "rico25": LaceDatasetName.rico25,
    "rico25_max25": LaceDatasetName.rico25,
}


def normalize_dataset(dataset: LaceDatasetName | str) -> LaceDatasetName:
    """Normalize a public dataset name to a LACE dataset enum.

    Args:
        dataset: Canonical dataset enum or a supported string alias.

    Returns:
        Canonical LACE dataset name.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> str(normalize_dataset("rico13_max25"))
        'rico13'
    """
    if isinstance(dataset, LaceDatasetName):
        return dataset
    key = dataset.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported LACE dataset: {dataset}") from exc


def get_dataset_spec(dataset: LaceDatasetName | str) -> LaceDatasetSpec:
    """Return dataset metadata for a LACE checkpoint.

    Args:
        dataset: Canonical dataset enum or a supported string alias.

    Returns:
        Dataset specification.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> get_dataset_spec("publaynet").seq_dim
        10
    """
    return DATASET_SPECS[normalize_dataset(dataset)]


def default_model_config(dataset: LaceDatasetName | str) -> LaceModelConfigKwargs:
    """Build the model config for a dataset.

    Args:
        dataset: Canonical dataset enum or a supported string alias.

    Returns:
        Keyword arguments accepted by ``LaceTransformerModel``.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> default_model_config("rico25")["seq_dim"]
        30
    """
    spec = get_dataset_spec(dataset)
    return {
        "seq_dim": spec.seq_dim,
        "max_seq_length": spec.max_seq_length,
        "num_layers": spec.num_layers,
        "dim_transformer": spec.dim_transformer,
        "nhead": spec.nhead,
        "dim_feedforward": spec.dim_feedforward,
        "diffusion_step": 1000,
        "timestep_type": "adalayernorm",
    }
