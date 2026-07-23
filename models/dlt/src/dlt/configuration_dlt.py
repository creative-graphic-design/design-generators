"""Configuration and dataset metadata for DLT pipelines."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final

from diffusers.configuration_utils import ConfigMixin, register_to_config
from laygen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    max_elements_for_dataset,
    normalize_dataset_name,
)


class DLTCoordinateRange(StrEnum):
    """Closed set of DLT public coordinate ranges."""

    normalized_0_1 = auto()


SUPPORTED_DATASETS: Final[frozenset[DatasetName]] = frozenset(
    {DatasetName.publaynet, DatasetName.rico13, DatasetName.magazine}
)


def normalize_dataset(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize and validate a DLT dataset name.

    Args:
        dataset_name: Shared dataset enum or public string alias.

    Returns:
        Canonical shared dataset enum.

    Raises:
        ValueError: If the dataset is not a supported DLT target.

    Examples:
        >>> str(normalize_dataset("rico13"))
        'rico13'
    """
    dataset = normalize_dataset_name(dataset_name)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported DLT dataset_name: {dataset_name}")
    return dataset


def default_id2label(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return DLT public labels for a dataset."""
    return id2label_for_dataset(normalize_dataset(dataset_name))


class DLTConfig(ConfigMixin):
    """Pipeline-level DLT configuration persisted with converted checkpoints.

    Args:
        dataset_name: Canonical dataset name.
        id2label: Optional public label mapping. When omitted, shared dataset
            labels are used.
        max_num_comp: Maximum number of layout elements.
        categories_num: Original category count including pad and mask/drop ids.
        latent_dim: Transformer latent dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        dropout_r: Dropout probability.
        activation: Transformer activation.
        cond_emb_size: Box-condition embedding size.
        cat_emb_size: Category embedding size.
        num_cont_timesteps: Continuous DDPM training timesteps.
        num_discrete_steps: Discrete category diffusion steps.
        beta_schedule: DDPM beta schedule.
        coordinate_range: Public coordinate range.
    """

    config_name = "dlt_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str = "publaynet",
        id2label: dict[int | str, str] | None = None,
        max_num_comp: int | None = None,
        categories_num: int | None = None,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout_r: float = 0.0,
        activation: str = "gelu",
        cond_emb_size: int = 224,
        cat_emb_size: int = 64,
        num_cont_timesteps: int = 100,
        num_discrete_steps: int = 10,
        beta_schedule: str = "squaredcos_cap_v2",
        coordinate_range: DLTCoordinateRange | str = DLTCoordinateRange.normalized_0_1,
    ) -> None:
        """Initialize DLT configuration."""
        dataset = normalize_dataset(dataset_name)
        labels = default_id2label(dataset) if id2label is None else id2label
        self.dataset_name = str(dataset)
        self.id2label = {int(key): value for key, value in labels.items()}
        self.max_num_comp = max_num_comp or max_elements_for_dataset(dataset)
        self.categories_num = categories_num or len(self.id2label) + 2
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_r = dropout_r
        self.activation = activation
        self.cond_emb_size = cond_emb_size
        self.cat_emb_size = cat_emb_size
        self.num_cont_timesteps = num_cont_timesteps
        self.num_discrete_steps = num_discrete_steps
        self.beta_schedule = beta_schedule
        self.coordinate_range = str(DLTCoordinateRange(coordinate_range))

    @property
    def label2id(self) -> dict[str, int]:
        """Return the public label-name to label-id mapping."""
        return {label: idx for idx, label in self.id2label.items()}
