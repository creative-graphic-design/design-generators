"""Configuration metadata for CGB-DM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

from diffusers.configuration_utils import ConfigMixin, register_to_config
from posgen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)

Id2LabelMapping: TypeAlias = dict[int | str, str]


@dataclass(frozen=True)
class CGBDMDatasetSpec:
    """Dataset defaults used by CGB-DM training and inference configs.

    Attributes:
        dataset_name: Canonical poster/content dataset enum.
        num_labels: Number of internal class channels, including invalid/pad.
        train_batch_size: Vendor-compatible train batch size.
        learning_rate: Vendor-compatible Adam learning rate.
        id2label: Public label map persisted in checkpoints.
    """

    dataset_name: DatasetName
    num_labels: int
    train_batch_size: int
    learning_rate: float
    id2label: dict[int, str]


def _public_labels(dataset_name: DatasetName) -> dict[int, str]:
    labels = id2label_for_dataset(dataset_name)
    return {key: value for key, value in labels.items() if value != "INVALID"}


DATASET_SPECS: Final[dict[DatasetName, CGBDMDatasetSpec]] = {
    DatasetName.pku_posterlayout: CGBDMDatasetSpec(
        dataset_name=DatasetName.pku_posterlayout,
        num_labels=4,
        train_batch_size=32,
        learning_rate=1.0e-4,
        id2label=_public_labels(DatasetName.pku_posterlayout),
    ),
    DatasetName.cgl: CGBDMDatasetSpec(
        dataset_name=DatasetName.cgl,
        num_labels=5,
        train_batch_size=128,
        learning_rate=2.0e-4,
        id2label=_public_labels(DatasetName.cgl),
    ),
}


class CGBDMConfig(ConfigMixin):
    """Store CGB-DM architecture, schedule, and dataset metadata.

    Args:
        dataset_name: Poster/content dataset key.
        num_labels: Internal class-channel count, including invalid/pad.
        max_seq_length: Maximum number of layout elements.
        image_size: Model image size as ``(height, width)``.
        canvas_size: Dataset canvas size as ``(width, height)``.
        num_train_timesteps: DDPM training timesteps.
        ddim_num_steps: Default DDIM inference steps.
        dim_model: Transformer hidden dimension.
        n_head: Attention head count.
        num_layers: Number of layout decoder layers.
        feature_dim: Feed-forward hidden dimension.
        id2label: Public id-to-label mapping, excluding invalid/pad.

    Examples:
        >>> CGBDMConfig().dataset_name
        'pku_posterlayout'
    """

    config_name = "cgb_dm_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: DatasetName | str = DatasetName.pku_posterlayout,
        num_labels: int | None = None,
        max_seq_length: int = 16,
        image_size: tuple[int, int] | list[int] = (384, 256),
        canvas_size: tuple[int, int] | list[int] = (513, 750),
        num_train_timesteps: int = 1000,
        ddim_num_steps: int = 100,
        dim_model: int = 512,
        n_head: int = 8,
        num_layers: int = 4,
        feature_dim: int = 1024,
        id2label: Id2LabelMapping | None = None,
        condition_types: list[str] | tuple[str, ...] | None = None,
        train_beta_schedule: str = "cosine",
        sampling_beta_schedule: str = "linear",
        model_subfolder: str = "model",
        scheduler_subfolder: str = "scheduler",
        processor_subfolder: str = "processor",
    ) -> None:
        """Initialize CGB-DM configuration."""
        dataset = normalize_dataset_name(dataset_name)
        spec = DATASET_SPECS.get(dataset)
        if spec is None:
            raise ValueError(f"Unsupported CGB-DM dataset_name: {dataset_name}")
        self.dataset_name = str(dataset)
        self.num_labels = int(num_labels or spec.num_labels)
        self.max_seq_length = int(max_seq_length)
        self.image_size: tuple[int, int] = (int(image_size[0]), int(image_size[1]))
        self.canvas_size: tuple[int, int] = (int(canvas_size[0]), int(canvas_size[1]))
        self.num_train_timesteps = int(num_train_timesteps)
        self.ddim_num_steps = int(ddim_num_steps)
        self.dim_model = int(dim_model)
        self.n_head = int(n_head)
        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.id2label = {int(k): v for k, v in (id2label or spec.id2label).items()}
        self.condition_types = list(
            condition_types
            or ["content_image", "label", "label_size", "completion", "refinement"]
        )
        self.train_beta_schedule = train_beta_schedule
        self.sampling_beta_schedule = sampling_beta_schedule
        self.model_subfolder = model_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.processor_subfolder = processor_subfolder

    @property
    def seq_dim(self) -> int:
        """Return the internal layout channel count."""
        return self.num_labels + 4

    @property
    def public_num_labels(self) -> int:
        """Return the public semantic label count."""
        return len(self.id2label)


def cgb_dm_config_for_dataset(dataset_name: DatasetName | str) -> CGBDMConfig:
    """Build a CGB-DM config for a supported dataset.

    Args:
        dataset_name: Dataset key or enum.

    Returns:
        Dataset-specific CGB-DM config.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> cgb_dm_config_for_dataset("cgl").num_labels
        5
    """
    dataset = normalize_dataset_name(dataset_name)
    spec = DATASET_SPECS[dataset]
    return CGBDMConfig(
        dataset_name=dataset,
        num_labels=spec.num_labels,
        id2label=spec.id2label,
    )
