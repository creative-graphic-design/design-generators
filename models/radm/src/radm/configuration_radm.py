"""Configuration for RADM poster layout diffusion pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import Final

from diffusers.configuration_utils import ConfigMixin, register_to_config

from posgen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)


class RADMLabelMode(StrEnum):
    """Closed set of public RADM label vocabularies."""

    english = auto()
    original = auto()


RADM_ORIGINAL_CGL_LABELS: Final[tuple[str, ...]] = (
    "Logo",
    "文字",
    "衬底",
    "符号元素",
    "强调突出子部分文字",
)


def default_id2label(
    dataset_name: DatasetName | str = DatasetName.cgl,
    *,
    label_mode: RADMLabelMode | str = RADMLabelMode.english,
) -> dict[int, str]:
    """Return the public RADM label mapping.

    Args:
        dataset_name: Poster dataset name or alias.
        label_mode: ``"english"`` uses shared CGL aliases; ``"original"`` keeps
            the checkpoint label spellings.

    Returns:
        Integer label ids mapped to display names.

    Raises:
        ValueError: If the dataset or label mode is unsupported.

    Examples:
        >>> default_id2label("cgl")[0]
        'logo'
        >>> default_id2label("cgl", label_mode="original")[1]
        '文字'
    """
    dataset = normalize_dataset_name(dataset_name)
    mode = RADMLabelMode(label_mode)
    if dataset not in {DatasetName.cgl, DatasetName.cgl_v2}:
        raise ValueError(f"Unsupported RADM dataset_name: {dataset_name}")
    if mode is RADMLabelMode.english:
        return id2label_for_dataset(DatasetName.cgl)
    if mode is RADMLabelMode.original:
        return dict(enumerate(RADM_ORIGINAL_CGL_LABELS))
    raise ValueError(f"Unsupported RADM label_mode: {label_mode}")


class RADMConfig(ConfigMixin):
    """Configuration saved with RADM pipeline artifacts.

    Args:
        dataset_name: Poster dataset variant.
        id2label: Public label mapping. Defaults to the shared CGL labels.
        original_id2label: Label spellings from the checked RADM source.
        num_proposals: Number of proposal boxes sampled during inference.
        num_classes: Number of semantic classes predicted by the denoiser.
        hidden_dim: Proposal and condition hidden dimension.
        text_feature_dim: Last dimension of RADM text-feature tensors.
        max_text_num: Maximum text-feature rows per example.
        image_size: Short-side image preprocessing target.
        size_divisibility: Padding multiple for image tensors.
        pixel_mean: RGB mean in 0-255 space.
        pixel_std: RGB standard deviation in 0-255 space.
        inference_steps: Default number of reverse-diffusion steps.
        num_train_timesteps: Number of training diffusion timesteps.
        snr_scale: RADM signal-to-noise scaling metadata.
        sample_step: Original sampling stride metadata.
        class_threshold: Default confidence threshold.
        nms_threshold: Default class-wise NMS threshold.
        denoiser_subfolder: Pipeline denoiser component subfolder.
        scheduler_subfolder: Pipeline scheduler component subfolder.
        processor_subfolder: Pipeline processor component subfolder.
        conversion_report: Conversion metadata persisted with local artifacts.
        checkpoint_status: Human-readable checkpoint status.
        license_status: Human-readable source/weight license status.

    Examples:
        >>> config = RADMConfig(num_proposals=2, hidden_dim=8)
        >>> config.num_labels
        5
    """

    config_name: str = "radm_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: DatasetName | str = DatasetName.cgl,
        id2label: Mapping[int | str, str] | None = None,
        original_id2label: Mapping[int | str, str] | None = None,
        num_proposals: int = 100,
        num_classes: int = 5,
        hidden_dim: int = 256,
        text_feature_dim: int = 768,
        max_text_num: int = 128,
        image_size: int = 800,
        size_divisibility: int = 32,
        pixel_mean: Sequence[float] = (123.675, 116.280, 103.530),
        pixel_std: Sequence[float] = (58.395, 57.120, 57.375),
        inference_steps: int = 50,
        num_train_timesteps: int = 1000,
        snr_scale: float = 2.0,
        sample_step: int = 1,
        class_threshold: float = 0.30,
        nms_threshold: float = 0.50,
        denoiser_subfolder: str = "denoiser",
        scheduler_subfolder: str = "scheduler",
        processor_subfolder: str = "processor",
        conversion_report: Mapping[str, object] | None = None,
        checkpoint_status: str = "RADM README confirms datasets only; no released checkpoint is published",
        license_status: str = "checked original source has no license file",
    ) -> None:
        """Initialize RADM settings."""
        dataset = normalize_dataset_name(dataset_name)
        if dataset not in {DatasetName.cgl, DatasetName.cgl_v2}:
            raise ValueError(f"Unsupported RADM dataset_name: {dataset_name}")
        raw_id2label = id2label or default_id2label(dataset)
        raw_original = original_id2label or default_id2label(
            dataset, label_mode=RADMLabelMode.original
        )
        self.dataset_name = str(dataset)
        self.id2label = {int(key): value for key, value in raw_id2label.items()}
        self.original_id2label = {
            int(key): value for key, value in raw_original.items()
        }
        self.num_proposals = int(num_proposals)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.text_feature_dim = int(text_feature_dim)
        self.max_text_num = int(max_text_num)
        self.image_size = int(image_size)
        self.size_divisibility = int(size_divisibility)
        self.pixel_mean = tuple(float(value) for value in pixel_mean)
        self.pixel_std = tuple(float(value) for value in pixel_std)
        self.inference_steps = int(inference_steps)
        self.num_train_timesteps = int(num_train_timesteps)
        self.snr_scale = float(snr_scale)
        self.sample_step = int(sample_step)
        self.class_threshold = float(class_threshold)
        self.nms_threshold = float(nms_threshold)
        self.denoiser_subfolder = denoiser_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.processor_subfolder = processor_subfolder
        self.conversion_report = dict(conversion_report or {})
        self.checkpoint_status = checkpoint_status
        self.license_status = license_status

    @property
    def label2id(self) -> dict[str, int]:
        """Return the public label-name to integer-id mapping."""
        return {value: key for key, value in self.id2label.items()}

    @property
    def num_labels(self) -> int:
        """Return the configured public class count."""
        return len(self.id2label)
