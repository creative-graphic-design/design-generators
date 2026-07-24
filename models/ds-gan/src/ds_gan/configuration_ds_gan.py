"""Configuration for converted DS-GAN checkpoints."""

from __future__ import annotations

from typing import TypeAlias, cast

from transformers import PretrainedConfig

from posgen.common.labels import (
    DatasetName,
    PKUPosterLayoutLabel,
    normalize_dataset_name,
)

Id2LabelMapping: TypeAlias = dict[int | str, str]


def _semantic_pku_id2label() -> dict[int, str]:
    labels = [
        str(PKUPosterLayoutLabel.text),
        str(PKUPosterLayoutLabel.logo),
        str(PKUPosterLayoutLabel.underlay),
    ]
    return dict(enumerate(labels))


class DSGANConfig(PretrainedConfig):
    """Store DS-GAN architecture and dataset metadata.

    Args:
        dataset_name: Dataset key. DS-GAN currently supports PKU PosterLayout.
        backbone: Timm ResNet backbone name used by the reference checkpoint.
        max_elem: Maximum number of generated elements.
        in_channels: CNN-LSTM input channels after flattening class and box planes.
        out_channels: CNN-LSTM convolution output channels.
        hidden_size: Bidirectional LSTM hidden size.
        num_layers: Number of LSTM layers.
        output_size: Combined class and box output width in the internal model.
        image_size: Processor/model input size as ``(height, width)``.
        reference_canvas_size: Reference normalization canvas as ``(width, height)``.
        backbone_feature_size: Flattened ResNet-FPN spatial size. The reference
            default is ``22 * 15 = 330`` for ``image_size=(350, 240)``.
        model_num_classes: Internal class channels including ``0 = no object``.
        id2label: Public zero-based semantic label mapping.
        model_subfolder: Pipeline subfolder for the model component.
        processor_subfolder: Pipeline subfolder for the processor component.

    Examples:
        >>> DSGANConfig().max_elem
        32
    """

    model_type = "ds_gan"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.pku_posterlayout,
        backbone: str = "resnet50",
        max_elem: int = 32,
        in_channels: int = 8,
        out_channels: int = 32,
        hidden_size: int | None = None,
        num_layers: int = 4,
        output_size: int = 8,
        image_size: tuple[int, int] | list[int] = (350, 240),
        reference_canvas_size: tuple[int, int] | list[int] = (513, 750),
        backbone_feature_size: int = 330,
        model_num_classes: int = 4,
        id2label: Id2LabelMapping | None = None,
        label2id: dict[str, int] | None = None,
        model_subfolder: str = "model",
        processor_subfolder: str = "processor",
        condition_types: list[str] | tuple[str, ...] | None = None,
        architectures: list[str] | None = None,
        model_type: str | None = None,
        transformers_version: str | None = None,
        torch_dtype: str | None = None,
        dtype: str | None = None,
        name_or_path: str = "",
        _commit_hash: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize DS-GAN configuration."""
        _ = (model_type, transformers_version)
        dataset = normalize_dataset_name(dataset_name)
        if dataset is not DatasetName.pku_posterlayout:
            raise ValueError(f"Unsupported DS-GAN dataset_name: {dataset_name}")
        public_id2label = {
            int(k): v for k, v in (id2label or _semantic_pku_id2label()).items()
        }
        public_label2id = label2id or {v: k for k, v in public_id2label.items()}
        super().__init__(
            id2label=public_id2label,
            label2id=public_label2id,
            architectures=architectures or ["DSGANModel"],
            torch_dtype=torch_dtype,  # ty: ignore[unknown-argument]
            dtype=dtype,
            name_or_path=name_or_path,  # ty: ignore[unknown-argument]
            _commit_hash=_commit_hash,  # ty: ignore[unknown-argument]
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        self.dataset_name = str(dataset)
        self.backbone = backbone
        self.max_elem = max_elem
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size if hidden_size is not None else max_elem * 8
        self.num_layers = num_layers
        self.output_size = output_size
        self.image_size = tuple(int(v) for v in image_size)
        self.reference_canvas_size = tuple(int(v) for v in reference_canvas_size)
        self.backbone_feature_size = backbone_feature_size
        self.model_num_classes = model_num_classes
        self.model_subfolder = model_subfolder
        self.processor_subfolder = processor_subfolder
        self.condition_types = list(condition_types or ["content_image"])

    @property
    def public_num_labels(self) -> int:
        """Return the number of public semantic labels."""
        return len(cast(dict[int, str], self.id2label))


def default_ds_gan_config() -> DSGANConfig:
    """Return the reference-compatible DS-GAN default configuration.

    Examples:
        >>> default_ds_gan_config().backbone
        'resnet50'
    """
    return DSGANConfig()


def pku_model_label2id() -> dict[str, int]:
    """Return DS-GAN model labels including the no-object class.

    Examples:
        >>> pku_model_label2id()["no_object"]
        0
    """
    return {"no_object": 0, "text": 1, "logo": 2, "underlay": 3}


def pku_dataset_label2id() -> dict[str, int]:
    """Return PKU dataset annotation labels including ``INVALID``."""
    return {"text": 0, "logo": 1, "underlay": 2, "INVALID": 3}
