"""Configuration for the Transformers-style LayoutDETR generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import Final, cast

from transformers import PretrainedConfig

from .datasets import id2label_for_ad_banner


class BackgroundPreprocessing(StrEnum):
    """Supported public background preprocessing modes."""

    none = auto()
    resize_256 = "256"
    resize_128 = "128"
    blur = auto()
    edge = auto()


DEFAULT_ID2LABEL: Final[dict[int, str]] = id2label_for_ad_banner()


class LayoutDetrConfig(PretrainedConfig):
    """Configuration for LayoutDETR model, processor, and pipeline."""

    model_type = "layout-detr"

    def __init__(
        self,
        *,
        dataset_name: str = "ad_banner",
        id2label: Mapping[int | str, str] | None = None,
        max_seq_length: int = 9,
        z_dim: int = 4,
        img_channels: int = 3,
        img_height: int = 256,
        img_width: int = 256,
        background_size: int = 256,
        hidden_dim: int = 256,
        bert_f_dim: int = 768,
        bert_num_encoder_layers: int = 12,
        bert_num_decoder_layers: int = 2,
        bert_num_heads: int = 4,
        max_text_length: int = 256,
        text_vocab_size: int = 30_522,
        med_config: Mapping[str, object] | None = None,
        backbone_name: str = "resnet50",
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        model_subfolder: str = "model",
        processor_subfolder: str = "processor",
        original_training_options: Mapping[str, object] | None = None,
        conversion_report: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize LayoutDETR configuration."""
        raw_id2label = id2label or DEFAULT_ID2LABEL
        normalized_id2label = {int(key): value for key, value in raw_id2label.items()}
        super().__init__(id2label=normalized_id2label, **kwargs)  # ty: ignore[invalid-argument-type]
        self.dataset_name = dataset_name
        self.id2label = normalized_id2label
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.max_seq_length = int(max_seq_length)
        self.z_dim = int(z_dim)
        self.img_channels = int(img_channels)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.background_size = int(background_size)
        self.hidden_dim = int(hidden_dim)
        self.bert_f_dim = int(bert_f_dim)
        self.bert_num_encoder_layers = int(bert_num_encoder_layers)
        self.bert_num_decoder_layers = int(bert_num_decoder_layers)
        self.bert_num_heads = int(bert_num_heads)
        self.max_text_length = int(max_text_length)
        self.text_vocab_size = int(text_vocab_size)
        self.med_config = dict(med_config or {})
        self.backbone_name = backbone_name
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.model_subfolder = model_subfolder
        self.processor_subfolder = processor_subfolder
        self.original_training_options = dict(original_training_options or {})
        self.conversion_report = dict(conversion_report or {})

    @property
    def num_labels(self) -> int:
        """Return the number of public semantic labels."""
        return len(cast(dict[int, str], self.id2label))

    @property
    def num_bbox_labels(self) -> int:
        """Return the vendor bbox-label count."""
        return self.num_labels

    @property
    def pad_label_id(self) -> int:
        """Return the internal padded label id."""
        return 0

    @property
    def max_elements(self) -> int:
        """Return the maximum generated element count."""
        return self.max_seq_length
