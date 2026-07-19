"""Configuration for converted LayoutTransformer checkpoints."""

from __future__ import annotations

from enum import StrEnum, auto
from collections.abc import Mapping
from typing import Final

from transformers import PretrainedConfig


class DecoderHeadType(StrEnum):
    """Supported LT-Net bbox decoder head types."""

    gmm = auto()
    linear = auto()


class BoxLossType(StrEnum):
    """Supported LT-Net box objective families."""

    pdf = auto()
    reg = auto()


DEFAULT_ID2LABEL: Final[dict[int, str]] = {
    0: "__image__",
    1: "object",
}
DEFAULT_RELATION_ID2LABEL: Final[dict[int, str]] = {
    0: "__in_image__",
    1: "left of",
    2: "right of",
    3: "above",
    4: "below",
    5: "inside",
    6: "surrounding",
}


def _normalize_head_type(value: DecoderHeadType | str) -> DecoderHeadType:
    lowered = str(value).lower()
    if lowered == "gmm":
        return DecoderHeadType.gmm
    if lowered == "linear":
        return DecoderHeadType.linear
    raise ValueError(f"Unsupported decoder head type: {value}")


def _normalize_box_loss(value: BoxLossType | str) -> BoxLossType:
    lowered = str(value).lower()
    if lowered == "pdf":
        return BoxLossType.pdf
    if lowered == "reg":
        return BoxLossType.reg
    raise ValueError(f"Unsupported box loss: {value}")


class LayoutTransformerConfig(PretrainedConfig):
    """Architecture and processor metadata for LT-Net checkpoints.

    Args:
        dataset_name: Dataset slug for the converted checkpoint.
        vocab_size: Mixed special/object/predicate token vocabulary size.
        obj_classes_size: Object-id embedding/classifier vocabulary size.
        hidden_size: Transformer hidden dimension.
        num_hidden_layers: Number of relation encoder layers.
        num_attention_heads: Number of relation encoder attention heads.
        dropout: Dropout used by embeddings, encoder, and bbox heads.
        enable_noise: Whether the original config enabled relation noise.
        noise_size: Original relation-noise size.
        decoder_head_type: ``GMM`` or ``Linear`` bbox head.
        decoder_box_loss: ``PDF`` or ``Reg`` objective family.
        decoder_schedule_sample: Whether training used scheduled sampling.
        decoder_two_path: Whether the original config enabled two-path decoding.
        decoder_global_feature: Whether to concatenate max-pooled global features.
        decoder_greedy: Whether inference uses GMM means instead of sampling.
        xy_temperature: GMM mixture temperature for center coordinates.
        wh_temperature: GMM mixture temperature for box size.
        refine: Whether a refinement head is present.
        refine_head_type: Refinement bbox head type.
        refine_box_loss: Refinement objective family.
        refine_x_softmax: Whether the decoder records XY PDF scores for refine.
        max_sequence_length: Processor/model sequence length.
        id2label: Public object label mapping.
        relation_id2label: Public relation label mapping.
        model_type: Ignored compatibility field from serialized configs.
        transformers_version: Ignored compatibility field.
        kwargs: Additional ``PretrainedConfig`` fields.

    Examples:
        >>> config = LayoutTransformerConfig(hidden_size=32, num_attention_heads=4)
        >>> config.model_type
        'layout-transformer'
    """

    model_type = "layout-transformer"

    def __init__(
        self,
        *,
        dataset_name: str = "coco",
        vocab_size: int = 206,
        obj_classes_size: int = 155,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        enable_noise: bool = False,
        noise_size: int = 64,
        decoder_head_type: DecoderHeadType | str = DecoderHeadType.gmm,
        decoder_box_loss: BoxLossType | str = BoxLossType.pdf,
        decoder_schedule_sample: bool = False,
        decoder_two_path: bool = False,
        decoder_global_feature: bool = True,
        decoder_greedy: bool = True,
        xy_temperature: float = 1.0,
        wh_temperature: float = 1.0,
        refine: bool = False,
        refine_head_type: DecoderHeadType | str = DecoderHeadType.linear,
        refine_box_loss: BoxLossType | str = BoxLossType.reg,
        refine_x_softmax: bool = True,
        max_sequence_length: int = 128,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        relation_id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        mask_token_id: int = 3,
        model_type: str | None = None,
        transformers_version: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize LT-Net architecture and metadata fields."""
        _ = (model_type, transformers_version)
        kwargs.pop("use_vendor_modules", None)
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.obj_classes_size = obj_classes_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.enable_noise = enable_noise
        self.noise_size = noise_size
        self.decoder_head_type = str(_normalize_head_type(decoder_head_type))
        self.decoder_box_loss = str(_normalize_box_loss(decoder_box_loss))
        self.decoder_schedule_sample = decoder_schedule_sample
        self.decoder_two_path = decoder_two_path
        self.decoder_global_feature = decoder_global_feature
        self.decoder_greedy = decoder_greedy
        self.xy_temperature = xy_temperature
        self.wh_temperature = wh_temperature
        self.refine = refine
        self.refine_head_type = str(_normalize_head_type(refine_head_type))
        self.refine_box_loss = str(_normalize_box_loss(refine_box_loss))
        self.refine_x_softmax = refine_x_softmax
        self.max_sequence_length = max_sequence_length
        self.mask_token_id = mask_token_id
        _ = kwargs
        super().__init__()
        normalized_id2label = id2label or DEFAULT_ID2LABEL
        normalized_relation_id2label = relation_id2label or DEFAULT_RELATION_ID2LABEL
        self.id2label = {
            int(key): str(value) for key, value in normalized_id2label.items()
        }
        self.relation_id2label = {
            int(key): str(value) for key, value in normalized_relation_id2label.items()
        }
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.label2id = {value: key for key, value in self.id2label.items()}
