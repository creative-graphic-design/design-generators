"""Configuration for RALF checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from transformers import PretrainedConfig

from posgen.common.labels import id2label_for_dataset

DEFAULT_VAR_ORDER: Final[tuple[str, ...]] = (
    "label",
    "width",
    "height",
    "center_x",
    "center_y",
)
DEFAULT_SPECIAL_TOKENS: Final[tuple[str, ...]] = ("pad", "bos", "eos")


class RalfConfig(PretrainedConfig):
    """Configuration carrying RALF architecture and tokenizer metadata.

    Args:
        dataset_name: Poster dataset key, usually `cgl` or `pku_posterlayout`.
        task: Canonical condition type or vendor task alias for the checkpoint.
        id2label: Dataset-local label vocabulary persisted with the checkpoint.
        max_seq_length: Maximum number of layout elements.
        num_bin: Number of linear geometry bins per variable.
        var_order: Vendor token variable order.
        special_tokens: Special tokens stored after label and geometry tokens.
        geo_quantization: Geometry quantizer name. The converted package supports
            `linear`; conversion records other values for audit.
        is_loc_vocab_shared: Whether geometry variables share one token range.
        d_model: Image encoder hidden size.
        decoder_d_model: Decoder hidden size.
        encoder_layers: Number of image encoder transformer layers.
        decoder_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout probability.
        retrieval_backbone: Vendor retrieval backbone name.
        top_k: Number of retrieved examples expected by the checkpoint.
        use_reference_image: Whether retrieved reference images participate in fusion.
        layout_backbone: Vendor layout encoder name.
        freeze_layout_encoder: Whether the vendor layout encoder was frozen.
        fusion: Retrieval fusion variant.
        use_flag_embedding: Whether task flag embeddings are enabled.
        use_multitask: Whether checkpoint was trained as multitask.
        global_task_embedding: Whether global task embedding is enabled.
        relation_size: Maximum number of relation constraints.
        image_channels: Number of image channels consumed by the model.
        image_size: Optional `(height, width)` resize target.
        sort_order: Processor sort order metadata.
        retrieval_metadata: Retrieval-cache metadata for conversion/parity.
        original_config: Original config serialized as plain data.
        kwargs: Extra `PretrainedConfig` keyword arguments.
    """

    model_type = "ralf"

    def __init__(
        self,
        dataset_name: str = "cgl",
        task: str = "unconditional",
        id2label: Mapping[int | str, str] | None = None,
        max_seq_length: int = 10,
        num_bin: int = 128,
        var_order: Sequence[str] = DEFAULT_VAR_ORDER,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        geo_quantization: str = "linear",
        is_loc_vocab_shared: bool = False,
        d_model: int = 256,
        decoder_d_model: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        retrieval_backbone: str = "dreamsim",
        saliency_k: int | str = "None",
        top_k: int = 16,
        use_reference_image: bool = False,
        layout_backbone: str = "feature_extractor",
        freeze_layout_encoder: bool = True,
        fusion: str = "concat_cross_attention",
        use_flag_embedding: bool = True,
        use_multitask: bool = False,
        global_task_embedding: bool = False,
        relation_size: int = 10,
        image_channels: int = 4,
        image_size: tuple[int, int] | list[int] | None = None,
        sort_order: Sequence[str] = ("label", "lexicographic"),
        retrieval_metadata: Mapping[str, object] | None = None,
        original_config: Mapping[str, object] | None = None,
        original_hydra_config: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize configuration values."""
        labels = (
            id2label_for_dataset(dataset_name)
            if id2label is None
            else {int(k): str(v) for k, v in id2label.items()}
        )
        self.dataset_name = dataset_name
        self.task = task
        self.id2label = labels
        self.max_seq_length = int(max_seq_length)
        self.num_bin = int(num_bin)
        self.var_order = tuple(var_order)
        self.special_tokens = tuple(special_tokens)
        self.geo_quantization = geo_quantization
        self.is_loc_vocab_shared = bool(is_loc_vocab_shared)
        self.d_model = int(d_model)
        self.decoder_d_model = int(decoder_d_model)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.retrieval_backbone = retrieval_backbone
        self.saliency_k = saliency_k
        self.top_k = int(top_k)
        self.use_reference_image = bool(use_reference_image)
        self.layout_backbone = layout_backbone
        self.freeze_layout_encoder = bool(freeze_layout_encoder)
        self.fusion = fusion
        self.use_flag_embedding = bool(use_flag_embedding)
        self.use_multitask = bool(use_multitask)
        self.global_task_embedding = bool(global_task_embedding)
        self.relation_size = int(relation_size)
        self.image_channels = int(image_channels)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.sort_order = tuple(sort_order)
        self.retrieval_metadata = dict(retrieval_metadata or {})
        self.original_config = dict(original_config or original_hydra_config or {})
        self.original_hydra_config = self.original_config
        pad_token_id = self.special_token_id("pad")
        bos_token_id = self.special_token_id("bos")
        eos_token_id = self.special_token_id("eos")
        kwargs.pop("model_type", None)
        kwargs.pop("id2label", None)
        kwargs.pop("label2id", None)
        kwargs.pop("pad_token_id", None)
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)
        super().__init__(
            id2label=self.id2label,
            label2id={label: idx for idx, label in self.id2label.items()},
        )
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_bbox_tokens(self) -> int:
        """Return the number of geometry tokens."""
        return self.num_bin if self.is_loc_vocab_shared else self.num_bin * 4

    @property
    def vocab_size(self) -> int:
        """Return total autoregressive token vocabulary size."""
        return self.num_labels + self.num_bbox_tokens + len(self.special_tokens)

    @property
    def max_token_length(self) -> int:
        """Return maximum generated token length excluding BOS."""
        return self.max_seq_length * len(self.var_order)

    def special_token_id(self, name: str) -> int:
        """Return the numeric id for a special token name.

        Args:
            name: Special token name without brackets.

        Returns:
            Numeric token id.

        Raises:
            ValueError: If the token is absent from this config.
        """
        if name not in self.special_tokens:
            raise ValueError(f"Unknown special token: {name}")
        return self.num_labels + self.num_bbox_tokens + self.special_tokens.index(name)

    def bbox_token_offset(self, key: str) -> int:
        """Return the first token id for a geometry variable."""
        if key == "label":
            return 0
        if self.is_loc_vocab_shared:
            return self.num_labels
        order = ("width", "height", "center_x", "center_y")
        return self.num_labels + order.index(key) * self.num_bin
