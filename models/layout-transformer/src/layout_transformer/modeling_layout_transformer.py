"""PyTorch model wrapper for LayoutTransformer (LT-Net)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_layout_transformer import LayoutTransformerConfig

if TYPE_CHECKING:
    LongTensor2D: TypeAlias = Int[torch.Tensor, "batch sequence"]
    BoolTensor3D: TypeAlias = Bool[torch.Tensor, "batch 1 sequence"]
    FloatTensor3D: TypeAlias = Float[torch.Tensor, "batch sequence hidden"]
    BBoxTensor: TypeAlias = Float[torch.Tensor, "batch sequence 4"]
else:
    LongTensor2D: TypeAlias = torch.Tensor
    BoolTensor3D: TypeAlias = torch.Tensor
    FloatTensor3D: TypeAlias = torch.Tensor
    BBoxTensor: TypeAlias = torch.Tensor


@dataclass
class LayoutTransformerModelOutput(ModelOutput):
    """Raw LT-Net model outputs.

    Attributes:
        vocab_logits: Mixed token vocabulary logits.
        obj_id_logits: Object-id classifier logits.
        token_type_logits: Token-type classifier logits.
        coarse_box: Coarse normalized center ``xywh`` boxes.
        coarse_gmm: Optional coarse GMM parameters.
        refine_box: Optional refined normalized center ``xywh`` boxes.
        refine_gmm: Optional refinement GMM parameters.
        hidden_states: Optional encoder hidden states.

    Examples:
        >>> import torch
        >>> out = LayoutTransformerModelOutput(
        ...     vocab_logits=torch.zeros(1, 2, 3),
        ...     obj_id_logits=torch.zeros(1, 2, 4),
        ...     token_type_logits=torch.zeros(1, 2, 4),
        ...     coarse_box=torch.zeros(1, 2, 4),
        ... )
        >>> out.coarse_box.shape
        torch.Size([1, 2, 4])
    """

    vocab_logits: FloatTensor3D | None = None
    obj_id_logits: FloatTensor3D | None = None
    token_type_logits: FloatTensor3D | None = None
    coarse_box: BBoxTensor | None = None
    coarse_gmm: torch.Tensor | None = None
    refine_box: BBoxTensor | None = None
    refine_gmm: torch.Tensor | None = None
    hidden_states: FloatTensor3D | None = None


class LayoutTransformerEmbeddings(nn.Module):
    """LT-Net relation token embeddings."""

    def __init__(self, config: LayoutTransformerConfig) -> None:
        """Initialize word, object, segment, and token-type embeddings."""
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.obj_id_embeddings = nn.Embedding(
            config.obj_classes_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.sentence_type = nn.Embedding(33, config.hidden_size, padding_idx=0)
        self.token_type = nn.Embedding(4, config.hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_token: LongTensor2D,
        input_obj_id: LongTensor2D,
        segment_label: LongTensor2D,
        token_type: LongTensor2D,
    ) -> tuple[FloatTensor3D, FloatTensor3D]:
        """Embed relation serializer tensors."""
        token_embeddings = self.word_embeddings(input_token)
        embeddings = (
            token_embeddings
            + self.obj_id_embeddings(input_obj_id)
            + self.sentence_type(segment_label)
            + self.token_type(token_type)
        )
        return self.dropout(embeddings), token_embeddings


class LayoutTransformerEncoder(nn.Module):
    """Transformer encoder matching LT-Net relation inputs."""

    def __init__(self, config: LayoutTransformerConfig) -> None:
        """Initialize relation encoder layers and token classifiers."""
        super().__init__()
        self.input_embeddings = LayoutTransformerEmbeddings(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.vocab_classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.obj_id_classifier = nn.Linear(config.hidden_size, config.obj_classes_size)
        self.token_type_classifier = nn.Linear(config.hidden_size, 4)

    def forward(
        self,
        input_token: LongTensor2D,
        input_obj_id: LongTensor2D,
        segment_label: LongTensor2D,
        token_type: LongTensor2D,
        src_mask: BoolTensor3D | None = None,
    ) -> tuple[FloatTensor3D, torch.Tensor, torch.Tensor, torch.Tensor, FloatTensor3D]:
        """Encode relation tokens and return token reconstruction logits."""
        embeddings, class_embeds = self.input_embeddings(
            input_token,
            input_obj_id,
            segment_label,
            token_type,
        )
        padding_mask = None
        if src_mask is not None:
            mask_2d = (
                src_mask.squeeze(1).bool() if src_mask.ndim == 3 else src_mask.bool()
            )
            padding_mask = ~mask_2d
        hidden_states = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        hidden_states = self.layer_norm(hidden_states)
        return (
            hidden_states,
            self.vocab_classifier(hidden_states),
            self.obj_id_classifier(hidden_states),
            self.token_type_classifier(hidden_states),
            class_embeds,
        )


class LayoutTransformerBBoxHead(nn.Module):
    """Lightweight bbox head with LT-Net-compatible output semantics."""

    def __init__(self, config: LayoutTransformerConfig) -> None:
        """Initialize coarse and optional refinement box predictors."""
        super().__init__()
        self.config = config
        self.coarse_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 4),
            nn.Sigmoid(),
        )
        self.refine_head = None
        if config.refine:
            self.refine_head = nn.Sequential(
                nn.Linear(config.hidden_size + 4, config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, 4),
                nn.Sigmoid(),
            )

    def forward(
        self,
        encoder_output: FloatTensor3D,
        trg_input_box: BBoxTensor | None = None,
    ) -> tuple[BBoxTensor, None, BBoxTensor | None, None]:
        """Predict coarse and optional refined boxes for each token."""
        coarse_box = cast(BBoxTensor, self.coarse_head(encoder_output))
        refine_box = None
        if self.refine_head is not None:
            refine_input_box = coarse_box if trg_input_box is None else trg_input_box
            refine_box = cast(
                BBoxTensor,
                self.refine_head(torch.cat((encoder_output, refine_input_box), dim=-1)),
            )
        return coarse_box, None, refine_box, None


class LayoutTransformerForLayoutGeneration(PreTrainedModel):
    """Transformers ``PreTrainedModel`` for LT-Net relation-to-layout inference."""

    config_class = LayoutTransformerConfig
    base_model_prefix = "layout_transformer"
    main_input_name = "input_token"
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: LayoutTransformerConfig) -> None:
        """Initialize relation encoder and bbox head."""
        super().__init__(config)
        self._uses_vendor_modules = False
        if config.use_vendor_modules:
            self._init_vendor_modules(config)
        else:
            self.encoder = LayoutTransformerEncoder(config)
            self.bbox_head = LayoutTransformerBBoxHead(config)
        self.all_tied_weights_keys = dict(self._tied_weights_keys)

    def _init_vendor_modules(self, config: LayoutTransformerConfig) -> None:
        """Attach vendor modules under matching state-dict key prefixes."""
        vendor_root = self._find_vendor_root()
        if vendor_root is None:
            raise FileNotFoundError(
                "vendor/layout-transformer is required for use_vendor_modules=True"
            )
        sys.path.insert(0, str(vendor_root))
        from model import Rel2Bbox

        cfg = {
            "MODEL": {
                "PRETRAIN": False,
                "ENCODER": {
                    "VOCAB_SIZE": config.vocab_size,
                    "OBJ_CLASSES_SIZE": config.obj_classes_size,
                    "HIDDEN_SIZE": config.hidden_size,
                    "NUM_LAYERS": config.num_hidden_layers,
                    "ATTN_HEADS": config.num_attention_heads,
                    "DROPOUT": config.dropout,
                    "ENABLE_NOISE": config.enable_noise,
                    "NOISE_SIZE": config.noise_size,
                },
                "DECODER": {
                    "HEAD_TYPE": config.decoder_head_type.upper(),
                    "BOX_LOSS": config.decoder_box_loss.upper(),
                    "SCHEDULE_SAMPLE": config.decoder_schedule_sample,
                    "TWO_PATH": config.decoder_two_path,
                    "GLOBAL_FEATURE": config.decoder_global_feature,
                    "GREEDY": config.decoder_greedy,
                    "XY_TEMP": config.xy_temperature,
                    "WH_TEMP": config.wh_temperature,
                },
                "REFINE": {
                    "REFINE": config.refine,
                    "HEAD_TYPE": config.refine_head_type.title(),
                    "BOX_LOSS": config.refine_box_loss.title(),
                    "X_Softmax": config.refine_x_softmax,
                },
            }
        }
        vendor_model = Rel2Bbox(
            vocab_size=config.vocab_size,
            obj_classes_size=config.obj_classes_size,
            noise_size=config.noise_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            attn_heads=config.num_attention_heads,
            dropout=config.dropout,
            cfg=cfg,
        )
        self.encoder = vendor_model.encoder
        self.bbox_head = vendor_model.bbox_head
        self._uses_vendor_modules = True

    @staticmethod
    def _find_vendor_root() -> Path | None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "vendor" / "layout-transformer"
            if (candidate / "model").is_dir():
                return candidate
        return None

    def forward(
        self,
        input_token: LongTensor2D,
        input_obj_id: LongTensor2D,
        segment_label: LongTensor2D,
        token_type: LongTensor2D,
        src_mask: BoolTensor3D | None = None,
        global_mask: torch.Tensor | None = None,
        bbox: BBoxTensor | None = None,
        bbox_mask: torch.Tensor | None = None,
        inference: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> LayoutTransformerModelOutput | tuple[torch.Tensor | None, ...]:
        """Run LT-Net relation encoding and bbox prediction.

        Args:
            input_token: Mixed object/predicate token ids.
            input_obj_id: Stable object ids for object-token positions.
            segment_label: Relation segment ids.
            token_type: Token type ids ``0/1/2/3``.
            src_mask: Valid-token mask shaped ``(batch, 1, sequence)``.
            global_mask: Optional vendor global-feature mask.
            bbox: Optional teacher-forced boxes for training/parity paths.
            bbox_mask: Optional box validity mask, reserved for parity paths.
            inference: Compatibility flag; greedy decoding is pipeline-owned.
            output_hidden_states: Whether to include encoder hidden states.
            return_dict: Whether to return a ``ModelOutput``.

        Returns:
            Raw LT-Net output dataclass or tuple.

        Raises:
            ValueError: If required tensor shapes are invalid.
        """
        _ = bbox_mask
        if input_token.shape != input_obj_id.shape:
            raise ValueError("input_token and input_obj_id must have the same shape")
        encoder_outputs = self.encoder(
            input_token,
            input_obj_id,
            segment_label,
            token_type,
            src_mask,
        )
        if self._uses_vendor_modules:
            (
                hidden_states,
                vocab_logits,
                obj_id_logits,
                token_type_logits,
                src,
                class_embeds,
            ) = encoder_outputs
        else:
            (
                hidden_states,
                vocab_logits,
                obj_id_logits,
                token_type_logits,
                class_embeds,
            ) = encoder_outputs
            src = hidden_states
        if self._uses_vendor_modules:
            effective_src_mask = (
                src_mask
                if src_mask is not None
                else input_token.ne(self.config.pad_token_id).unsqueeze(1)
            )
            effective_global_mask = (
                global_mask if global_mask is not None else input_token.ge(2)
            )
            if inference:
                coarse_box, coarse_gmm, refine_box, refine_gmm = (
                    self.bbox_head.inference(
                        hidden_states,
                        effective_src_mask,
                        src,
                        class_embeds,
                        effective_global_mask,
                    )
                )
            else:
                if bbox is None:
                    raise ValueError("bbox is required for vendor training forward")
                trg_mask = effective_src_mask.new_ones([1, 1, 1])
                coarse_box, coarse_gmm, refine_box, refine_gmm = self.bbox_head(
                    0,
                    hidden_states,
                    effective_src_mask,
                    src,
                    class_embeds,
                    bbox,
                    trg_mask,
                    effective_global_mask,
                )
        else:
            coarse_box, coarse_gmm, refine_box, refine_gmm = self.bbox_head(
                hidden_states,
                trg_input_box=bbox,
            )
        output = LayoutTransformerModelOutput(
            vocab_logits=vocab_logits,
            obj_id_logits=obj_id_logits,
            token_type_logits=token_type_logits,
            coarse_box=coarse_box,
            coarse_gmm=coarse_gmm,
            refine_box=refine_box,
            refine_gmm=refine_gmm,
            hidden_states=hidden_states if output_hidden_states else None,
        )
        if return_dict:
            return output
        return output.to_tuple()

    @torch.no_grad()
    def _generate_boxes(
        self,
        input_token: LongTensor2D,
        input_obj_id: LongTensor2D,
        segment_label: LongTensor2D,
        token_type: LongTensor2D,
        src_mask: BoolTensor3D | None = None,
        global_mask: torch.Tensor | None = None,
    ) -> LayoutTransformerModelOutput:
        """Private pipeline helper for layout-level generation."""
        output = self(
            input_token=input_token,
            input_obj_id=input_obj_id,
            segment_label=segment_label,
            token_type=token_type,
            src_mask=src_mask,
            global_mask=global_mask,
            inference=True,
            return_dict=True,
        )
        return cast(LayoutTransformerModelOutput, output)
