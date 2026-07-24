"""PyTorch model wrapper for LayoutTransformer (LT-Net)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_layout_transformer import LayoutTransformerConfig
from .modeling_lt_compatible import BBoxHead, RelEncoder


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

    vocab_logits: Float[torch.Tensor, "batch sequence vocab"] | None = None
    obj_id_logits: Float[torch.Tensor, "batch sequence object_classes"] | None = None
    token_type_logits: Float[torch.Tensor, "batch sequence token_types"] | None = None
    coarse_box: Float[torch.Tensor, "batch sequence 4"] | None = None
    coarse_gmm: Float[torch.Tensor, "batch sequence gmm_params"] | None = None
    refine_box: Float[torch.Tensor, "batch sequence 4"] | None = None
    refine_gmm: Float[torch.Tensor, "batch sequence gmm_params"] | None = None
    hidden_states: Float[torch.Tensor, "batch sequence hidden"] | None = None


class LayoutTransformerForLayoutGeneration(PreTrainedModel):
    """Transformers ``PreTrainedModel`` for LT-Net relation-to-layout inference."""

    config_class = LayoutTransformerConfig
    base_model_prefix = "layout_transformer"
    main_input_name = "input_token"
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: LayoutTransformerConfig) -> None:
        """Initialize relation encoder and bbox head."""
        super().__init__(config)
        self.encoder = RelEncoder(config)
        self.bbox_head = BBoxHead(config)
        self.all_tied_weights_keys = dict(self._tied_weights_keys)

    def forward(
        self,
        input_token: Int[torch.Tensor, "batch sequence"],
        input_obj_id: Int[torch.Tensor, "batch sequence"],
        segment_label: Int[torch.Tensor, "batch sequence"],
        token_type: Int[torch.Tensor, "batch sequence"],
        src_mask: Bool[torch.Tensor, "batch 1 sequence"] | None = None,
        global_mask: Bool[torch.Tensor, "batch sequence"] | None = None,
        bbox: Float[torch.Tensor, "batch sequence 4"] | None = None,
        bbox_mask: Bool[torch.Tensor, "batch sequence"] | None = None,
        inference: bool = False,
        generator: torch.Generator | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> (
        LayoutTransformerModelOutput
        | tuple[Float[torch.Tensor, "batch sequence feature"] | None, ...]
    ):
        """Run LT-Net relation encoding and bbox prediction.

        Args:
            input_token: Mixed object/predicate token ids.
            input_obj_id: Stable object ids for object-token positions.
            segment_label: Relation segment ids.
            token_type: Token type ids ``0/1/2/3``.
            src_mask: Valid-token mask shaped ``(batch, 1, sequence)``.
            global_mask: Optional reference global-feature mask.
            bbox: Optional teacher-forced boxes for training/parity paths.
            bbox_mask: Optional box validity mask, reserved for parity paths.
            inference: Compatibility flag; greedy decoding is pipeline-owned.
            generator: Optional PyTorch generator for stochastic GMM sampling.
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
        effective_src_mask = (
            src_mask
            if src_mask is not None
            else input_token.ne(self.config.pad_token_id).unsqueeze(1)
        )
        if effective_src_mask.ndim == 2:
            effective_src_mask = effective_src_mask.unsqueeze(1)
        encoder_outputs = self.encoder(
            input_token,
            input_obj_id,
            segment_label,
            token_type,
            effective_src_mask,
        )
        (
            hidden_states,
            vocab_logits,
            obj_id_logits,
            token_type_logits,
            src,
            class_embeds,
        ) = encoder_outputs
        effective_global_mask = (
            global_mask if global_mask is not None else input_token.ge(2)
        )
        if inference:
            coarse_box, coarse_gmm, refine_box, refine_gmm = self.bbox_head.inference(
                hidden_states,
                effective_src_mask,
                src,
                class_embeds,
                effective_global_mask,
                generator=generator,
            )
        else:
            if bbox is None:
                bbox = input_token.new_full(
                    (input_token.size(0), input_token.size(1) - 1, 4), 2.0
                ).float()
            elif bbox.size(1) == input_token.size(1):
                bbox = bbox[:, :-1, :]
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
                generator=generator,
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
        input_token: Int[torch.Tensor, "batch sequence"],
        input_obj_id: Int[torch.Tensor, "batch sequence"],
        segment_label: Int[torch.Tensor, "batch sequence"],
        token_type: Int[torch.Tensor, "batch sequence"],
        src_mask: Bool[torch.Tensor, "batch 1 sequence"] | None = None,
        global_mask: Bool[torch.Tensor, "batch sequence"] | None = None,
        generator: torch.Generator | None = None,
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
            generator=generator,
            return_dict=True,
        )
        return cast(LayoutTransformerModelOutput, output)
