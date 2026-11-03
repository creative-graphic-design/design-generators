"""Cross-attention processor for structured diffusion guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.models.attention_processor import Attention
from jaxtyping import Bool, Float


@dataclass(frozen=True)
class StructuredConditioning:
    """Full-prompt keys, phrase-composed values, and per-example validity."""

    keys: tuple[Float[torch.Tensor, "batch tokens dim"], ...]
    values: tuple[Float[torch.Tensor, "batch tokens dim"], ...]
    key_validity: tuple[Bool[torch.Tensor, "batch"], ...]
    value_validity: tuple[Bool[torch.Tensor, "batch"], ...]
    do_classifier_free_guidance: bool


class StructuredAttentionProcessor:
    """Apply noun-phrase key/value attention in a Diffusers cross-attention layer."""

    def __init__(self, layer_name: str, save_attention_maps: bool) -> None:
        """Create per-layer state for the requested attention-map behavior.

        Args:
            layer_name: U-Net attention processor name used to label saved maps.
            save_attention_maps: Whether to retain conditional attention maps.
        """
        self.layer_name = layer_name
        self.save_attention_maps = save_attention_maps
        self.conditioning: StructuredConditioning | None = None
        self.attention_maps: list[Float[torch.Tensor, "..."]] = []
        self._last_probabilities: list[
            Float[torch.Tensor, "batch_heads query tokens"]
        ] = []

    def __call__(
        self,
        attn: Attention,
        hidden_states: Float[torch.Tensor, "..."],
        encoder_hidden_states: Float[torch.Tensor, "batch tokens dim"] | None = None,
        attention_mask: Bool[torch.Tensor, "..."]
        | Float[torch.Tensor, "..."]
        | None = None,
        temb: Float[torch.Tensor, "batch channels"] | None = None,
        scale: float | None = None,
    ) -> Float[torch.Tensor, "..."]:
        """Run ordinary attention or noun-phrase-aware key/value attention."""
        del scale
        if (
            attn.to_q is None
            or attn.to_k is None
            or attn.to_v is None
            or attn.to_out is None
        ):
            raise ValueError(
                "Attention layers must define query, key, value, and output projections."
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        spatial_shape: tuple[int, int] | None = None
        spatial_dimensions: tuple[int, int, int] | None = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_shape = (height, width)
            spatial_dimensions = (channel, height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width)
            hidden_states = hidden_states.transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        conditioning = self.conditioning
        if conditioning is None:
            output = self._normal_attention(
                attn,
                query,
                encoder_hidden_states,
                attention_mask,
            )
        elif conditioning.do_classifier_free_guidance:
            base_batch = conditioning.keys[0].shape[0]
            if batch_size != 2 * base_batch:
                raise ValueError(
                    "Classifier-free guidance queries must contain matching unconditional and conditional batches."
                )

            unconditional_query, conditional_query = query.split(base_batch, dim=0)
            unconditional_context = encoder_hidden_states[:base_batch]
            unconditional_mask, conditional_mask = self._split_mask(
                attention_mask,
                base_batch,
                batch_size,
            )
            unconditional_output = self._normal_attention(
                attn,
                unconditional_query,
                unconditional_context,
                unconditional_mask,
            )
            conditional_output = self._structured_attention(
                attn,
                conditional_query,
                conditional_mask,
            )
            output = torch.cat((unconditional_output, conditional_output), dim=0)
        else:
            if batch_size != conditioning.keys[0].shape[0]:
                raise ValueError(
                    "Structured conditioning and query batches must match."
                )

            output = self._structured_attention(attn, query, attention_mask)

        if self.save_attention_maps:
            if spatial_shape is None:
                query_length = self._last_probabilities[0].shape[-2]
                side_length = int(query_length**0.5)
                if side_length * side_length == query_length:
                    spatial_shape = (side_length, side_length)

        if self.save_attention_maps and spatial_shape is not None:
            height, width = spatial_shape
            for probabilities in self._last_probabilities:
                map_batch = probabilities.shape[0] // attn.heads
                self.attention_maps.append(
                    probabilities.reshape(
                        map_batch,
                        attn.heads,
                        height,
                        width,
                        probabilities.shape[-1],
                    )[..., :20]
                    .detach()
                    .cpu()
                )

        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        if spatial_dimensions is not None:
            channel, height, width = spatial_dimensions
            output = output.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            output = output + residual

        return output / attn.rescale_output_factor

    def _normal_attention(
        self,
        attn: Attention,
        query: Float[torch.Tensor, "batch sequence channels"],
        context: Float[torch.Tensor, "batch tokens dim"],
        attention_mask: Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
    ) -> Float[torch.Tensor, "batch sequence channels"]:
        if attn.to_k is None or attn.to_v is None:
            raise ValueError("Attention layers must define key and value projections.")

        query_heads = self._to_heads(query, attn.heads)
        key_heads = self._to_heads(attn.to_k(context), attn.heads)
        value_heads = self._to_heads(attn.to_v(context), attn.heads)
        if attn.norm_q is not None:
            query_heads = attn.norm_q(query_heads)

        if attn.norm_k is not None:
            key_heads = attn.norm_k(key_heads)

        probabilities = self._attention_probabilities(
            attn,
            query_heads,
            key_heads,
            attention_mask,
        )
        self._last_probabilities = [probabilities]
        return self._from_heads(torch.bmm(probabilities, value_heads), attn.heads)

    def _structured_attention(
        self,
        attn: Attention,
        query: Float[torch.Tensor, "batch sequence channels"],
        attention_mask: Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
    ) -> Float[torch.Tensor, "batch sequence channels"]:
        conditioning = self.conditioning
        if conditioning is None:
            raise RuntimeError("Structured conditioning has not been installed.")

        if attn.to_k is None or attn.to_v is None:
            raise ValueError("Attention layers must define key and value projections.")

        query_heads = self._to_heads(query, attn.heads)
        if attn.norm_q is not None:
            query_heads = attn.norm_q(query_heads)

        key_heads: list[Float[torch.Tensor, "batch_heads tokens head_dim"]] = []
        for context in conditioning.keys:
            normalized = (
                attn.norm_encoder_hidden_states(context) if attn.norm_cross else context
            )
            projected = self._to_heads(attn.to_k(normalized), attn.heads)
            if attn.norm_k is not None:
                projected = attn.norm_k(projected)

            key_heads.append(projected)

        value_heads = [
            self._to_heads(attn.to_v(context), attn.heads)
            for context in conditioning.values
        ]
        probabilities = [
            self._attention_probabilities(attn, query_heads, key, attention_mask)
            for key in key_heads
        ]

        if len(probabilities) == len(value_heads):
            pairs = list(
                zip(range(len(probabilities)), range(len(value_heads)), strict=True)
            )
        elif len(probabilities) == 1:
            pairs = [(0, value_index) for value_index in range(len(value_heads))]
        elif len(value_heads) == 1:
            pairs = [(key_index, 0) for key_index in range(len(probabilities))]
        else:
            raise ValueError(
                "Structured attention requires matching lists or a singleton side."
            )

        outputs = torch.stack(
            [
                torch.bmm(probabilities[key_index], value_heads[value_index])
                for key_index, value_index in pairs
            ]
        )
        validity = torch.stack(
            [
                conditioning.key_validity[key_index]
                & conditioning.value_validity[value_index]
                for key_index, value_index in pairs
            ]
        ).repeat_interleave(attn.heads, dim=1)
        weights = validity[:, :, None, None].to(outputs.dtype)
        outputs = (outputs * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1)

        self._last_probabilities = [
            attention
            * conditioning.key_validity[key_index]
            .repeat_interleave(attn.heads)
            .to(attention.dtype)[:, None, None]
            for key_index, attention in enumerate(probabilities)
        ]
        return self._from_heads(outputs, attn.heads)

    @staticmethod
    def _to_heads(
        projected: Float[torch.Tensor, "batch sequence inner_dim"], heads: int
    ) -> Float[torch.Tensor, "batch_heads sequence head_dim"]:
        batch_size, sequence_length, inner_dim = projected.shape
        head_dim = inner_dim // heads
        return (
            projected.view(batch_size, sequence_length, heads, head_dim)
            .transpose(1, 2)
            .reshape(batch_size * heads, sequence_length, head_dim)
        )

    @staticmethod
    def _from_heads(
        values: Float[torch.Tensor, "batch_heads sequence head_dim"], heads: int
    ) -> Float[torch.Tensor, "batch sequence inner_dim"]:
        batch_heads, sequence_length, head_dim = values.shape
        batch_size = batch_heads // heads
        return (
            values.view(batch_size, heads, sequence_length, head_dim)
            .transpose(1, 2)
            .reshape(batch_size, sequence_length, heads * head_dim)
        )

    @staticmethod
    def _attention_probabilities(
        attn: Attention,
        query: Float[torch.Tensor, "batch_heads query head_dim"],
        key: Float[torch.Tensor, "batch_heads tokens head_dim"],
        attention_mask: Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
    ) -> Float[torch.Tensor, "batch_heads query tokens"]:
        scores = torch.bmm(query, key.transpose(1, 2)) * attn.scale
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = attention_mask.repeat_interleave(attn.heads, dim=0).unsqueeze(1)
            elif attention_mask.ndim == 4:
                mask = attention_mask.reshape(
                    -1, attention_mask.shape[-2], attention_mask.shape[-1]
                )
            elif attention_mask.shape[0] * attn.heads == scores.shape[0]:
                mask = attention_mask.repeat_interleave(attn.heads, dim=0)
            else:
                mask = attention_mask

            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask

        return scores.softmax(dim=-1)

    @staticmethod
    def _split_mask(
        attention_mask: Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
        base_batch: int,
        total_batch: int,
    ) -> tuple[
        Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
        Bool[torch.Tensor, "..."] | Float[torch.Tensor, "..."] | None,
    ]:
        if attention_mask is None or attention_mask.shape[0] != total_batch:
            return attention_mask, attention_mask

        unconditional_mask, conditional_mask = attention_mask.split(base_batch, dim=0)
        return unconditional_mask, conditional_mask
