"""PyTorch model wrapper for RALF checkpoints."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from typing import cast
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_ralf import RalfConfig


class RalfForConditionalLayoutGeneration(PreTrainedModel):
    """Forward-compatible `PreTrainedModel` for RALF autoregressive decoding."""

    config_class = RalfConfig
    base_model_prefix = "ralf"
    main_input_name = "input_ids"

    def __init__(self, config: RalfConfig) -> None:
        """Initialize a compact transformer decoder surface."""
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_d_model)
        self.image_projection = nn.Linear(config.image_channels, config.decoder_d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.decoder_d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.decoder_layers)
        self.lm_head = nn.Linear(config.decoder_d_model, config.vocab_size, bias=False)
        self.post_init()

    def _image_context(
        self,
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
    ) -> Float[torch.Tensor, "batch hidden"]:
        if pixel_values is None:
            return self.embed_tokens.weight.new_zeros((1, self.config.decoder_d_model))
        if saliency is None:
            saliency = pixel_values.new_zeros(
                pixel_values.size(0), 1, pixel_values.size(2), pixel_values.size(3)
            )
        merged = torch.cat([pixel_values, saliency], dim=1)
        pooled = merged.mean(dim=(-1, -2))
        return self.image_projection(pooled)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        labels: Int[torch.Tensor, "batch tokens"] | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        """Run teacher-forced token prediction.

        Args:
            input_ids: Autoregressive token ids.
            pixel_values: Optional RGB images.
            saliency: Optional one-channel saliency maps.
            attention_mask: Optional valid-token mask.
            labels: Optional next-token labels.
            return_dict: Whether to return a ModelOutput.
            kwargs: Reserved converted-checkpoint inputs.

        Returns:
            CausalLMOutput or tuple with logits/loss.
        """
        _ = kwargs
        embeddings = self.embed_tokens(input_ids)
        context = self._image_context(pixel_values, saliency)
        if context.size(0) == 1 and embeddings.size(0) != 1:
            context = context.expand(embeddings.size(0), -1)
        hidden = embeddings + context.unsqueeze(1)
        causal_mask = torch.triu(
            torch.full(
                (input_ids.size(1), input_ids.size(1)),
                -math.inf,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.decoder(
            hidden, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            targets = labels.clone()
            targets[targets == self.config.pad_token_id] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=cast(torch.FloatTensor | None, loss), logits=logits)

    @torch.no_grad()
    def _generate_sequences(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        *,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        token_mask: Bool[torch.Tensor, "tokens vocab"] | None = None,
    ) -> Int[torch.Tensor, "batch tokens"]:
        """Run the autoregressive token loop used by `RalfPipeline`."""
        generated = input_ids[:, :1].clone()
        max_length = max_length or self.config.max_token_length
        for step in range(max_length):
            outputs = self(
                input_ids=generated,
                pixel_values=pixel_values,
                saliency=saliency,
                attention_mask=torch.ones_like(generated, dtype=torch.bool),
            )
            logits = outputs.logits[:, -1, :] / temperature
            if token_mask is not None and step < token_mask.size(0):
                logits = logits.masked_fill(
                    ~token_mask[step].to(logits.device), -math.inf
                )
            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                values = torch.topk(logits, top_k).values
                logits = logits.masked_fill(logits < values[:, [-1]], -math.inf)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
