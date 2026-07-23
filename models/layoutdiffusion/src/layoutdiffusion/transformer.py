"""Transformer denoiser for converted LayoutDiffusion checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange
from jaxtyping import Float, Int
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


@dataclass
class LayoutDiffusionTransformerOutput(BaseOutput):
    """Output returned by ``LayoutDiffusionTransformer``."""

    logits: Float[torch.Tensor, "batch vocab tokens"]


class LayoutDiffusionTransformer(ModelMixin, ConfigMixin):
    """BERT-encoder denoiser ported from the vendor DiscreteTransformerModel.

    Args:
        vocab_size: Full tokenizer vocabulary size including mask.
        num_channels: OpenAI timestep embedding dimension.
        hidden_size: BERT hidden size.
        num_hidden_layers: Number of BERT encoder layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: BERT feed-forward size.
        dropout: Hidden dropout probability.
        max_position_embeddings: Position embedding count.
        constrained: Optional vendor constraint mode.

    Examples:
        >>> model = LayoutDiffusionTransformer(
        ...     vocab_size=16, hidden_size=32, num_channels=8,
        ...     num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
        ... )
        >>> out = model(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, dtype=torch.long))
        >>> out.logits.shape
        torch.Size([2, 15, 5])
    """

    config_name = "transformer_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        vocab_size: int,
        num_channels: int = 128,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        constrained: str | None = None,
    ) -> None:
        """Initialize the transformer."""
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
        )
        self.constrained = constrained
        self.in_channels = 768
        self.model_channels = num_channels
        self.out_channels = vocab_size - 1
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        time_embed_dim = num_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_size),
        )
        self.input_up_proj = nn.Sequential(
            nn.Linear(self.in_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_transformers = BertEncoder(config)
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(dropout)
        self.output_down_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.out_channels),
        )

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings for parity diagnostics."""
        return self.word_embedding(input_ids)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
        condition_ids: Int[torch.Tensor, "batch tokens"] | None = None,
        condition_type: str | None = None,
        return_dict: bool = True,
    ) -> (
        LayoutDiffusionTransformerOutput
        | tuple[Float[torch.Tensor, "batch vocab tokens"]]
    ):
        """Predict start-token logits for a reverse diffusion step.

        Args:
            input_ids: Current token ids shaped ``(B, L)``.
            timesteps: Diffusion timestep per batch item.
            condition_ids: Optional vendor condition token ids.
            condition_type: Optional condition mode.
            return_dict: Whether to return a dataclass output.

        Returns:
            Logits shaped ``(B, vocab_size - 1, L)``.
        """
        x = input_ids
        if condition_ids is not None and condition_type == "label":
            mask = condition_ids.le(self.out_channels - 129).unsqueeze(-1)
            hidden = self.word_embedding(condition_ids) * mask + self.word_embedding(
                x
            ) * (~mask)
        elif condition_ids is not None and condition_type == "completion":
            keep = torch.tensor(
                [1] * 6 + [0] * (condition_ids.shape[1] - 6),
                device=x.device,
                dtype=torch.bool,
            ).expand(condition_ids.shape[0], -1)
            hidden = self.word_embedding(condition_ids) * keep.unsqueeze(
                -1
            ) + self.word_embedding(x) * (~keep).unsqueeze(-1)
        else:
            hidden = self.word_embedding(x)
        emb = self.time_embed(
            get_timestep_embedding(
                timesteps,
                self.model_channels,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
            ).to(hidden)
        )
        seq_length = hidden.size(1)
        position_ids = self.position_ids[:, :seq_length]
        inputs = self.input_up_proj(hidden)
        inputs = inputs + self.position_embeddings(position_ids) + emb.unsqueeze(1)
        inputs = self.dropout_layer(self.LayerNorm(inputs))
        encoded = self.input_transformers(inputs).last_hidden_state
        logits = rearrange(self.output_down_proj(encoded), "b l c -> b c l").type(
            hidden.dtype
        )
        if not return_dict:
            return (logits,)
        return LayoutDiffusionTransformerOutput(logits=logits)
