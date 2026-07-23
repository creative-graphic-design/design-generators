"""PyTorch model classes for Flex-DM masked document modeling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_flex_dm import FlexDmColumnSpec, FlexDmConfig


@dataclass
class FlexDmModelOutput(ModelOutput):
    """Output of ``FlexDmForMaskedDocumentModeling``.

    Args:
        logits: Per-column logits or numerical predictions.
        loss: Optional summed reconstruction loss.
        hidden_states: Optional final hidden states.
        masks: Optional per-column hidden-field masks.
    """

    logits: dict[str, torch.Tensor]
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    masks: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Keep the logits dictionary as one ModelOutput field."""
        if self.logits is not None:
            self["logits"] = self.logits
        if self.loss is not None:
            self["loss"] = self.loss
        if self.hidden_states is not None:
            self["hidden_states"] = self.hidden_states
        if self.masks is not None:
            self["masks"] = self.masks


def _module_key(key: str) -> str:
    return f"field_{key}"


class FlexDmPreTrainedModel(PreTrainedModel):
    """Base class for Flex-DM Transformers models."""

    config_class = FlexDmConfig
    base_model_prefix = "flex_dm"
    supports_gradient_checkpointing = False


class FlexDmInputEncoder(nn.Module):
    """Encode heterogeneous Flex-DM input columns into one hidden sequence."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create per-column embeddings and projections."""
        super().__init__()
        self.config = config
        self.input_embeddings = nn.ModuleDict()
        self.input_projections = nn.ModuleDict()
        self.special_embeddings = nn.ModuleDict()
        for key, column in config.input_columns.items():
            if not column["is_sequence"]:
                continue
            if column["type"] == "categorical":
                input_dim = cast(int, column["input_dim"])
                self.input_embeddings[_module_key(key)] = nn.Embedding(
                    input_dim + 2,
                    config.latent_dim,
                )
            else:
                self.input_projections[_module_key(key)] = nn.Linear(
                    int(column["shape"][-1]),
                    config.latent_dim,
                )
                self.special_embeddings[_module_key(key)] = nn.Embedding(
                    2,
                    config.latent_dim,
                )
        self.task_embedding = (
            nn.Embedding(len(config.task_names), config.latent_dim)
            if config.context == "id"
            else None
        )
        self.length_embedding = (
            nn.Embedding(config.max_seq_length + 1, config.latent_dim)
            if config.context == "length"
            else None
        )
        self.position_embedding = (
            nn.Embedding(config.max_seq_length, config.latent_dim)
            if config.input_dtype != "set"
            else None
        )

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch seq channels"],
        Bool[torch.Tensor, "batch seq"],
    ]:
        """Encode model inputs.

        Args:
            inputs: Per-column tensors.
            task_ids: Optional vendor task ids.

        Returns:
            Hidden sequence and valid-element mask.
        """
        first_key = self.config.valid_sequence_keys[0]
        batch, seq_len = inputs[first_key].shape[:2]
        device = inputs[first_key].device
        hidden = torch.zeros(
            batch,
            seq_len,
            self.config.latent_dim,
            device=device,
            dtype=torch.float32,
        )
        for key in self.config.valid_sequence_keys:
            column = self.config.input_columns[key]
            value = inputs[key].to(device)
            if column["type"] == "categorical":
                embedded = self.input_embeddings[_module_key(key)](value.long())
                if embedded.ndim == 4:
                    embedded = embedded.sum(dim=-2)
            else:
                projected = self.input_projections[_module_key(key)](value.float())
                masked = (value == 10.0).all(dim=-1)
                unused = (value == 0.0).all(dim=-1)
                special_ids = torch.zeros_like(masked, dtype=torch.long)
                special_ids = torch.where(
                    unused, torch.ones_like(special_ids), special_ids
                )
                special = self.special_embeddings[_module_key(key)](special_ids)
                embedded = torch.where(
                    (masked | unused).unsqueeze(-1), special, projected
                )
            hidden = hidden + embedded
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=device).clamp(
                max=self.config.max_seq_length - 1
            )
            hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        if self.task_embedding is not None and task_ids is not None:
            hidden = hidden + self.task_embedding(task_ids.to(device)).unsqueeze(1)
        if self.length_embedding is not None and "length" in inputs:
            length_ids = (
                inputs["length"]
                .reshape(batch)
                .long()
                .clamp(
                    min=0,
                    max=self.config.max_seq_length,
                )
            )
            hidden = hidden + self.length_embedding(length_ids).unsqueeze(1)
        if "length" in inputs:
            from .masking import get_seq_mask

            seq_mask = get_seq_mask(
                inputs["length"].reshape(batch).long(),
                maxlen=seq_len,
            )
        else:
            seq_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        return hidden, seq_mask


class FlexDmMultiHeadSelfAttention(nn.Module):
    """Vendor-style explicit multi-head self-attention."""

    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        """Create attention projections."""
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _split(
        self, x: Float[torch.Tensor, "batch seq channels"]
    ) -> Float[torch.Tensor, "batch heads seq head_dim"]:
        batch, seq_len, hidden = x.shape
        return x.view(
            batch, seq_len, self.num_heads, hidden // self.num_heads
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq channels"],
        attention_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq channels"]:
        """Apply self-attention using an additive ``-1e9`` padding mask."""
        query = self._split(self.q_proj(hidden_states))
        key = self._split(self.k_proj(hidden_states))
        value = self._split(self.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        additive = (~attention_mask).to(scores.device).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(additive, -1e9)
        weights = scores.softmax(dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        batch, seq_len = hidden_states.shape[:2]
        context = context.view(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(context)


class FlexDmDeepSvgBlock(nn.Module):
    """DeepSVG-style pre-norm transformer block."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create one transformer block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_epsilon)
        self.attention = FlexDmMultiHeadSelfAttention(config.latent_dim)
        self.norm2 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq channels"],
        attention_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq channels"]:
        """Apply pre-norm attention and MLP residuals."""
        hidden_states = hidden_states + self.dropout(
            self.attention(self.norm1(hidden_states), attention_mask)
        )
        return hidden_states + self.dropout(self.mlp(self.norm2(hidden_states)))


class FlexDmDecoder(nn.Module):
    """Decode hidden states into one head per vendor column."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create per-column output heads."""
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict()
        for key in config.valid_sequence_keys:
            column = config.input_columns[key]
            units = int(column["shape"][-1])
            if column["type"] == "categorical":
                units *= cast(int, column["input_dim"])
            self.heads[_module_key(key)] = nn.Linear(config.latent_dim, units)

    def forward(
        self, hidden_states: Float[torch.Tensor, "batch seq channels"]
    ) -> dict[str, torch.Tensor]:
        """Return per-column logits and predictions."""
        outputs: dict[str, torch.Tensor] = {}
        for key in self.config.valid_sequence_keys:
            head = self.heads[_module_key(key)]
            column = self.config.input_columns[key]
            raw = head(hidden_states)
            if column["type"] == "categorical":
                shape_dim = int(column["shape"][-1])
                input_dim = cast(int, column["input_dim"])
                outputs[key] = raw.view(*raw.shape[:2], shape_dim, input_dim)
            else:
                outputs[key] = raw.view(*raw.shape[:2], int(column["shape"][-1]))
        return outputs


class FlexDmForMaskedDocumentModeling(FlexDmPreTrainedModel):
    """Flex-DM MFP model with a standard Transformers ``forward`` method."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Initialize encoder, transformer blocks, and decoder."""
        super().__init__(config)
        if config.arch_type != "oneshot":
            raise ValueError("Only arch_type='oneshot' is supported")
        if config.block_type != "deepsvg":
            raise ValueError("Only block_type='deepsvg' is supported")
        self.encoder = FlexDmInputEncoder(config)
        self.blocks = nn.ModuleList(
            [FlexDmDeepSvgBlock(config) for _ in range(config.num_blocks)]
        )
        self.decoder = FlexDmDecoder(config)
        self.post_init()

    def forward(
        self,
        *,
        inputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor] | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
    ) -> FlexDmModelOutput | tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Run a Flex-DM forward pass.

        Args:
            inputs: Per-column model input tensors.
            masks: Optional hidden-field masks for diagnostics.
            labels: Optional per-column reconstruction targets.
            task_ids: Optional vendor task ids.
            output_hidden_states: Whether to include final hidden states.
            return_dict: Whether to return a ``ModelOutput``.

        Returns:
            ``FlexDmModelOutput`` by default.
        """
        hidden_states, seq_mask = self.encoder(inputs, task_ids=task_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, seq_mask)
        logits = self.decoder(hidden_states)
        loss = self._compute_loss(logits, labels) if labels is not None else None
        output = FlexDmModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states if output_hidden_states else None,
            masks=dict(masks) if masks is not None else None,
        )
        if return_dict is False:
            return logits, loss
        return output

    def _compute_loss(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for key, target in labels.items():
            if key not in logits:
                continue
            column: FlexDmColumnSpec = self.config.input_columns[key]
            pred = logits[key]
            if column["type"] == "categorical":
                vocab = cast(int, column["input_dim"])
                losses.append(
                    F.cross_entropy(pred.view(-1, vocab), target.long().view(-1))
                )
            else:
                losses.append(F.mse_loss(pred, target.float()))
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(losses).sum()
