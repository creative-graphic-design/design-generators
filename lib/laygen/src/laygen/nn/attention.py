"""Attention modules shared by layout-generation models.

``MultiHeadSelfAttention`` follows the hand-written Keras self-attention of the
DeepSVG-style transformer blocks in the Flex-DM checkpoint backbone.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with separate projections and a padding mask.

    Origin:
        This class keeps separate ``q_proj``, ``k_proj``, ``v_proj``, and
        ``out_proj`` linear layers and fills padded key positions with ``-1e9``
        before softmax, so checkpoints converted from Keras
        ``dense_query``/``dense_key``/``dense_value``/``combine_heads`` layers
        load without key remapping inside the module.

        The math equals ``torch.nn.MultiheadAttention`` without dropout, but this
        class keeps three differences on purpose: separate q/k/v projections
        instead of a packed ``in_proj_weight``; a ``-1e9`` fill instead of
        ``-inf``, so a fully padded row yields uniform weights instead of NaN;
        and an explicit matmul-softmax-matmul path instead of fused kernels,
        whose different reduction order would consume the tight agreement
        tolerance against the Keras original.

    Args:
        hidden_size: Hidden dimension of queries, keys, values, and outputs.
        num_heads: Number of attention heads; must divide ``hidden_size``.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.

    Examples:
        >>> import torch
        >>> attention = MultiHeadSelfAttention(hidden_size=8, num_heads=2)
        >>> hidden_states = torch.zeros(1, 3, 8)
        >>> mask = torch.tensor([[True, True, False]])
        >>> attention(hidden_states, mask).shape
        torch.Size([1, 3, 8])
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
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
        """Apply self-attention using an additive ``-1e9`` padding mask.

        Args:
            hidden_states: Input sequence.
            attention_mask: Key mask where ``True`` marks a valid element.

        Returns:
            Attended sequence with the input hidden size.
        """
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
