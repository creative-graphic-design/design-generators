"""PyTorch model wrapper for LayoutAction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
from jaxtyping import Int
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .configuration_layout_action import LayoutActionConfig, LayoutActionSamplingMode
from .generation_layout_action import (
    LayoutActionSamplingConfig,
    sample_action_tokens,
)

if TYPE_CHECKING:
    LongTensor2D: TypeAlias = Int[torch.Tensor, "batch sequence"]
else:
    LongTensor2D: TypeAlias = torch.Tensor


class LayoutActionCausalSelfAttention(nn.Module):
    """Vendor-compatible masked multi-head self-attention."""

    def __init__(self, config: LayoutActionConfig, mask: torch.Tensor) -> None:
        """Initialize key, query, value, and output projections."""
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", mask)
        self.n_head = config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention."""
        batch, steps, channels = x.size()
        key = self.key(x).view(batch, steps, self.n_head, channels // self.n_head)
        query = self.query(x).view(batch, steps, self.n_head, channels // self.n_head)
        value = self.value(x).view(batch, steps, self.n_head, channels // self.n_head)
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        mask = cast(torch.Tensor, self.mask)
        att = att.masked_fill(mask[:, :, :steps, :steps] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ value
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_drop(self.proj(y))


class LayoutActionBlock(nn.Module):
    """Vendor-compatible GPT block."""

    def __init__(self, config: LayoutActionConfig, mask: torch.Tensor) -> None:
        """Initialize layer norms, self-attention, and MLP."""
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = LayoutActionCausalSelfAttention(config, mask)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer block."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class LayoutActionForCausalLM(PreTrainedModel):
    """Transformers ``PreTrainedModel`` for LayoutAction token prediction.

    Args:
        config: LayoutAction architecture and vocabulary metadata.

    Examples:
        >>> config = LayoutActionConfig(n_layer=1, n_head=2, n_embd=16, max_elements=1)
        >>> model = LayoutActionForCausalLM(config)
        >>> out = model(torch.tensor([[config.bos_token_id]]))
        >>> out.logits.shape[-1] == config.vocab_size
        True
    """

    config_class = LayoutActionConfig
    base_model_prefix = "layout_action"
    main_input_name = "input_ids"
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: LayoutActionConfig) -> None:
        """Initialize vendor-compatible GPT modules."""
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.blocks = nn.ModuleList(
            [LayoutActionBlock(config, mask) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.all_tied_weights_keys = dict(self._tied_weights_keys)
        self.post_init()

    def get_block_size(self) -> int:
        """Return the maximum context length."""
        return self.block_size

    def get_input_embeddings(self) -> nn.Embedding:
        """Return token embeddings."""
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace token embeddings."""
        self.tok_emb = value

    def forward(
        self,
        input_ids: LongTensor2D,
        attention_mask: torch.Tensor | None = None,
        labels: LongTensor2D | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ) -> CausalLMOutputWithCrossAttentions | tuple[torch.Tensor, ...]:
        """Run a standard causal language-model forward pass.

        Args:
            input_ids: Token ids shaped ``(batch, sequence)``.
            attention_mask: Accepted for Transformers compatibility; causal
                masking follows the vendor implementation.
            labels: Optional next-token labels.
            return_dict: Whether to return a dataclass output.
            output_hidden_states: Include final hidden states.
            output_attentions: Accepted for API compatibility; attentions are
                not materialized by the vendor-compatible blocks.

        Returns:
            Causal LM output or tuple.

        Raises:
            ValueError: If sequence length exceeds ``block_size``.
        """
        _ = (attention_mask, output_attentions)
        use_return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        batch, steps = input_ids.size()
        if steps > self.block_size:
            raise ValueError("Cannot forward; model block size is exhausted.")
        token_embeddings = self.tok_emb(input_ids)
        position_embeddings = self.pos_emb[:, :steps, :]
        hidden_states = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            target = labels.masked_fill(labels == self.config.pad_token_id, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=-100,
            )
        if not use_return_dict:
            values: tuple[torch.Tensor, ...]
            values = (logits,) if loss is None else (loss, logits)
            if output_hidden_states:
                values = (*values, hidden_states)
            return values
        return CausalLMOutputWithCrossAttentions(
            loss=cast(torch.FloatTensor | None, loss),
            logits=logits,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: LongTensor2D,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = False,
        forced_token_ids: LongTensor2D | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate token ids with the vendor sampling loop.

        Args:
            input_ids: Prompt token ids.
            max_new_tokens: Number of new tokens.
            temperature: Sampling temperature.
            top_k: Optional top-k crop size.
            do_sample: Whether to use multinomial sampling. If ``False``, greedy
                decoding is used.
            forced_token_ids: Optional ids to force at each generation step.
            generator: Optional torch generator for multinomial sampling.

        Returns:
            Prompt plus generated token ids.
        """
        mode = (
            LayoutActionSamplingMode.top_k
            if do_sample and top_k is not None
            else LayoutActionSamplingMode.multinomial
            if do_sample
            else LayoutActionSamplingMode.greedy
        )
        return sample_action_tokens(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            sampling=LayoutActionSamplingConfig(
                mode=mode,
                temperature=temperature,
                top_k=top_k,
            ),
            forced_token_ids=forced_token_ids,
            generator=generator,
        )
