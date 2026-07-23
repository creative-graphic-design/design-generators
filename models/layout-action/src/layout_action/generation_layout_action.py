"""Token sampling helpers for LayoutAction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import functional as F

from .configuration_layout_action import (
    LayoutActionSamplingMode,
    normalize_sampling_mode,
)


class TokenModelOutput(Protocol):
    """Model output carrying logits."""

    logits: torch.Tensor


class ActionTokenModel(Protocol):
    """Minimal protocol implemented by LayoutAction token models."""

    def get_block_size(self) -> int:
        """Return the maximum context length."""

    def __call__(self, input_ids: torch.Tensor) -> TokenModelOutput:
        """Return a model output with logits."""


@dataclass(frozen=True)
class LayoutActionSamplingConfig:
    """Sampling parameters for LayoutAction token generation.

    Args:
        mode: Greedy, multinomial, or top-k sampling.
        temperature: Positive logit temperature.
        top_k: Optional top-k crop size.

    Examples:
        >>> str(LayoutActionSamplingConfig().mode)
        'top_k'
    """

    mode: LayoutActionSamplingMode = LayoutActionSamplingMode.top_k
    temperature: float = 1.0
    top_k: int | None = 5

    @classmethod
    def from_values(
        cls,
        *,
        mode: LayoutActionSamplingMode | str,
        temperature: float,
        top_k: int | None,
    ) -> "LayoutActionSamplingConfig":
        """Build a normalized sampling config from public values."""
        return cls(
            mode=normalize_sampling_mode(mode),
            temperature=float(temperature),
            top_k=None if top_k is None else int(top_k),
        )


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask logits outside the top ``k`` values exactly like the reference helper.

    Args:
        logits: Logits shaped ``(batch, vocab)``.
        k: Number of top logits to keep.

    Returns:
        Logits with non-top-k entries set to negative infinity.
    """
    values, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < values[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample_action_tokens(
    model: ActionTokenModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    sampling: LayoutActionSamplingConfig,
    forced_token_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively sample LayoutAction token ids.

    Args:
        model: Token model returning logits.
        input_ids: Prompt ids shaped ``(batch, prompt)``.
        max_new_tokens: Number of new tokens to append.
        sampling: Sampling parameters.
        forced_token_ids: Optional ids shaped ``(batch, max_new_tokens)`` with
            ``-100`` for freely sampled positions.
        generator: Optional torch generator for multinomial sampling.

    Returns:
        Prompt plus sampled token ids.
    """
    block_size = model.get_block_size()
    sequence = input_ids.long()
    for step in range(max_new_tokens):
        if forced_token_ids is not None:
            forced = forced_token_ids[:, step]
            if bool(torch.all(forced.ge(0))):
                sequence = torch.cat((sequence, forced.unsqueeze(1)), dim=1)
                continue
        context = (
            sequence if sequence.size(1) <= block_size else sequence[:, -block_size:]
        )
        raw_output = model(context)
        logits = raw_output.logits
        next_logits = logits[:, -1, :] / sampling.temperature
        if (
            sampling.mode is LayoutActionSamplingMode.top_k
            and sampling.top_k is not None
        ):
            next_logits = top_k_logits(next_logits, sampling.top_k)
        probs = F.softmax(next_logits, dim=-1)
        if sampling.mode is LayoutActionSamplingMode.greedy:
            _, next_id = torch.topk(probs, k=1, dim=-1)
        else:
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        if forced_token_ids is not None:
            forced = forced_token_ids[:, step].unsqueeze(1)
            next_id = torch.where(forced.ge(0), forced, next_id)
        sequence = torch.cat((sequence, next_id), dim=1)
    return sequence
