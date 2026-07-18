"""Discrete diffusion tensor utilities shared by layout generators."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING
from typing import Final, assert_never

if TYPE_CHECKING:
    import torch

LOG_EPS: Final[float] = -70.0


class SamplingMode(StrEnum):
    """Supported categorical sampling modes."""

    deterministic = auto()
    random = auto()
    gumbel = auto()
    top_k = auto()
    top_p = auto()
    top_k_top_p = auto()


def normalize_sampling_mode(sampling: SamplingMode | str) -> SamplingMode:
    """Convert a public sampling value to ``SamplingMode``.

    Args:
        sampling: Sampling enum or its string value.

    Returns:
        Normalized ``SamplingMode`` enum.

    Raises:
        ValueError: If ``sampling`` is not supported.
    """
    if isinstance(sampling, SamplingMode):
        return sampling
    try:
        return SamplingMode(sampling)
    except ValueError as exc:
        raise ValueError(f"Unsupported sampling mode: {sampling}") from exc


def index_to_log_onehot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert categorical ids to log one-hot tensors.

    Args:
        input_ids: Integer tensor with categorical ids.
        vocab_size: Size of the categorical vocabulary.

    Returns:
        Log one-hot tensor shaped ``(batch, vocab, ...)``.

    Raises:
        ValueError: If any id is outside the vocabulary.

    Examples:
        >>> import torch
        >>> index_to_log_onehot(torch.tensor([[0, 1]]), 3).shape
        torch.Size([1, 3, 2])
    """
    import torch
    import torch.nn.functional as F

    if input_ids.numel() and input_ids.max().item() >= vocab_size:
        raise ValueError(
            f"input id {input_ids.max().item()} exceeds vocab_size {vocab_size}"
        )
    onehot = F.one_hot(input_ids.long(), vocab_size)
    order = (0, -1) + tuple(range(1, input_ids.ndim))
    return torch.log(onehot.permute(order).float().clamp(min=1e-30))


def log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    """Convert log one-hot tensors back to categorical ids."""
    return log_x.argmax(dim=1)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a numerically stable elementwise ``log(exp(a) + exp(b))``."""
    import torch

    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(
    values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    """Gather timestep values and reshape them for broadcast operations."""
    batch, *_ = timesteps.shape
    out = values.to(timesteps.device).gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def gumbel_noise_like(
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Gumbel noise with the same shape, dtype, and device as ``x``."""
    import torch

    uniform = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return -torch.log(-torch.log(uniform + 1e-30) + 1e-30)


def log_sample_categorical(
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample categorical ids from log probabilities with Gumbel-max."""
    return (logits + gumbel_noise_like(logits, generator=generator)).argmax(dim=1)


def top_k_logits(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Mask logits outside the top-k entries along ``dim``."""
    import torch

    if k <= 0 or k >= logits.size(dim):
        return logits
    values = torch.topk(logits, k, dim=dim).values
    threshold = values.select(dim, k - 1).unsqueeze(dim)
    return logits.masked_fill(logits < threshold, LOG_EPS)


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    import torch

    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, LOG_EPS)
    return torch.empty_like(logits).scatter(
        dim=-1, index=sorted_indices, src=sorted_logits
    )


def sample_categorical(
    logits: torch.Tensor,
    *,
    sampling: SamplingMode | str = SamplingMode.random,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample categorical ids from logits using LayoutDM sampling modes.

    Args:
        logits: Tensor whose last dimension is the categorical vocabulary.
        sampling: Sampling mode name.
        temperature: Positive temperature used before random sampling.
        top_k: Number of logits retained for top-k modes.
        top_p: Cumulative probability retained for top-p modes.
        generator: Optional torch generator for deterministic sampling.

    Returns:
        Tensor of sampled ids with shape ``logits.shape[:-1]``.

    Examples:
        >>> import torch
        >>> sample_categorical(
        ...     torch.tensor([[[0.0, 1.0]]]),
        ...     sampling="deterministic",
        ... )
        tensor([[1]])
    """
    import torch

    mode = normalize_sampling_mode(sampling)
    match mode:
        case SamplingMode.deterministic:
            return logits.argmax(dim=-1)
        case SamplingMode.random:
            scaled = logits / temperature
        case SamplingMode.gumbel:
            scaled = logits / temperature
            return (scaled + gumbel_noise_like(scaled, generator=generator)).argmax(
                dim=-1
            )
        case SamplingMode.top_k:
            scaled = logits / temperature
            if top_k is not None:
                scaled = top_k_logits(scaled, top_k, dim=-1)
        case SamplingMode.top_p:
            scaled = logits / temperature
            if top_p is not None:
                scaled = _top_p_logits(scaled, top_p)
        case SamplingMode.top_k_top_p:
            scaled = logits / temperature
            if top_k is not None:
                scaled = top_k_logits(scaled, top_k, dim=-1)
            if top_p is not None:
                scaled = _top_p_logits(scaled, top_p)
        case _:
            assert_never(mode)
    probs = scaled.softmax(dim=-1).reshape(-1, scaled.size(-1))
    sampled = torch.multinomial(probs, 1, generator=generator).reshape(
        scaled.shape[:-1]
    )
    return sampled


def batch_topk_mask(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return a per-row boolean mask for the top ``k`` scores."""
    import torch

    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    max_k = int(k.max().item()) if k.numel() else 0
    if max_k == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    _, indices = torch.topk(scores, max_k, dim=1)
    ranks = torch.arange(max_k, device=scores.device).unsqueeze(0)
    active = ranks < k.to(scores.device).unsqueeze(1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, active)
