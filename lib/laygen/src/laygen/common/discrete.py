from __future__ import annotations

import torch
import torch.nn.functional as F

LOG_EPS = -70.0


def index_to_log_onehot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert token ids to channel-first log one-hot probabilities.

    Args:
        input_ids: Integer token tensor shaped `(batch, ...)`.
        vocab_size: Size of the token vocabulary.

    Returns:
        Log one-hot tensor shaped `(batch, vocab_size, ...)`.

    Raises:
        ValueError: If any token id is greater than or equal to `vocab_size`.

    Examples:
        >>> import torch
        >>> index_to_log_onehot(torch.tensor([[0, 1]]), 3).shape
        torch.Size([1, 3, 2])
    """

    if input_ids.numel() and input_ids.max().item() >= vocab_size:
        raise ValueError(
            f"input id {input_ids.max().item()} exceeds vocab_size {vocab_size}"
        )
    onehot = F.one_hot(input_ids.long(), vocab_size)
    order = (0, -1) + tuple(range(1, input_ids.ndim))
    return torch.log(onehot.permute(order).float().clamp(min=1e-30))


def log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    """Convert channel-first log one-hot probabilities to token ids.

    Args:
        log_x: Tensor shaped `(batch, vocab_size, ...)`.

    Returns:
        Integer tensor containing the maximum-probability token ids.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> log_onehot_to_index(index_to_log_onehot(torch.tensor([[0, 1]]), 3))
        tensor([[0, 1]])
    """

    return log_x.argmax(dim=1)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two log-space tensors with a numerically stable transform.

    Args:
        a: First log-space tensor.
        b: Second log-space tensor.

    Returns:
        Elementwise `log(exp(a) + exp(b))`.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> log_add_exp(torch.zeros(1), torch.zeros(1)).round(decimals=4)
        tensor([0.6931])
    """

    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(
    values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    """Gather per-timestep values and reshape them for broadcasting.

    Args:
        values: One-dimensional schedule tensor.
        timesteps: Batch timestep tensor.
        broadcast_shape: Target tensor shape whose rank determines output rank.

    Returns:
        Tensor shaped `(batch, 1, ..., 1)` for broadcasting with the target rank.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> extract(torch.arange(4), torch.tensor([2]), torch.Size([1, 3, 5])).shape
        torch.Size([1, 1, 1])
    """

    batch, *_ = timesteps.shape
    out = values.to(timesteps.device).gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def gumbel_noise_like(
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Gumbel noise with the same shape, dtype, and device as a tensor.

    Args:
        x: Reference tensor.
        generator: Optional PyTorch generator for reproducible sampling.

    Returns:
        Gumbel noise tensor matching `x.shape`.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> gumbel_noise_like(torch.zeros(2), generator=torch.Generator().manual_seed(0)).shape
        torch.Size([2])
    """

    uniform = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return -torch.log(-torch.log(uniform + 1e-30) + 1e-30)


def log_sample_categorical(
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample categorical token ids from log probabilities with Gumbel noise.

    Args:
        logits: Channel-first logit tensor.
        generator: Optional PyTorch generator for reproducible sampling.

    Returns:
        Sampled ids from the channel dimension.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> log_sample_categorical(torch.zeros(1, 3, 2), generator=torch.Generator().manual_seed(0)).shape
        torch.Size([1, 2])
    """

    return (logits + gumbel_noise_like(logits, generator=generator)).argmax(dim=1)


def top_k_logits(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Mask logits outside the top-k values along a dimension.

    Args:
        logits: Input logit tensor.
        k: Number of values to keep. Non-positive values keep all logits.
        dim: Dimension to rank.

    Returns:
        Tensor with non-top-k entries replaced by `LOG_EPS`.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> top_k_logits(torch.tensor([[1.0, 2.0, 3.0]]), 2).shape
        torch.Size([1, 3])
    """

    if k <= 0 or k >= logits.size(dim):
        return logits
    values = torch.topk(logits, k, dim=dim).values
    threshold = values.select(dim, k - 1).unsqueeze(dim)
    return logits.masked_fill(logits < threshold, LOG_EPS)


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
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
    sampling: str = "random",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample token ids from logits using the named sampling strategy.

    Args:
        logits: Logit tensor with vocabulary on the final dimension.
        sampling: Sampling strategy name.
        temperature: Temperature applied before stochastic sampling.
        top_k: Optional top-k cutoff for `top_k` and `top_k_top_p`.
        top_p: Optional nucleus cutoff for `top_p` and `top_k_top_p`.
        generator: Optional PyTorch generator for reproducible sampling.

    Returns:
        Sampled token ids with the final vocabulary dimension removed.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> sample_categorical(torch.zeros(1, 2, 3), sampling="deterministic").shape
        torch.Size([1, 2])
    """

    if sampling == "deterministic":
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    if sampling in {"top_k", "top_k_top_p"} and top_k is not None:
        scaled = top_k_logits(scaled, top_k, dim=-1)
    if sampling in {"top_p", "top_k_top_p"} and top_p is not None:
        scaled = _top_p_logits(scaled, top_p)
    if sampling == "gumbel":
        return (scaled + gumbel_noise_like(scaled, generator=generator)).argmax(dim=-1)
    probs = scaled.softmax(dim=-1).reshape(-1, scaled.size(-1))
    sampled = torch.multinomial(probs, 1, generator=generator).reshape(
        scaled.shape[:-1]
    )
    return sampled


def batch_topk_mask(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Build a per-row mask for the top `k` scores.

    Args:
        scores: Rank-2 score tensor shaped `(batch, items)`.
        k: Number of active positions to keep for each batch row.

    Returns:
        Boolean mask with the same shape as `scores`.

    Raises:
        ValueError: If `scores` is not rank-2.

    Examples:
        >>> import torch
        >>> batch_topk_mask(torch.tensor([[0.1, 0.9]]), torch.tensor([1])).tolist()
        [[False, True]]
    """

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
