"""Discrete diffusion tensor utilities shared by layout generators."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Final, assert_never

from jaxtyping import Bool, Float, Int

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        pass

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


def index_to_log_onehot(
    input_ids: Int[torch.Tensor, "batch ..."], vocab_size: int
) -> Float[torch.Tensor, "batch vocab ..."]:
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


def log_onehot_to_index(
    log_x: Float[torch.Tensor, "batch vocab ..."],
) -> Int[torch.Tensor, "batch ..."]:
    """Convert log one-hot tensors back to categorical ids."""
    return log_x.argmax(dim=1)


def multinomial_kl(
    log_prob1: Float[torch.Tensor, "batch vocab tokens"],
    log_prob2: Float[torch.Tensor, "batch vocab tokens"],
) -> Float[torch.Tensor, "batch tokens"]:
    """Categorical KL divergence summed over the vocabulary dimension.

    Args:
        log_prob1: Log probabilities of the reference distribution.
        log_prob2: Log probabilities of the compared distribution.

    Returns:
        Per-token KL divergence with the vocabulary dimension reduced.

    Examples:
        >>> import torch
        >>> a = torch.log(torch.tensor([[[1.0], [0.0]]]).clamp_min(1e-30))
        >>> float(multinomial_kl(a, a).sum())
        0.0
    """
    return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)


def log_categorical(
    log_x_start: Float[torch.Tensor, "batch vocab tokens"],
    log_prob: Float[torch.Tensor, "batch vocab tokens"],
) -> Float[torch.Tensor, "batch tokens"]:
    """Categorical log-likelihood of ``log_x_start`` under ``log_prob``.

    Args:
        log_x_start: Log one-hot targets.
        log_prob: Predicted log probabilities.

    Returns:
        Per-token log-likelihood with the vocabulary dimension reduced.

    Examples:
        >>> import torch
        >>> target = torch.log(torch.tensor([[[1.0], [0.0]]]).clamp_min(1e-30))
        >>> probs = torch.log(torch.tensor([[[0.25], [0.75]]]))
        >>> log_categorical(target, probs).shape
        torch.Size([1, 1])
    """
    return (log_x_start.exp() * log_prob).sum(dim=1)


def sample_time_importance(
    batch_size: int,
    *,
    num_timesteps: int,
    lt_history: Float[torch.Tensor, "timesteps"],
    lt_count: Float[torch.Tensor, "timesteps"],
    generator: torch.Generator | None = None,
) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """Sample diffusion timesteps with loss-aware importance sampling.

    Until every timestep bucket has more than ten observations the sampler falls
    back to a uniform draw. Afterwards timesteps are drawn proportionally to the
    square root of the running squared-loss history.

    Args:
        batch_size: Number of timesteps to draw.
        num_timesteps: Total diffusion timesteps.
        lt_history: Running squared-loss history buffer.
        lt_count: Per-timestep observation-count buffer.
        generator: Optional random generator for deterministic draws.

    Returns:
        Sampled timesteps and their sampling probabilities.

    Examples:
        >>> import torch
        >>> hist = torch.arange(1, 5, dtype=torch.float32)
        >>> count = torch.full((4,), 11.0)
        >>> gen = torch.Generator().manual_seed(0)
        >>> t, pt = sample_time_importance(
        ...     2, num_timesteps=4, lt_history=hist, lt_count=count, generator=gen
        ... )
        >>> t.shape, pt.shape
        (torch.Size([2]), torch.Size([2]))
    """
    import torch

    device = lt_history.device
    if not bool((lt_count > 10).all()):
        return sample_time_uniform(
            batch_size,
            num_timesteps=num_timesteps,
            device=device,
            generator=generator,
        )

    lt_sqrt = torch.sqrt(lt_history + 1e-10) + 0.0001
    lt_sqrt[0] = lt_sqrt[1]
    pt_all = lt_sqrt / lt_sqrt.sum()
    t = torch.multinomial(
        pt_all, num_samples=batch_size, replacement=True, generator=generator
    )
    pt = pt_all.gather(dim=0, index=t)
    return t, pt


def sample_time_uniform(
    batch_size: int,
    *,
    num_timesteps: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """Sample diffusion timesteps uniformly.

    Args:
        batch_size: Number of timesteps to draw.
        num_timesteps: Total diffusion timesteps.
        device: Device for the sampled tensors.
        generator: Optional random generator for deterministic draws.

    Returns:
        Sampled timesteps and their uniform sampling probabilities.

    Examples:
        >>> import torch
        >>> gen = torch.Generator().manual_seed(0)
        >>> t, pt = sample_time_uniform(
        ...     2, num_timesteps=4, device=torch.device("cpu"), generator=gen
        ... )
        >>> t.shape, pt.tolist()
        (torch.Size([2]), [0.25, 0.25])
    """
    import torch

    t = torch.randint(
        0, num_timesteps, (batch_size,), device=device, generator=generator
    ).long()
    pt = torch.ones_like(t).float() / num_timesteps
    return t, pt


def update_loss_history(
    kl_loss: Float[torch.Tensor, "batch"],
    t: Int[torch.Tensor, "batch"],
    lt_history: Float[torch.Tensor, "timesteps"],
    lt_count: Float[torch.Tensor, "timesteps"],
) -> None:
    """Update squared-loss history buffers in place.

    The update matches the D3PM-style training loop used by LayoutDM and
    LayoutDiffusion: each sampled timestep receives ``0.1 * loss**2 + 0.9``
    times the previous bucket value, and the observation count is incremented.

    Args:
        kl_loss: Per-example KL or decoder loss for the sampled timesteps.
        t: Sampled timestep ids for each example.
        lt_history: Running squared-loss history buffer to mutate.
        lt_count: Per-timestep observation count buffer to mutate.

    Returns:
        None. The history and count tensors are updated in place.

    Examples:
        >>> import torch
        >>> history = torch.zeros(3)
        >>> count = torch.zeros(3)
        >>> update_loss_history(torch.tensor([2.0]), torch.tensor([1]), history, count)
        >>> history.tolist(), count.tolist()
        ([0.0, 0.4000000059604645, 0.0], [0.0, 1.0, 0.0])
    """
    import torch

    lt2 = kl_loss.pow(2)
    lt2_prev = lt_history.gather(dim=0, index=t)
    new_history = (0.1 * lt2 + 0.9 * lt2_prev).detach()
    lt_history.scatter_(dim=0, index=t, src=new_history)
    lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(lt2))


def log_add_exp(
    a: Float[torch.Tensor, "..."], b: Float[torch.Tensor, "..."]
) -> Float[torch.Tensor, "..."]:
    """Compute a numerically stable elementwise ``log(exp(a) + exp(b))``."""
    import torch

    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(
    values: Float[torch.Tensor, "timesteps"],
    timesteps: Int[torch.Tensor, "batch"],
    broadcast_shape: torch.Size,
) -> Float[torch.Tensor, "batch ..."]:
    """Gather timestep values and reshape them for broadcast operations."""
    batch, *_ = timesteps.shape
    out = values.to(timesteps.device).gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def gumbel_noise_like(
    x: Float[torch.Tensor, "..."],
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "..."]:
    """Sample Gumbel noise with the same shape, dtype, and device as ``x``."""
    import torch

    uniform = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return -torch.log(-torch.log(uniform + 1e-30) + 1e-30)


def log_sample_categorical(
    logits: Float[torch.Tensor, "batch vocab ..."],
    *,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, "batch ..."]:
    """Sample categorical ids from log probabilities with Gumbel-max."""
    return (logits + gumbel_noise_like(logits, generator=generator)).argmax(dim=1)


def top_k_logits(
    logits: Float[torch.Tensor, "... vocab"], k: int, dim: int = -1
) -> Float[torch.Tensor, "... vocab"]:
    """Mask logits outside the top-k entries along ``dim``."""
    import torch

    if k <= 0 or k >= logits.size(dim):
        return logits

    values = torch.topk(logits, k, dim=dim).values
    threshold = values.select(dim, k - 1).unsqueeze(dim)
    return logits.masked_fill(logits < threshold, LOG_EPS)


def _top_p_logits(
    logits: Float[torch.Tensor, "... vocab"], top_p: float
) -> Float[torch.Tensor, "... vocab"]:
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
    logits: Float[torch.Tensor, "... vocab"],
    *,
    sampling: SamplingMode | str = SamplingMode.random,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, "batch ..."]:
    """Sample categorical ids from logits using LayoutDM sampling modes.

    Args:
        logits: torch.Tensor whose last dimension is the categorical vocabulary.
        sampling: Sampling mode name.
        temperature: Positive temperature used before random sampling.
        top_k: Number of logits retained for top-k modes.
        top_p: Cumulative probability retained for top-p modes.
        generator: Optional torch generator for deterministic sampling.

    Returns:
        torch.Tensor of sampled ids with shape ``logits.shape[:-1]``.

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


def batch_topk_mask(
    scores: Float[torch.Tensor, "batch candidates"], k: Int[torch.Tensor, "batch"]
) -> Bool[torch.Tensor, "batch candidates"]:
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
