"""Categorical diffusion training-loss helpers for LayoutDM.

These helpers reproduce the discrete-diffusion variational objective used by the
original LayoutDM training loop: a KL term between the true and predicted
posteriors plus an adaptive auxiliary cross-entropy term, with importance
sampling of diffusion timesteps.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int


def mean_except_batch(
    x: Float[torch.Tensor, "batch ..."],
) -> Float[torch.Tensor, "batch"]:
    """Average every non-batch dimension.

    Args:
        x: Tensor whose leading dimension is the batch.

    Returns:
        Per-example mean over all trailing dimensions.

    Examples:
        >>> mean_except_batch(torch.ones(2, 3)).tolist()
        [1.0, 1.0]
    """
    return x.reshape(x.shape[0], -1).mean(dim=-1)


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
    square-root of the running squared-loss history.

    Args:
        batch_size: Number of timesteps to draw.
        num_timesteps: Total diffusion timesteps.
        lt_history: Running squared-loss history buffer.
        lt_count: Per-timestep observation-count buffer.
        generator: Optional random generator for deterministic draws.

    Returns:
        Sampled timesteps and their sampling probabilities.
    """
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
    """
    t = torch.randint(
        0, num_timesteps, (batch_size,), device=device, generator=generator
    ).long()
    pt = torch.ones_like(t).float() / num_timesteps
    return t, pt
