"""DDIM-style scheduler utilities for LACE layout diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import assert_never

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class LaceSchedulerOutput(BaseOutput):
    """Output returned by one reverse diffusion step.

    Attributes:
        prev_sample: Sample for the next denoising iteration.
        pred_original_sample: Scheduler estimate of the clean layout tensor.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


class BetaSchedule(StrEnum):
    """Supported DDPM beta schedules."""

    linear = auto()
    const = auto()
    quad = auto()
    jsd = auto()
    sigmoid = auto()
    cosine = auto()
    cosine_reverse = auto()
    cosine_anneal = auto()


class DDIMDiscretization(StrEnum):
    """Supported DDIM timestep discretization methods."""

    uniform = auto()
    quad = auto()
    new = auto()


def normalize_beta_schedule(schedule: BetaSchedule | str) -> BetaSchedule:
    """Normalize a beta schedule value.

    Args:
        schedule: Schedule enum or string value.

    Returns:
        Canonical beta schedule enum.

    Raises:
        ValueError: If the schedule is unsupported.
    """
    if isinstance(schedule, BetaSchedule):
        return schedule
    try:
        return BetaSchedule(schedule)
    except ValueError as exc:
        raise ValueError(f"Unsupported beta schedule: {schedule}") from exc


def normalize_ddim_discretization(
    method: DDIMDiscretization | str,
) -> DDIMDiscretization:
    """Normalize a DDIM timestep discretization method.

    Args:
        method: Method enum or string value.

    Returns:
        Canonical DDIM discretization enum.

    Raises:
        ValueError: If the method is unsupported.
    """
    if isinstance(method, DDIMDiscretization):
        return method
    try:
        return DDIMDiscretization(method)
    except ValueError as exc:
        raise ValueError(f"Unsupported ddim discretization: {method}") from exc


def make_beta_schedule(
    schedule: BetaSchedule | str = BetaSchedule.cosine,
    num_timesteps: int = 1000,
    start: float = 0.0001,
    end: float = 0.02,
) -> torch.Tensor:
    """Create the beta schedule used by the LACE diffusion process.

    Args:
        schedule: Schedule enum or string value.
        num_timesteps: Number of training timesteps.
        start: Initial beta value for schedules that use a range.
        end: Final beta value for schedules that use a range.

    Returns:
        One-dimensional beta tensor.

    Raises:
        ValueError: If the schedule is unsupported.
    """
    canonical = normalize_beta_schedule(schedule)
    if canonical is BetaSchedule.linear:
        return torch.linspace(start, end, num_timesteps)
    if canonical is BetaSchedule.const:
        return end * torch.ones(num_timesteps)
    if canonical is BetaSchedule.quad:
        return torch.linspace(start**0.5, end**0.5, num_timesteps) ** 2
    if canonical is BetaSchedule.jsd:
        return 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    if canonical is BetaSchedule.sigmoid:
        betas = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(betas) * (end - start) + start
    if canonical is BetaSchedule.cosine:
        max_beta = 0.999
        cosine_s = 0.008
        return torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
    if canonical is BetaSchedule.cosine_reverse:
        max_beta = 0.999
        cosine_s = 0.008
        return torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
    if canonical is BetaSchedule.cosine_anneal:
        return torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    assert_never(canonical)


def make_ddim_timesteps(
    method: DDIMDiscretization | str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
) -> np.ndarray:
    """Create one-indexed DDIM timesteps matching the vendor implementation.

    Args:
        method: Discretization enum or string value.
        num_ddim_timesteps: Number of inference timesteps.
        num_ddpm_timesteps: Number of training timesteps.

    Returns:
        NumPy array of one-indexed timesteps.

    Raises:
        ValueError: If the method is unsupported.
    """
    canonical = normalize_ddim_discretization(method)
    if canonical is DDIMDiscretization.uniform:
        c = num_ddpm_timesteps // num_ddim_timesteps
        timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif canonical is DDIMDiscretization.quad:
        timesteps = (
            np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2
        ).astype(int)
    elif canonical is DDIMDiscretization.new:
        c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
        timesteps = np.asarray(
            list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c))
        )
    else:
        assert_never(canonical)
    return timesteps + 1


class LaceScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for the converted LACE diffusion process.

    Args:
        num_train_timesteps: Number of timesteps used during training.
        beta_schedule: Beta schedule enum or string value.
        ddim_num_steps: Default number of inference steps.
        ddim_discretize: DDIM discretization enum or string value.
        eta: Stochasticity parameter used by DDIM.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        beta_schedule: BetaSchedule | str = BetaSchedule.cosine,
        ddim_num_steps: int = 100,
        ddim_discretize: DDIMDiscretization | str = DDIMDiscretization.uniform,
        eta: float = 0.0,
    ) -> None:
        """Initialize scheduler state and default timesteps."""
        self.num_train_timesteps = num_train_timesteps
        canonical_beta = normalize_beta_schedule(beta_schedule)
        canonical_ddim = normalize_ddim_discretization(ddim_discretize)
        self.beta_schedule = str(canonical_beta)
        self.ddim_num_steps = ddim_num_steps
        self.ddim_discretize = str(canonical_ddim)
        self.eta = eta
        betas = make_beta_schedule(
            canonical_beta,
            num_timesteps=num_train_timesteps,
            start=0.0001,
            end=0.02,
        ).float()
        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.timesteps = torch.empty(0, dtype=torch.long)
        self.set_timesteps(ddim_num_steps)

    def set_timesteps(
        self, num_inference_steps: int | None = None, device: torch.device | None = None
    ) -> None:
        """Set the inference timesteps.

        Args:
            num_inference_steps: Number of denoising steps. Uses the configured
                default when omitted.
            device: Optional device for scheduler tensors.
        """
        steps = num_inference_steps or self.ddim_num_steps
        ddim = make_ddim_timesteps(
            self.ddim_discretize, steps, self.num_train_timesteps
        )
        self.ddim_timesteps = torch.as_tensor(ddim, dtype=torch.long, device=device)
        self.timesteps = torch.flip(self.ddim_timesteps, dims=(0,))
        alphas_cumprod = self.alphas_cumprod.to(device)
        self.ddim_alphas = alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = torch.as_tensor(
            [alphas_cumprod[0].item()]
            + alphas_cumprod[self.ddim_timesteps[:-1]].tolist(),
            dtype=torch.float32,
            device=device,
        )
        self.ddim_sigmas = self.eta * torch.sqrt(
            (1 - self.ddim_alphas_prev)
            / (1 - self.ddim_alphas)
            * (1 - self.ddim_alphas / self.ddim_alphas_prev)
        )
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add forward-process noise to clean samples.

        Args:
            original_samples: Clean layout tensor.
            noise: Noise tensor with the same shape.
            timesteps: Per-sample timestep ids.

        Returns:
            Noisy samples at the requested timesteps.
        """
        alphas = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha = torch.gather(alphas.sqrt(), 0, timesteps).reshape(-1, 1, 1)
        sqrt_one_minus = torch.gather(torch.sqrt(1.0 - alphas), 0, timesteps).reshape(
            -1, 1, 1
        )
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def initial_sample(
        self,
        batch_size: int,
        seq_len: int,
        seq_dim: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """Create the initial denoising sample.

        Args:
            batch_size: Number of layouts.
            seq_len: Number of elements per layout.
            seq_dim: Number of channels per element.
            device: Device for the output tensor.
            generator: Optional torch generator.
            stochastic: Whether to sample noise or return zeros.

        Returns:
            Initial sample tensor.
        """
        if not stochastic:
            return torch.zeros(batch_size, seq_len, seq_dim, device=device)
        return torch.randn(
            batch_size, seq_len, seq_dim, device=device, generator=generator
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        index: int,
        generator: torch.Generator | None = None,
    ) -> LaceSchedulerOutput:
        """Take one reverse diffusion step.

        Args:
            model_output: Predicted noise from the denoiser.
            timestep: Public timestep tensor, kept for scheduler compatibility.
            sample: Current sample.
            index: Index into the scheduler timestep buffers.
            generator: Optional torch generator for stochastic DDIM noise.

        Returns:
            Previous sample and predicted clean sample.
        """
        del timestep
        alpha_t = self.ddim_alphas[index].to(sample.device)
        alpha_prev = self.ddim_alphas_prev[index].to(sample.device)
        sigma_t = self.ddim_sigmas[index].to(sample.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas[index].to(sample.device)
        pred_original = (sample - sqrt_one_minus * model_output) / alpha_t.sqrt()
        direction = (1.0 - alpha_prev - sigma_t**2).sqrt() * model_output
        noise = sigma_t * torch.randn(
            sample.shape,
            dtype=sample.dtype,
            device=sample.device,
            generator=generator,
        )
        prev_sample = alpha_prev.sqrt() * pred_original + direction + noise
        return LaceSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original
        )

    def refinement_indices(self, max_timestep: int = 201) -> list[int]:
        """Return scheduler indices used by LACE refinement sampling.

        Args:
            max_timestep: Maximum one-indexed DDIM timestep included.

        Returns:
            Descending list of scheduler buffer indices.
        """
        total = int(torch.sum(self.ddim_timesteps <= max_timestep).item())
        return list(range(total - 1, -1, -1))
