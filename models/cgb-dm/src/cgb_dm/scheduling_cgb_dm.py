"""Schedulers for CGB-DM training noising and DDIM sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from laygen.common import ConditionType
from laygen.schedulers.continuous import (
    make_beta_schedule as make_continuous_beta_schedule,
)


class CGBDMBetaSchedule(StrEnum):
    """Supported CGB-DM beta schedules."""

    cosine = auto()
    linear = auto()


@dataclass
class CGBDMSchedulerOutput(BaseOutput):
    """Output returned by one CGB-DM DDIM step.

    Attributes:
        prev_sample: Layout sample for the next denoising step.
        pred_original_sample: Estimated clean layout sample.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


def make_beta_schedule(
    schedule: CGBDMBetaSchedule | str,
    num_timesteps: int,
    *,
    start: float = 2.0e-4,
    end: float = 4.0e-2,
) -> torch.Tensor:
    """Create a CGB-DM beta schedule.

    Args:
        schedule: Schedule name.
        num_timesteps: Number of diffusion timesteps.
        start: Linear schedule start.
        end: Linear schedule end.

    Returns:
        Beta tensor.

    Raises:
        ValueError: If the schedule is unsupported.

    Examples:
        >>> make_beta_schedule("linear", 4).shape
        torch.Size([4])
    """
    canonical = CGBDMBetaSchedule(schedule)
    if canonical in {CGBDMBetaSchedule.linear, CGBDMBetaSchedule.cosine}:
        return make_continuous_beta_schedule(
            str(canonical),
            num_timesteps=num_timesteps,
            start=start,
            end=end,
        ).float()
    raise ValueError(f"Unsupported beta schedule: {schedule}")


def make_ddim_timesteps(
    *,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    mode: str = "uniform",
) -> np.ndarray:
    """Create DDIM timestep ids using CGB-DM discretization rules."""
    if mode == "uniform":
        stride = num_ddpm_timesteps // num_ddim_timesteps
        return np.asarray(list(range(0, num_ddpm_timesteps, stride)))
    if mode == "refine":
        return np.asarray(list(range(0, num_ddpm_timesteps, 2)))
    raise ValueError(f"Unsupported DDIM mode: {mode}")


class CGBDMScheduler(SchedulerMixin, ConfigMixin):
    """CGB-DM scheduler preserving separate training and sampling schedules.

    Args:
        num_train_timesteps: Number of DDPM training steps.
        ddim_num_steps: Default DDIM inference step count.
        train_beta_schedule: Schedule for training noising buffers.
        sampling_beta_schedule: Schedule for DDIM sampling buffers.
        eta: DDIM stochasticity.

    Examples:
        >>> scheduler = CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=2)
        >>> scheduler.ddim_timesteps.tolist()
        [0, 5]
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        ddim_num_steps: int = 100,
        train_beta_schedule: CGBDMBetaSchedule | str = CGBDMBetaSchedule.cosine,
        sampling_beta_schedule: CGBDMBetaSchedule | str = CGBDMBetaSchedule.linear,
        eta: float = 0.0,
    ) -> None:
        """Initialize noising and sampling buffers."""
        self.num_train_timesteps = int(num_train_timesteps)
        self.ddim_num_steps = int(ddim_num_steps)
        self.train_beta_schedule = str(CGBDMBetaSchedule(train_beta_schedule))
        self.sampling_beta_schedule = str(CGBDMBetaSchedule(sampling_beta_schedule))
        self.eta = float(eta)
        train_betas = make_beta_schedule(
            self.train_beta_schedule, self.num_train_timesteps
        ).float()
        train_alphas = 1.0 - train_betas
        self.train_alphas_cumprod = train_alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.train_alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - self.train_alphas_cumprod)
        sampling_betas = make_beta_schedule(
            self.sampling_beta_schedule, self.num_train_timesteps
        ).float()
        sampling_alphas = 1.0 - sampling_betas
        self.sampling_alphas_cumprod = sampling_alphas.cumprod(dim=0)
        self.timesteps = torch.empty(0, dtype=torch.long)
        self.set_timesteps(self.ddim_num_steps)

    def set_timesteps(
        self, num_inference_steps: int | None = None, device: torch.device | None = None
    ) -> None:
        """Set DDIM timesteps and derived sampling parameters."""
        steps = int(num_inference_steps or self.ddim_num_steps)
        ddim = make_ddim_timesteps(
            num_ddim_timesteps=steps,
            num_ddpm_timesteps=self.num_train_timesteps,
        )
        self.ddim_timesteps = torch.as_tensor(ddim, dtype=torch.long, device=device)
        self.timesteps = torch.flip(self.ddim_timesteps, dims=(0,))
        alphas = self.sampling_alphas_cumprod.to(device)
        self.ddim_alphas = alphas[self.ddim_timesteps]
        self.ddim_alphas_prev = torch.as_tensor(
            [alphas[0].item()] + alphas[self.ddim_timesteps[:-1]].tolist(),
            dtype=torch.float32,
            device=device,
        )
        self.ddim_sigmas = self.eta * torch.sqrt(
            (1 - self.ddim_alphas_prev)
            / (1 - self.ddim_alphas)
            * (1 - self.ddim_alphas / self.ddim_alphas_prev)
        )
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
        t_max: int | None = None,
    ) -> torch.Tensor:
        """Sample training timesteps."""
        high = int(t_max or self.num_train_timesteps - 1)
        return torch.randint(0, high, (batch_size,), device=device, generator=generator)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        fix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add training noise, preserving fixed channels when requested."""
        alphas = self.alphas_bar_sqrt.to(original_samples.device)
        one_minus = self.one_minus_alphas_bar_sqrt.to(original_samples.device)
        sqrt_alpha = torch.gather(alphas, 0, timesteps).reshape(-1, 1, 1)
        sqrt_one_minus = torch.gather(one_minus, 0, timesteps).reshape(-1, 1, 1)
        noised = sqrt_alpha * original_samples + sqrt_one_minus * noise
        if fix_mask is None:
            return noised
        return torch.where(fix_mask, original_samples, noised)

    def initial_sample(
        self,
        batch_size: int,
        seq_len: int,
        seq_dim: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create the initial DDIM sample."""
        return torch.randn(
            batch_size, seq_len, seq_dim, device=device, generator=generator
        )

    def condition_mask(
        self,
        layout: torch.Tensor,
        condition_type: ConditionType,
        *,
        completion_ratio: float = 0.2,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Build a channel-level mask for fixed conditioning values."""
        mask = torch.zeros_like(layout, dtype=torch.bool)
        num_labels = layout.shape[-1] - 4
        if condition_type is ConditionType.content_image:
            return mask
        if condition_type is ConditionType.label:
            mask[:, :, :num_labels] = True
            return mask
        if condition_type is ConditionType.label_size:
            mask[:, :, :num_labels] = True
            mask[:, :, num_labels + 2 : num_labels + 4] = True
            return mask
        if condition_type is ConditionType.completion:
            label_ids = layout[:, :, :num_labels].argmax(dim=-1)
            valid = label_ids != 0
            rand = torch.rand(valid.shape, device=layout.device, generator=generator)
            elem_mask = (rand <= completion_ratio) & valid
            return elem_mask.unsqueeze(-1).expand_as(layout)
        if condition_type is ConditionType.refinement:
            return torch.ones_like(mask)
        raise ValueError(f"Unsupported CGB-DM condition_type: {condition_type}")

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        index: int,
        generator: torch.Generator | None = None,
    ) -> CGBDMSchedulerOutput:
        """Take one DDIM reverse step."""
        del timestep
        alpha_t = self.ddim_alphas[index].to(sample.device)
        alpha_prev = self.ddim_alphas_prev[index].to(sample.device)
        sigma_t = self.ddim_sigmas[index].to(sample.device)
        sqrt_one_minus = self.ddim_sqrt_one_minus_alphas[index].to(sample.device)
        pred_original = (sample - sqrt_one_minus * model_output) / alpha_t.sqrt()
        direction = (1.0 - alpha_prev - sigma_t**2).sqrt() * model_output
        noise = sigma_t * torch.randn(
            sample.shape,
            dtype=sample.dtype,
            device=sample.device,
            generator=generator,
        )
        prev_sample = alpha_prev.sqrt() * pred_original + direction + noise
        return CGBDMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original,
        )
