"""DDIM-style scheduler utilities for LACE layout diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from laygen.schedulers.continuous import (
    BetaSchedule,
    DDIMDiscretization,
    make_beta_schedule,
    make_ddim_timesteps,
    normalize_beta_schedule,
    normalize_ddim_discretization,
)


@dataclass
class LaceSchedulerOutput(BaseOutput):
    """Output returned by one reverse diffusion step.

    Attributes:
        prev_sample: Sample for the next denoising iteration.
        pred_original_sample: Scheduler estimate of the clean layout tensor.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


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
