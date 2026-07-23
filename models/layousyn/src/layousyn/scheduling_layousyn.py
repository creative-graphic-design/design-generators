"""Scheduler preserving LayouSyn's OpenAI Gaussian/DDIM diffusion math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from diffusers import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from laygen.schedulers.continuous import get_layousyn_beta_schedule


@dataclass
class LayouSynSchedulerOutput(BaseOutput):
    """Output returned by a LayouSyn scheduler step."""

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class LayouSynScheduler(SchedulerMixin, ConfigMixin):
    """OpenAI-style Gaussian scheduler for LayouSyn layout tensors."""

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_train_timesteps: int = 100,
        beta_schedule: Literal["linear", "squaredcos_cap_v2"] = "linear",
        alpha_scale: float = 1.0,
        prediction_type: Literal["epsilon"] = "epsilon",
        variance_type: Literal["learned_range"] = "learned_range",
        sampling_type: Literal["ddim", "ddpm"] = "ddim",
    ) -> None:
        """Initialize scheduler buffers."""
        if prediction_type != "epsilon":
            raise ValueError("LayouSyn only supports prediction_type='epsilon'")
        if variance_type != "learned_range":
            raise ValueError("LayouSyn only supports variance_type='learned_range'")
        self.num_train_timesteps = num_train_timesteps
        self.sampling_type = sampling_type
        self.original_betas = get_layousyn_beta_schedule(
            beta_schedule, num_train_timesteps, alpha_scale=alpha_scale
        )
        self.timestep_map = torch.arange(num_train_timesteps, dtype=torch.long)
        self.model_timesteps = torch.empty(0, dtype=torch.long)
        self.timesteps = torch.empty(0, dtype=torch.long)
        self._set_betas(self.original_betas)
        self.set_timesteps(num_train_timesteps)

    def _set_betas(self, betas: torch.Tensor) -> None:
        """Set derived buffers from the active beta sequence."""
        self.betas = betas.double()
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )
        self.posterior_log_variance_clipped = clipped
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Set descending denoising timesteps with reference respacing."""
        steps = num_inference_steps or self.num_train_timesteps
        if steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")
        use_timesteps = _space_timesteps(self.num_train_timesteps, steps)
        base_alphas = torch.cumprod(1.0 - self.original_betas.double(), dim=0)
        last_alpha_cumprod = torch.tensor(1.0, dtype=torch.float64)
        new_betas = []
        timestep_map = []
        for index, alpha_cumprod in enumerate(base_alphas):
            if index in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(index)
        self.timestep_map = torch.tensor(timestep_map, dtype=torch.long, device=device)
        self._set_betas(torch.stack(new_betas))
        self.timesteps = torch.arange(
            len(timestep_map) - 1, -1, -1, dtype=torch.long, device=device
        )
        self.model_timesteps = self.timestep_map[self.timesteps]

    def initial_sample(
        self,
        batch_size: int,
        seq_len: int,
        channels: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create initial Gaussian noise."""
        return torch.randn(
            batch_size,
            seq_len,
            channels,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add forward-process noise to clean samples."""
        alphas = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha = torch.gather(alphas.sqrt(), 0, timesteps).reshape(-1, 1, 1)
        sqrt_one_minus = torch.gather(torch.sqrt(1.0 - alphas), 0, timesteps).reshape(
            -1, 1, 1
        )
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        eta: float = 0.0,
        clip_denoised: bool = False,
        sampling_type: Literal["ddim", "ddpm"] | None = None,
        return_dict: bool = True,
    ) -> LayouSynSchedulerOutput | tuple[torch.Tensor]:
        """Take one reverse diffusion step."""
        mode = sampling_type or self.sampling_type
        eps, model_var_values = torch.split(model_output, sample.shape[1], dim=1)
        pred_xstart = self._predict_xstart_from_eps(sample, timestep, eps)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        if mode == "ddim":
            prev_sample = self._ddim_step(
                sample, timestep, pred_xstart, generator=generator, eta=eta
            )
        elif mode == "ddpm":
            prev_sample = self._ddpm_step(
                sample, timestep, pred_xstart, model_var_values, generator=generator
            )
        else:
            raise ValueError(f"Unsupported sampling_type: {mode}")
        output = LayouSynSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_xstart
        )
        if not return_dict:
            return (output.prev_sample,)
        return output

    def _ddim_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        pred_xstart: torch.Tensor,
        *,
        generator: torch.Generator | None,
        eta: float,
    ) -> torch.Tensor:
        eps = self._predict_eps_from_xstart(sample, timestep, pred_xstart)
        alpha_bar = self._extract(self.alphas_cumprod, timestep, sample.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, timestep, sample.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn(
            sample.shape, dtype=sample.dtype, device=sample.device, generator=generator
        )
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (timestep != 0).float().view(-1, 1, 1)
        return mean_pred + nonzero_mask * sigma * noise

    def _ddpm_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        pred_xstart: torch.Tensor,
        model_var_values: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        mean = (
            self._extract(self.posterior_mean_coef1, timestep, sample.shape)
            * pred_xstart
        )
        mean = (
            mean
            + self._extract(self.posterior_mean_coef2, timestep, sample.shape) * sample
        )
        min_log = self._extract(
            self.posterior_log_variance_clipped, timestep, sample.shape
        )
        max_log = self._extract(torch.log(self.betas), timestep, sample.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        noise = torch.randn(
            sample.shape, dtype=sample.dtype, device=sample.device, generator=generator
        )
        nonzero_mask = (timestep != 0).float().view(-1, 1, 1)
        return mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def _predict_xstart_from_eps(
        self, sample: torch.Tensor, timestep: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return self._extract(
            self.sqrt_recip_alphas_cumprod, timestep, sample.shape
        ) * sample - (
            self._extract(self.sqrt_recipm1_alphas_cumprod, timestep, sample.shape)
            * eps
        )

    def _predict_eps_from_xstart(
        self, sample: torch.Tensor, timestep: torch.Tensor, pred_xstart: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timestep, sample.shape)
            * sample
            - pred_xstart
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, timestep, sample.shape)

    @staticmethod
    def _extract(
        values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
    ) -> torch.Tensor:
        out = values.to(device=timesteps.device).gather(0, timesteps).float()
        while out.ndim < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out


def _space_timesteps(num_timesteps: int, section_count: int) -> set[int]:
    """Match reference ``space_timesteps`` for one section count."""
    if num_timesteps < section_count:
        raise ValueError(
            f"cannot divide section of {num_timesteps} into {section_count}"
        )
    frac_stride = (
        1.0 if section_count <= 1 else (num_timesteps - 1) / (section_count - 1)
    )
    cur_idx = 0.0
    steps = []
    for _ in range(section_count):
        steps.append(round(cur_idx))
        cur_idx += frac_stride
    return set(steps)
