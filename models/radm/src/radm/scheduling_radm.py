"""Proposal-box diffusion scheduler for RADM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, SchedulerMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from jaxtyping import Float, Int


@dataclass
class RADMSchedulerOutput(BaseOutput):
    """Output of one RADM scheduler step."""

    prev_sample: Float[torch.Tensor, "batch proposals 4"]
    pred_original_sample: Float[torch.Tensor, "batch proposals 4"]
    noise: Float[torch.Tensor, "batch proposals 4"] | None = None


class RADMScheduler(SchedulerMixin, ConfigMixin):
    """DDIM-style scheduler for normalized proposal boxes.

    Args:
        num_train_timesteps: Number of training diffusion timesteps.
        num_inference_steps: Default inference step count.
        beta_schedule: Schedule name. Only ``"cosine"`` is supported.
        eta: DDIM stochasticity parameter.
        prediction_type: Model prediction type.

    Examples:
        >>> scheduler = RADMScheduler(num_train_timesteps=10, num_inference_steps=3)
        >>> scheduler.set_timesteps(3)
        >>> scheduler.timesteps.tolist()
        [9, 5, 2]
    """

    config_name: str = "scheduler_config.json"
    order: int = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_schedule: Literal["cosine"] = "cosine",
        eta: float = 1.0,
        prediction_type: Literal["sample"] = "sample",
    ) -> None:
        """Initialize RADM scheduler metadata."""
        if beta_schedule != "cosine":
            raise ValueError("RADM only supports beta_schedule='cosine'")
        if prediction_type != "sample":
            raise ValueError("RADM only supports prediction_type='sample'")
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.eta = float(eta)
        betas = cosine_beta_schedule(self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod_prev = F.pad(
            torch.cumprod(alphas, dim=0)[:-1], (1, 0), value=1.0
        )
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Set reversed inference timesteps.

        Args:
            num_inference_steps: Optional step count.
            device: Optional target device.
        """
        steps = int(num_inference_steps or self.config.num_inference_steps)
        raw = torch.linspace(-1, self.config.num_train_timesteps - 1, steps + 1)
        self.timesteps = raw.long().flip(0)[:-1].to(device)
        self.num_inference_steps = steps

    def sample_initial_proposals(
        self,
        *,
        batch_size: int,
        num_proposals: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Sample initial normalized ``xyxy`` proposal boxes.

        Args:
            batch_size: Number of examples.
            num_proposals: Number of proposal boxes.
            generator: Optional random generator.
            device: Output device.
            dtype: Output dtype.

        Returns:
            Sorted normalized boxes in ``xyxy`` order.
        """
        noise = torch.randn(
            (batch_size, num_proposals, 4),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        points = noise.sigmoid()
        left_top = torch.minimum(points[..., :2], points[..., 2:])
        right_bottom = torch.maximum(points[..., :2], points[..., 2:])
        return torch.cat((left_top, right_bottom), dim=-1)

    def scale_model_input(
        self,
        sample: Float[torch.Tensor, "batch proposals 4"],
        timestep: Float[torch.Tensor, ""] | float,
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Return sample unchanged for Diffusers scheduler compatibility."""
        del timestep
        return sample

    def predict_noise_from_start(
        self,
        sample: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, "batch"] | Int[torch.Tensor, ""] | int,
        pred_original_sample: Float[torch.Tensor, "batch proposals 4"],
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Return the noise implied by ``x_t`` and predicted ``x_0``.

        Args:
            sample: Current latent sample.
            timestep: Training timestep indices.
            pred_original_sample: Predicted denoised sample.

        Returns:
            Predicted noise tensor.
        """
        t = _as_batch_timesteps(timestep, sample.shape[0], sample.device)
        sqrt_recip = _extract(self.sqrt_recip_alphas_cumprod, t, sample.shape).to(
            device=sample.device, dtype=sample.dtype
        )
        sqrt_recipm1 = _extract(self.sqrt_recipm1_alphas_cumprod, t, sample.shape).to(
            device=sample.device, dtype=sample.dtype
        )
        return (sqrt_recip * sample - pred_original_sample) / sqrt_recipm1

    def q_sample(
        self,
        x_start: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, "batch"] | Int[torch.Tensor, ""] | int,
        noise: Float[torch.Tensor, "batch proposals 4"] | None = None,
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Diffuse ``x_start`` to ``x_t`` with the RADM forward process.

        Args:
            x_start: Initial clean sample.
            timestep: Training timestep indices.
            noise: Optional fixed noise tensor.

        Returns:
            Noised sample at ``timestep``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        t = _as_batch_timesteps(timestep, x_start.shape[0], x_start.device)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(
            device=x_start.device, dtype=x_start.dtype
        )
        sqrt_one_minus = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ).to(device=x_start.device, dtype=x_start.dtype)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    @overload
    def step(
        self,
        model_output: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, ""] | int,
        sample: Float[torch.Tensor, "batch proposals 4"],
        *,
        generator: torch.Generator | None = None,
        return_dict: Literal[True] = True,
    ) -> RADMSchedulerOutput: ...

    @overload
    def step(
        self,
        model_output: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, ""] | int,
        sample: Float[torch.Tensor, "batch proposals 4"],
        *,
        generator: torch.Generator | None = None,
        return_dict: Literal[False],
    ) -> tuple[
        Float[torch.Tensor, "batch proposals 4"],
        Float[torch.Tensor, "batch proposals 4"],
    ]: ...

    def step(
        self,
        model_output: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, ""] | int,
        sample: Float[torch.Tensor, "batch proposals 4"],
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> (
        RADMSchedulerOutput
        | tuple[
            Float[torch.Tensor, "batch proposals 4"],
            Float[torch.Tensor, "batch proposals 4"],
        ]
    ):
        """Advance one reverse-diffusion step.

        Args:
            model_output: Predicted denoised sample.
            timestep: Current training timestep.
            sample: Current proposal sample.
            generator: Optional generator for DDIM noise.
            return_dict: Whether to return a dataclass.

        Returns:
            Scheduler output dataclass or tuple.
        """
        timestep_i = int(torch.as_tensor(timestep).item())
        prev_timestep = _previous_timestep(self.timesteps, timestep_i)
        if prev_timestep < 0:
            prev_sample = model_output
            noise = None
        else:
            alpha_prod_t = self.alphas_cumprod[timestep_i].to(
                device=model_output.device, dtype=model_output.dtype
            )
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(
                device=model_output.device, dtype=model_output.dtype
            )
            pred_noise = self.predict_noise_from_start(sample, timestep_i, model_output)
            sigma = (
                float(self.config.eta)
                * (
                    (1.0 - alpha_prod_t / alpha_prod_t_prev)
                    * (1.0 - alpha_prod_t_prev)
                    / (1.0 - alpha_prod_t)
                ).sqrt()
            )
            direction = (1.0 - alpha_prod_t_prev - sigma**2).sqrt() * pred_noise
            noise = None
            prev_sample = alpha_prod_t_prev.sqrt() * model_output + direction
            if float(self.config.eta):
                noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample + sigma * noise
        prev_sample = prev_sample.clamp(0.0, 1.0)
        if not return_dict:
            return (prev_sample, model_output)
        return RADMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=model_output,
            noise=noise,
        )

    def posterior_mean_variance(
        self,
        x_start: Float[torch.Tensor, "batch proposals 4"],
        sample: Float[torch.Tensor, "batch proposals 4"],
        timestep: Int[torch.Tensor, "batch"] | Int[torch.Tensor, ""] | int,
    ) -> tuple[
        Float[torch.Tensor, "batch proposals 4"],
        Float[torch.Tensor, "batch proposals 4"],
        Float[torch.Tensor, "batch proposals 4"],
    ]:
        """Return RADM posterior mean, variance, and clipped log variance."""
        t = _as_batch_timesteps(timestep, sample.shape[0], sample.device)
        mean = (
            _extract(self.posterior_mean_coef1, t, sample.shape).to(
                device=sample.device, dtype=sample.dtype
            )
            * x_start
            + _extract(self.posterior_mean_coef2, t, sample.shape).to(
                device=sample.device, dtype=sample.dtype
            )
            * sample
        )
        variance = _extract(self.posterior_variance, t, sample.shape).to(
            device=sample.device, dtype=sample.dtype
        )
        log_variance = _extract(
            self.posterior_log_variance_clipped, t, sample.shape
        ).to(device=sample.device, dtype=sample.dtype)
        return mean, variance, log_variance


def _as_batch_timesteps(
    timestep: Int[torch.Tensor, "batch"] | Int[torch.Tensor, ""] | int,
    batch_size: int,
    device: torch.device,
) -> Int[torch.Tensor, "batch"]:
    t = torch.as_tensor(timestep, device=device, dtype=torch.long)
    if t.ndim == 0:
        t = t.repeat(batch_size)
    return t


def _extract(
    values: Float[torch.Tensor, "steps"],
    timesteps: Int[torch.Tensor, "batch"],
    sample_shape: torch.Size,
) -> Float[torch.Tensor, "batch ..."]:
    out = values.to(device=timesteps.device).gather(-1, timesteps)
    return out.reshape(
        timesteps.shape[0],
        *((1,) * (len(sample_shape) - 1)),
    )


def cosine_beta_schedule(
    timesteps: int, *, s: float = 0.008
) -> Float[torch.Tensor, "timesteps"]:
    """Return the cosine beta schedule used by RADM-style diffusion.

    Args:
        timesteps: Number of training timesteps.
        s: Small schedule offset.

    Returns:
        One-dimensional beta tensor.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


def _previous_timestep(timesteps: Int[torch.Tensor, "steps"], timestep: int) -> int:
    matches = (timesteps.cpu() == timestep).nonzero()
    if len(matches) == 0:
        return timestep - 1
    index = int(matches[0].item())
    if index >= len(timesteps) - 1:
        return -1
    return int(timesteps[index + 1].item())
