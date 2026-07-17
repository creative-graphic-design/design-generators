from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class LaceSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


def make_beta_schedule(
    schedule: str = "cosine",
    num_timesteps: int = 1000,
    start: float = 0.0001,
    end: float = 0.02,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(start, end, num_timesteps)
    if schedule == "const":
        return end * torch.ones(num_timesteps)
    if schedule == "quad":
        return torch.linspace(start**0.5, end**0.5, num_timesteps) ** 2
    if schedule == "jsd":
        return 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    if schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(betas) * (end - start) + start
    if schedule in {"cosine", "cosine_reverse"}:
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
    if schedule == "cosine_anneal":
        return torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    raise ValueError(f"Unsupported beta schedule: {schedule}")


def make_ddim_timesteps(
    method: Literal["uniform", "quad", "new"],
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
) -> np.ndarray:
    if method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif method == "quad":
        timesteps = (
            np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2
        ).astype(int)
    elif method == "new":
        c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
        timesteps = np.asarray(
            list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c))
        )
    else:
        raise ValueError(f"Unsupported ddim discretization: {method}")
    return timesteps + 1


class LaceScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        ddim_num_steps: int = 100,
        ddim_discretize: Literal["uniform", "quad", "new"] = "uniform",
        eta: float = 0.0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.ddim_num_steps = ddim_num_steps
        self.ddim_discretize = ddim_discretize
        self.eta = eta
        betas = make_beta_schedule(
            beta_schedule, num_timesteps=num_train_timesteps, start=0.0001, end=0.02
        ).float()
        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.timesteps = torch.empty(0, dtype=torch.long)
        self.set_timesteps(ddim_num_steps)

    def set_timesteps(
        self, num_inference_steps: int | None = None, device: torch.device | None = None
    ) -> None:
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
        total = int(torch.sum(self.ddim_timesteps <= max_timestep).item())
        return list(range(total - 1, -1, -1))
