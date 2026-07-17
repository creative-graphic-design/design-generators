from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import ConfigMixin, SchedulerMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput


@dataclass
class LayoutFlowSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class LayoutFlowEulerScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self, num_inference_steps: int = 100, start: float = 0.0, end: float = 1.0
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.start = start
        self.end = end
        self.timesteps = torch.linspace(start, end, num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        device: torch.device | str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        steps = num_inference_steps or self.config.num_inference_steps
        start = self.config.start if start is None else start
        end = self.config.end if end is None else end
        self.timesteps = torch.linspace(start, end, steps, device=device)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.Tensor | float,
        sample: torch.FloatTensor,
        *,
        next_timestep: torch.Tensor | float | None = None,
        return_dict: bool = True,
    ) -> LayoutFlowSchedulerOutput | tuple[torch.FloatTensor]:
        t = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        if next_timestep is None:
            matches = torch.isclose(self.timesteps.to(sample.device, sample.dtype), t)
            idx = int(matches.nonzero()[0].item())
            if idx >= len(self.timesteps) - 1:
                next_timestep = t
            else:
                next_timestep = self.timesteps[idx + 1]
        t_next = torch.as_tensor(
            next_timestep, device=sample.device, dtype=sample.dtype
        )
        prev_sample = sample + (t_next - t) * model_output
        if not return_dict:
            return (prev_sample,)
        return LayoutFlowSchedulerOutput(prev_sample=prev_sample)
