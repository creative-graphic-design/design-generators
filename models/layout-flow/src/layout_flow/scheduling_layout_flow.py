"""Euler scheduler for LayoutFlow flow-matching inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
from diffusers import ConfigMixin, SchedulerMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput


@dataclass
class LayoutFlowSchedulerOutput(BaseOutput):
    """Output of one LayoutFlow Euler scheduler step."""

    prev_sample: torch.Tensor


class LayoutFlowEulerScheduler(SchedulerMixin, ConfigMixin):
    """Increasing-time Euler scheduler used by LayoutFlow."""

    config_name: str = "scheduler_config.json"
    order: int = 1

    @register_to_config
    def __init__(
        self, num_inference_steps: int = 100, start: float = 0.0, end: float = 1.0
    ) -> None:
        """Initialize the scheduler.

        Args:
            num_inference_steps: Number of Euler steps.
            start: Initial integration time.
            end: Final integration time.
        """
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
        """Set the integration timesteps.

        Args:
            num_inference_steps: Optional number of inference steps.
            device: Optional target device.
            start: Optional start time.
            end: Optional end time.
        """
        steps = num_inference_steps or self.config.num_inference_steps
        start = self.config.start if start is None else start
        end = self.config.end if end is None else end
        self.timesteps = torch.linspace(start, end, steps, device=device)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        """Return the sample unchanged for Diffusers scheduler compatibility."""
        del timestep
        return sample

    @overload
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        next_timestep: torch.Tensor | float | None = None,
        return_dict: Literal[True] = True,
    ) -> LayoutFlowSchedulerOutput: ...

    @overload
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        next_timestep: torch.Tensor | float | None = None,
        return_dict: Literal[False],
    ) -> tuple[torch.Tensor]: ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        next_timestep: torch.Tensor | float | None = None,
        return_dict: bool = True,
    ) -> LayoutFlowSchedulerOutput | tuple[torch.Tensor]:
        """Advance the sample with one Euler step.

        Args:
            model_output: Predicted vector field.
            timestep: Current integration time.
            sample: Current sample state.
            next_timestep: Optional next integration time.
            return_dict: Whether to return a scheduler output dataclass.

        Returns:
            Scheduler output dataclass or single-item tuple.
        """
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
