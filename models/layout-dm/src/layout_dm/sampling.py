"""Sampling configuration for LayoutDM reverse diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from laygen.common.discrete import SamplingMode, normalize_sampling_mode


@dataclass
class LayoutDMSamplingConfig:
    """Sampling parameters passed from the pipeline to the scheduler."""

    name: SamplingMode | str = SamplingMode.random
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.9
    num_inference_steps: int | None = None
    time_difference: float = 0.0
    refine_lambda: float = 3.0
    refine_mode: Literal["uniform", "gaussian", "negative"] = "uniform"
    refine_offset_ratio: float = 0.1

    def __post_init__(self) -> None:
        """Normalize public string sampling values to ``SamplingMode``."""
        self.name = normalize_sampling_mode(self.name)
