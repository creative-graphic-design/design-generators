from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LayoutDMSamplingConfig:
    """Sampling options for LayoutDM reverse diffusion.

    Args:
        name: Sampling strategy name.
        temperature: Sampling temperature.
        top_k: Top-k cutoff for top-k sampling.
        top_p: Nucleus cutoff for top-p sampling.
        num_inference_steps: Optional inference timestep count.
        time_difference: Reserved timestep offset.
        refine_lambda: Refinement strength used by compatible schedulers.
        refine_mode: Refinement noise mode.
        refine_offset_ratio: Refinement offset ratio.

    Returns:
        Dataclass carrying scheduler sampling options.

    Raises:
        ValueError: Construction does not raise directly.

    Examples:
        >>> LayoutDMSamplingConfig(name="deterministic").name
        'deterministic'
    """

    name: Literal[
        "deterministic", "random", "gumbel", "top_k", "top_p", "top_k_top_p"
    ] = "random"
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.9
    num_inference_steps: int | None = None
    time_difference: float = 0.0
    refine_lambda: float = 3.0
    refine_mode: Literal["uniform", "gaussian", "negative"] = "uniform"
    refine_offset_ratio: float = 0.1
