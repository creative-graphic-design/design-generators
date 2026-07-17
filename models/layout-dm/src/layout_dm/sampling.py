from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LayoutDMSamplingConfig:
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
