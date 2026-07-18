"""Sampling configuration for LayoutDiffusion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class LayoutDiffusionSamplingName(StrEnum):
    """Supported LayoutDiffusion sampling modes."""

    vendor_gumbel = auto()
    argmax = auto()


@dataclass(frozen=True)
class LayoutDiffusionSamplingConfig:
    """Runtime sampling options for the reverse diffusion loop."""

    name: LayoutDiffusionSamplingName | str = LayoutDiffusionSamplingName.vendor_gumbel
    num_inference_steps: int | None = None
    skip_step: int = 0
    multistep: bool = False
