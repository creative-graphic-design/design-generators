from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from layout_generation_common.discrete import batch_topk_mask, gumbel_noise_like


@dataclass
class LayoutCorrectorSamplingConfig:
    sampling: Literal[
        "deterministic", "random", "gumbel", "top_k", "top_p", "top_k_top_p"
    ] = "random"
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.9
    num_inference_steps: int | None = None
    corrector_steps: int = 1
    corrector_t_list: tuple[int, ...] = (10, 20, 30)
    corrector_start: int = -1
    corrector_end: int = -1
    corrector_mask_mode: Literal["thresh", "topk"] = "thresh"
    corrector_mask_threshold: float = 0.7
    corrector_temperature: float = 1.0
    use_gumbel_noise: bool = True
    gumbel_temperature: float = 1.0
    time_adaptive_temperature: bool = False


def should_apply_corrector(
    diffusion_index: int,
    config: LayoutCorrectorSamplingConfig,
) -> bool:
    if config.corrector_t_list:
        return diffusion_index in set(config.corrector_t_list)
    if config.corrector_start < 0 or config.corrector_end < 0:
        return False
    start, end = sorted((config.corrector_start, config.corrector_end))
    return start <= diffusion_index <= end


def add_confidence_gumbel_noise(
    confidence_logits: torch.FloatTensor,
    *,
    timestep: torch.LongTensor,
    mask_ratio: float,
    temperature: float,
    time_adaptive_temperature: bool,
    generator: torch.Generator | None = None,
) -> torch.FloatTensor:
    scale = torch.full(
        (confidence_logits.shape[0], 1),
        float(temperature),
        device=confidence_logits.device,
        dtype=confidence_logits.dtype,
    )
    if time_adaptive_temperature:
        scale = scale * (1.0 - mask_ratio)
    return confidence_logits + scale * gumbel_noise_like(
        confidence_logits, generator=generator
    )


def select_tokens_to_remask(
    confidence_logits: torch.FloatTensor,
    *,
    mask_ratio: float,
    mode: Literal["thresh", "topk"],
    threshold: float,
    temperature: float = 1.0,
) -> torch.BoolTensor:
    if confidence_logits.ndim != 2:
        raise ValueError("confidence_logits must be rank-2")
    if mode == "thresh":
        confidence = torch.sigmoid(confidence_logits / temperature)
        return confidence < threshold
    if mode == "topk":
        num_token = confidence_logits.shape[1]
        k = torch.full(
            (confidence_logits.shape[0],),
            int(mask_ratio * num_token),
            device=confidence_logits.device,
            dtype=torch.long,
        )
        return batch_topk_mask(-confidence_logits, k)
    raise ValueError(f"Unsupported corrector mask mode: {mode}")
