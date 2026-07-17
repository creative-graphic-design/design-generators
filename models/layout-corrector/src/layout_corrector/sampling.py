"""Sampling helpers for Layout-Corrector guided generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from laygen.common.discrete import batch_topk_mask, gumbel_noise_like


@dataclass
class LayoutCorrectorSamplingConfig:
    """Sampling options for Layout-Corrector-guided diffusion.

    Args:
        sampling: Base LayoutDM sampling strategy.
        temperature: Base sampling temperature.
        top_k: Top-k cutoff for top-k sampling.
        top_p: Nucleus cutoff for top-p sampling.
        num_inference_steps: Optional inference timestep count.
        corrector_steps: Number of correction passes per selected timestep.
        corrector_t_list: Explicit timesteps where the corrector is applied.
        corrector_start: Start timestep for range-based correction.
        corrector_end: End timestep for range-based correction.
        corrector_mask_mode: Strategy for selecting tokens to remask.
        corrector_mask_threshold: Confidence threshold for threshold masking.
        corrector_temperature: Temperature used for confidence masking.
        use_gumbel_noise: Whether to perturb confidence logits.
        gumbel_temperature: Temperature for confidence Gumbel noise.
        time_adaptive_temperature: Whether to scale noise by timestep ratio.

    Returns:
        Dataclass carrying Layout-Corrector sampling options.

    Raises:
        ValueError: Construction does not raise directly.

    Examples:
        >>> LayoutCorrectorSamplingConfig(corrector_t_list=(10,)).corrector_t_list
        (10,)
    """

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
    """Return whether the corrector should run at a diffusion timestep.

    Args:
        diffusion_index: Current diffusion timestep.
        config: Corrector sampling options.

    Returns:
        `True` when the timestep matches the explicit list or configured range.

    Raises:
        ValueError: This function does not raise.

    Examples:
        >>> should_apply_corrector(10, LayoutCorrectorSamplingConfig(corrector_t_list=(10,)))
        True
    """
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
    """Add Gumbel noise to confidence logits.

    Args:
        confidence_logits: Rank-2 confidence logits.
        timestep: Current timestep tensor.
        mask_ratio: Fraction of tokens still masked.
        temperature: Base noise temperature.
        time_adaptive_temperature: Whether to scale temperature by `mask_ratio`.
        generator: Optional PyTorch generator for reproducible noise.

    Returns:
        Confidence logits after noise injection.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> logits = torch.zeros(1, 2)
        >>> add_confidence_gumbel_noise(
        ...     logits,
        ...     timestep=torch.tensor([1]),
        ...     mask_ratio=0.5,
        ...     temperature=1.0,
        ...     time_adaptive_temperature=False,
        ...     generator=torch.Generator().manual_seed(0),
        ... ).shape
        torch.Size([1, 2])
    """
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
    """Select low-confidence tokens that should be masked again.

    Args:
        confidence_logits: Rank-2 confidence logits shaped `(batch, tokens)`.
        mask_ratio: Ratio used to choose top-k remasking count.
        mode: Selection mode, either `"thresh"` or `"topk"`.
        threshold: Sigmoid confidence threshold for `"thresh"`.
        temperature: Temperature applied before thresholding.

    Returns:
        Boolean mask with `True` where tokens should be remasked.

    Raises:
        ValueError: If logits are not rank-2 or `mode` is unsupported.

    Examples:
        >>> import torch
        >>> select_tokens_to_remask(torch.zeros(1, 2), mask_ratio=0.5, mode="topk", threshold=0.7).shape
        torch.Size([1, 2])
    """
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
