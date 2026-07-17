import torch
import pytest

from laygen.common.discrete import SamplingMode
from layout_corrector.sampling import (
    CorrectorMaskMode,
    LayoutCorrectorSamplingConfig,
    add_confidence_gumbel_noise,
    normalize_corrector_mask_mode,
    select_tokens_to_remask,
    should_apply_corrector,
)


def test_should_apply_corrector_prefers_explicit_timesteps():
    config = LayoutCorrectorSamplingConfig(
        corrector_t_list=(10, 20),
        corrector_start=0,
        corrector_end=999,
    )

    assert should_apply_corrector(10, config)
    assert not should_apply_corrector(11, config)
    assert config.sampling is SamplingMode.random
    assert config.corrector_mask_mode is CorrectorMaskMode.thresh


def test_sampling_config_normalizes_public_string_modes():
    config = LayoutCorrectorSamplingConfig(
        sampling="top_k",
        corrector_mask_mode="topk",
    )

    assert config.sampling is SamplingMode.top_k
    assert config.corrector_mask_mode is CorrectorMaskMode.topk


def test_normalize_corrector_mask_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported corrector mask mode"):
        normalize_corrector_mask_mode("unknown")


def test_threshold_masks_low_sigmoid_confidence():
    logits = torch.tensor([[0.0, 2.0, -2.0]])

    mask = select_tokens_to_remask(
        logits,
        mask_ratio=0.5,
        mode=CorrectorMaskMode.thresh,
        threshold=0.6,
    )

    assert mask.tolist() == [[True, False, True]]


def test_topk_masks_lowest_confidence_count():
    logits = torch.tensor([[3.0, -1.0, 2.0, 0.0]])

    mask = select_tokens_to_remask(
        logits,
        mask_ratio=0.5,
        mode=CorrectorMaskMode.topk,
        threshold=0.7,
    )

    assert mask.tolist() == [[False, True, False, True]]


def test_gumbel_noise_uses_generator():
    logits = torch.zeros(2, 3)
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    first = add_confidence_gumbel_noise(
        logits,
        timestep=torch.tensor([1, 1]),
        mask_ratio=0.5,
        temperature=1.0,
        time_adaptive_temperature=False,
        generator=generator_a,
    )
    second = add_confidence_gumbel_noise(
        logits,
        timestep=torch.tensor([1, 1]),
        mask_ratio=0.5,
        temperature=1.0,
        time_adaptive_temperature=False,
        generator=generator_b,
    )

    assert torch.equal(first, second)
