import torch

from layout_corrector.sampling import (
    LayoutCorrectorSamplingConfig,
    add_confidence_gumbel_noise,
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


def test_threshold_masks_low_sigmoid_confidence():
    logits = torch.tensor([[0.0, 2.0, -2.0]])

    mask = select_tokens_to_remask(
        logits,
        mask_ratio=0.5,
        mode="thresh",
        threshold=0.6,
    )

    assert mask.tolist() == [[True, False, True]]


def test_topk_masks_lowest_confidence_count():
    logits = torch.tensor([[3.0, -1.0, 2.0, 0.0]])

    mask = select_tokens_to_remask(
        logits,
        mask_ratio=0.5,
        mode="topk",
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
