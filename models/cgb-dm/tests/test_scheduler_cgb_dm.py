import pytest
import torch

from cgb_dm import CGBDMScheduler
from laygen.common import ConditionType


def test_scheduler_keeps_training_and_sampling_buffers_separate():
    scheduler = CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=2)
    assert scheduler.train_beta_schedule == "cosine"
    assert scheduler.sampling_beta_schedule == "linear"
    assert not torch.allclose(
        scheduler.train_alphas_cumprod, scheduler.sampling_alphas_cumprod
    )
    assert scheduler.ddim_timesteps.tolist() == [0, 5]


def test_add_noise_and_step_shapes():
    scheduler = CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=2)
    sample = torch.zeros(2, 3, 8)
    noise = torch.ones_like(sample)
    t = torch.tensor([0, 1])
    fix_mask = torch.zeros_like(sample, dtype=torch.bool)
    fix_mask[:, :, :4] = True

    noised = scheduler.add_noise(sample, noise, t, fix_mask=fix_mask)
    assert torch.equal(noised[:, :, :4], sample[:, :, :4])

    step = scheduler.step(noise, t, noised, 0)
    assert step.prev_sample.shape == sample.shape


def test_condition_masks_for_supported_modes():
    scheduler = CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=2)
    layout = torch.zeros(1, 3, 8)
    layout[:, :, 1] = 1

    assert not scheduler.condition_mask(layout, ConditionType.content_image).any()
    assert scheduler.condition_mask(layout, ConditionType.label)[:, :, :4].all()
    label_size = scheduler.condition_mask(layout, ConditionType.label_size)
    assert label_size[:, :, :4].all()
    assert label_size[:, :, 6:8].all()
    assert scheduler.condition_mask(layout, ConditionType.refinement).all()

    completion = scheduler.condition_mask(
        layout,
        ConditionType.completion,
        generator=torch.Generator().manual_seed(0),
    )
    assert completion.shape == layout.shape

    with pytest.raises(ValueError, match="Unsupported CGB-DM"):
        scheduler.condition_mask(layout, ConditionType.text)
