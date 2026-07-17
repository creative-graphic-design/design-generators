import numpy as np
import pytest
import torch

from lace.scheduling_lace import (
    BetaSchedule,
    DDIMDiscretization,
    LaceScheduler,
    make_beta_schedule,
    make_ddim_timesteps,
    normalize_beta_schedule,
    normalize_ddim_discretization,
)


def test_vendor_uniform_timesteps_add_one() -> None:
    assert np.array_equal(
        make_ddim_timesteps("uniform", 10, 100),
        np.asarray([1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    )


def test_beta_schedules_accept_enum_and_reject_unknown() -> None:
    for schedule in BetaSchedule:
        betas = make_beta_schedule(schedule, num_timesteps=8)
        assert betas.shape == (8,)
        assert torch.isfinite(betas).all()
    assert normalize_beta_schedule("linear") is BetaSchedule.linear
    with pytest.raises(ValueError, match="Unsupported beta schedule"):
        normalize_beta_schedule("bad")


def test_ddim_discretization_accepts_enum_and_rejects_unknown() -> None:
    assert np.array_equal(
        make_ddim_timesteps(DDIMDiscretization.quad, 4, 100),
        make_ddim_timesteps("quad", 4, 100),
    )
    assert normalize_ddim_discretization("new") is DDIMDiscretization.new
    with pytest.raises(ValueError, match="Unsupported ddim discretization"):
        normalize_ddim_discretization("bad")


def test_scheduler_step_shapes_and_refinement_indices() -> None:
    scheduler = LaceScheduler(num_train_timesteps=1000, ddim_num_steps=100)
    scheduler.set_timesteps(100)
    sample = torch.randn(2, 25, 10)
    model_output = torch.randn_like(sample)
    timestep = torch.full((2,), int(scheduler.ddim_timesteps[-1]), dtype=torch.long)
    out = scheduler.step(model_output, timestep, sample, index=99)
    assert out.prev_sample.shape == sample.shape
    assert out.pred_original_sample.shape == sample.shape
    assert scheduler.refinement_indices() == list(range(20, -1, -1))


def test_scheduler_noise_helpers_are_shape_stable() -> None:
    scheduler = LaceScheduler(
        num_train_timesteps=100,
        ddim_num_steps=10,
        beta_schedule=BetaSchedule.cosine,
        ddim_discretize=DDIMDiscretization.uniform,
    )
    clean = torch.zeros(2, 3, 4)
    noise = torch.ones_like(clean)
    noisy = scheduler.add_noise(clean, noise, torch.tensor([0, 1]))
    zeros = scheduler.initial_sample(
        2, 3, 4, device=torch.device("cpu"), stochastic=False
    )
    assert noisy.shape == clean.shape
    assert torch.equal(zeros, clean)
