import torch
import pytest

from radm.scheduling_radm import RADMScheduler, cosine_beta_schedule


def test_cosine_beta_schedule_range() -> None:
    betas = cosine_beta_schedule(10)
    assert betas.shape == (10,)
    assert torch.all((betas > 0) & (betas < 1))


def test_scheduler_timesteps_and_generator_reproducibility() -> None:
    scheduler = RADMScheduler(num_train_timesteps=10, num_inference_steps=3)
    scheduler.set_timesteps(3)
    assert scheduler.timesteps.tolist() == [9, 5, 2]
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    sample1 = scheduler.sample_initial_proposals(
        batch_size=1, num_proposals=2, generator=g1
    )
    sample2 = scheduler.sample_initial_proposals(
        batch_size=1, num_proposals=2, generator=g2
    )
    assert torch.allclose(sample1, sample2)


def test_scheduler_step_tuple() -> None:
    scheduler = RADMScheduler(num_train_timesteps=10, num_inference_steps=2)
    scheduler.set_timesteps(2)
    sample = torch.full((1, 2, 4), 0.4)
    out = scheduler.step(sample, scheduler.timesteps[0], sample, return_dict=False)
    assert out[0].shape == sample.shape


def test_scheduler_invalid_config_and_eta_noise() -> None:
    with pytest.raises(ValueError, match="beta_schedule"):
        RADMScheduler(beta_schedule="linear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prediction_type"):
        RADMScheduler(prediction_type="epsilon")  # type: ignore[arg-type]
    scheduler = RADMScheduler(num_train_timesteps=10, num_inference_steps=2, eta=0.1)
    scheduler.set_timesteps(2)
    sample = torch.full((1, 2, 4), 0.4)
    out = scheduler.step(
        sample,
        scheduler.timesteps[0],
        sample,
        generator=torch.Generator().manual_seed(0),
    )
    assert out.noise is not None
