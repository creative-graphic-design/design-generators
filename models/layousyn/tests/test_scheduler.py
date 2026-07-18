from typing import Literal, cast

import torch

from layousyn.scheduling_layousyn import LayouSynScheduler
from layousyn.scheduling_layousyn import LayouSynSchedulerOutput


def test_scheduler_generator_reproducible() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=4)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(
        scheduler.initial_sample(1, 2, 4, device=torch.device("cpu"), generator=g1),
        scheduler.initial_sample(1, 2, 4, device=torch.device("cpu"), generator=g2),
    )
    noised = scheduler.add_noise(
        torch.zeros(1, 2, 4), torch.ones(1, 2, 4), torch.tensor([1])
    )
    assert noised.shape == (1, 2, 4)


def test_ddim_step_shape() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=4)
    sample = torch.zeros(1, 2, 4)
    model_output = torch.zeros(1, 4, 4)
    out = scheduler.step(model_output, torch.tensor([1]), sample)
    assert isinstance(out, LayouSynSchedulerOutput)
    assert out.prev_sample.shape == sample.shape
    assert out.pred_original_sample.shape == sample.shape
    tuple_out = scheduler.step(
        model_output, torch.tensor([1]), sample, return_dict=False
    )
    assert isinstance(tuple_out, tuple)
    clip_scheduler = LayouSynScheduler(num_train_timesteps=100)
    clipped = clip_scheduler.step(
        torch.ones_like(model_output) * -2,
        torch.tensor([99]),
        sample,
        clip_denoised=True,
    )
    assert isinstance(clipped, LayouSynSchedulerOutput)
    assert clipped.pred_original_sample.min() >= -1.0


def test_ddpm_and_error_paths() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=4)
    sample = torch.zeros(1, 2, 4)
    model_output = torch.zeros(1, 4, 4)
    out = scheduler.step(model_output, torch.tensor([1]), sample, sampling_type="ddpm")
    assert isinstance(out, LayouSynSchedulerOutput)
    for kwargs, message in [
        ({"prediction_type": "sample"}, "prediction_type"),
        ({"variance_type": "fixed"}, "variance_type"),
    ]:
        try:
            LayouSynScheduler(**kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected failure containing {message}")
    try:
        scheduler.step(
            model_output,
            torch.tensor([1]),
            sample,
            sampling_type=cast(Literal["ddim"], "bad"),
        )
    except ValueError as exc:
        assert "sampling_type" in str(exc)
    else:
        raise AssertionError("bad sampling type should fail")
    try:
        scheduler.set_timesteps(5)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("too many timesteps should fail")


def test_scheduler_respacing_matches_vendor_single_section() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(40)
    assert scheduler.timestep_map[:5].tolist() == [0, 3, 5, 8, 10]
    assert scheduler.model_timesteps[:3].tolist() == [99, 96, 94]
    assert scheduler.timesteps[:3].tolist() == [39, 38, 37]
