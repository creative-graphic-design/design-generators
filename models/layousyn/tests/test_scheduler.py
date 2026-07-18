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


def test_ddim_step_shape() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=4)
    sample = torch.zeros(1, 2, 4)
    model_output = torch.zeros(1, 4, 4)
    out = scheduler.step(model_output, torch.tensor([1]), sample)
    assert isinstance(out, LayouSynSchedulerOutput)
    assert out.prev_sample.shape == sample.shape
    assert out.pred_original_sample.shape == sample.shape


def test_scheduler_respacing_matches_vendor_single_section() -> None:
    scheduler = LayouSynScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(40)
    assert scheduler.timestep_map[:5].tolist() == [0, 3, 5, 8, 10]
    assert scheduler.model_timesteps[:3].tolist() == [99, 96, 94]
    assert scheduler.timesteps[:3].tolist() == [39, 38, 37]
