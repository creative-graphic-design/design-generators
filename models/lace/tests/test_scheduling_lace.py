import numpy as np
import torch

from lace.scheduling_lace import LaceScheduler, make_ddim_timesteps


def test_vendor_uniform_timesteps_add_one() -> None:
    assert np.array_equal(
        make_ddim_timesteps("uniform", 10, 100),
        np.asarray([1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    )


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
