import torch

from layout_flow import LayoutFlowEulerScheduler


def test_euler_scheduler_matches_hand_update() -> None:
    scheduler = LayoutFlowEulerScheduler(num_inference_steps=3)
    scheduler.set_timesteps(3)
    sample = torch.zeros(2, 4)
    velocity = torch.ones_like(sample)
    out = scheduler.step(
        velocity,
        scheduler.timesteps[0],
        sample,
        next_timestep=scheduler.timesteps[1],
    ).prev_sample
    assert torch.allclose(out, torch.full_like(sample, 0.5))


def test_refinement_timesteps_are_increasing_tail() -> None:
    scheduler = LayoutFlowEulerScheduler()
    scheduler.set_timesteps(4, start=0.97, end=1.0)
    assert torch.allclose(scheduler.timesteps, torch.tensor([0.97, 0.98, 0.99, 1.0]))


def test_scheduler_auto_next_tuple_and_scale_model_input() -> None:
    scheduler = LayoutFlowEulerScheduler(num_inference_steps=3)
    scheduler.set_timesteps(device="cpu")
    sample = torch.zeros(1, 2)
    velocity = torch.ones_like(sample)

    assert scheduler.scale_model_input(sample, scheduler.timesteps[0]) is sample
    first = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    assert torch.allclose(first, torch.full_like(sample, 0.5))
    last = scheduler.step(velocity, scheduler.timesteps[-1], first, return_dict=False)[
        0
    ]
    assert torch.allclose(last, first)
