import torch

from layout_dm.sampling import LayoutDMSamplingConfig
from layout_dm.scheduler import LayoutDMScheduler


def test_scheduler_step_shape():
    scheduler = LayoutDMScheduler(
        vocab_size=10, mask_token_id=9, pad_token_id=8, token_mask=[[True] * 10] * 6
    )
    scheduler.set_timesteps(2)
    sample = scheduler.initial_sample(2, 6, device=torch.device("cpu"))
    logits = torch.zeros(2, 6, 10)
    out = scheduler.step(
        logits,
        torch.zeros(2, dtype=torch.long),
        sample,
        previous_timestep=1,
        sampling=LayoutDMSamplingConfig(name="deterministic"),
    )
    assert out.prev_sample.shape == (2, 10, 6)
