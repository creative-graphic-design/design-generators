import torch

from layoutdiffusion.training.config import (
    LayoutDiffusionSeedMode,
    LayoutDiffusionTimeSampler,
    LayoutDiffusionTrainingDatasetName,
    LayoutDiffusionTrainingScheduler,
)
from layoutdiffusion.training.losses import (
    log_categorical,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
    sum_except_batch,
)
from layoutdiffusion.training.seed import apply_layoutdiffusion_seed_mode


def test_loss_helpers_match_categorical_identities() -> None:
    log_prob = torch.log(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).clamp_min(1e-30))
    assert torch.equal(sum_except_batch(torch.ones(2, 3)), torch.full((2,), 3.0))
    assert torch.equal(multinomial_kl(log_prob, log_prob), torch.zeros(1, 2))
    assert torch.allclose(log_categorical(log_prob, log_prob), torch.zeros(1, 2))


def test_time_samplers_are_reproducible() -> None:
    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    t1, pt1 = sample_time_uniform(
        4, num_timesteps=10, device=torch.device("cpu"), generator=gen1
    )
    t2, pt2 = sample_time_uniform(
        4, num_timesteps=10, device=torch.device("cpu"), generator=gen2
    )
    assert torch.equal(t1, t2)
    assert torch.equal(pt1, pt2)
    assert torch.equal(pt1, torch.full((4,), 0.1))

    history = torch.arange(1, 11, dtype=torch.float32)
    count = torch.full((10,), 11.0)
    t_imp, pt_imp = sample_time_importance(
        4,
        num_timesteps=10,
        lt_history=history,
        lt_count=count,
        generator=torch.Generator().manual_seed(8),
    )
    assert t_imp.shape == pt_imp.shape == (4,)
    assert torch.isfinite(pt_imp).all()

    t_fallback, pt_fallback = sample_time_importance(
        4,
        num_timesteps=10,
        lt_history=history,
        lt_count=torch.zeros(10),
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(t_fallback, t1)
    assert torch.equal(pt_fallback, pt1)


def test_seed_modes_and_training_string_options_are_constrained() -> None:
    assert LayoutDiffusionSeedMode("default") is LayoutDiffusionSeedMode.default
    apply_layoutdiffusion_seed_mode("default", seed=1)
    apply_layoutdiffusion_seed_mode("deterministic", seed=1)
    assert LayoutDiffusionTrainingDatasetName.__args__ == ("rico25", "publaynet")
    assert LayoutDiffusionTrainingScheduler.__args__ == ("linear_anneal",)
    assert LayoutDiffusionTimeSampler.__args__ == ("importance", "uniform")
