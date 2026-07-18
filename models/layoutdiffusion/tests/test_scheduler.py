import torch

from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionScheduler
from laygen.common.discrete import index_to_log_onehot


def _scheduler() -> LayoutDiffusionScheduler:
    cfg = LayoutDiffusionConfig(dataset_name="publaynet")
    return LayoutDiffusionScheduler(
        num_train_timesteps=10,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        type_classes=cfg.type_classes,
    )


def test_scheduler_buffers_have_expected_shapes() -> None:
    scheduler = _scheduler()
    assert scheduler.q_onestep_mats.shape == (11, 128, 128)
    assert scheduler.q_mats.shape == (11, 128, 128)
    assert torch.allclose(
        scheduler.q_onestep_mats[0].sum(dim=-1),
        torch.ones(128),
        atol=1e-5,
    )


def test_predict_start_appends_mask_logit() -> None:
    scheduler = _scheduler()
    logits = torch.zeros(2, scheduler.vocab_size - 1, 6)
    pred = scheduler.predict_start(logits, 2, 6)
    assert pred.shape == (2, scheduler.vocab_size, 6)
    assert torch.equal(pred[:, -1, :], torch.full((2, 6), -70.0))


def test_scheduler_step_is_generator_reproducible() -> None:
    scheduler = _scheduler()
    sample_ids = torch.full((1, 6), scheduler.mask_token_id, dtype=torch.long)
    sample = index_to_log_onehot(sample_ids, scheduler.vocab_size)
    logits = torch.zeros(1, scheduler.vocab_size - 1, 6)
    t = torch.tensor([1])
    out1 = scheduler.step(logits, t, sample, generator=torch.Generator().manual_seed(1))
    out2 = scheduler.step(logits, t, sample, generator=torch.Generator().manual_seed(1))
    assert torch.equal(out1.prev_sample, out2.prev_sample)
