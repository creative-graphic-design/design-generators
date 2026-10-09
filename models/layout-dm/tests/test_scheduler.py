import math

import pytest
import torch

from laygen.common.discrete import SamplingMode
from layout_dm.conditioning import LayoutDMCondition
from layout_dm.sampling import LayoutDMSamplingConfig
from layout_dm.scheduling_layout_dm import LayoutDMScheduler


def test_scheduler_step_shape():
    scheduler = LayoutDMScheduler(vocab_size=10, mask_token_id=9, pad_token_id=8)
    scheduler.set_timesteps(2)
    sample = scheduler.initial_sample(2, 6, device=torch.device("cpu"))
    logits = torch.zeros(2, 6, 10)
    out = scheduler.step(
        logits,
        torch.zeros(2, dtype=torch.long),
        sample,
        previous_timestep=1,
        sampling=LayoutDMSamplingConfig(name=SamplingMode.deterministic),
    )
    assert out.prev_sample.shape == (2, 10, 6)


def test_scheduler_keeps_constrained_posterior_floor():
    per_var_full_ids = {"x": [0, 2, 3], "y": [0, 1, 3]}
    scheduler = LayoutDMScheduler(
        num_timesteps=4,
        vocab_size=4,
        mask_token_id=3,
        pad_token_id=2,
        var_order=("x", "y"),
        per_var_full_ids=per_var_full_ids,
    )
    sample = scheduler.initial_sample(1, 2, device=torch.device("cpu"))
    logits = torch.zeros(1, 2, 4)
    out = scheduler.step(
        logits,
        torch.tensor([3]),
        sample,
        previous_timestep=4,
        sampling=LayoutDMSamplingConfig(name=SamplingMode.deterministic),
    )

    assert out.model_log_prob is not None
    assert out.model_log_prob[0, 1, 0].item() == pytest.approx(math.log(1e-30))


def test_scheduler_predict_start_softmaxes_vocab_axis_before_flattening() -> None:
    scheduler = LayoutDMScheduler(vocab_size=4, mask_token_id=3, pad_token_id=2)
    logits = torch.tensor([[[1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 9.0]]])

    result = scheduler.predict_start(logits)
    expected = torch.log_softmax(logits[:, :, :-1].permute(0, 2, 1).double(), dim=1)
    expected = expected.float()
    expected = torch.cat((expected, torch.full((1, 1, 2), -70.0)), dim=1).clamp(
        -70.0, 0.0
    )

    assert torch.equal(result, expected)


def test_scheduler_masks_weak_condition_tokens_at_initialization():
    scheduler = LayoutDMScheduler(
        vocab_size=4,
        mask_token_id=3,
        pad_token_id=2,
    )
    condition = LayoutDMCondition(
        input_ids=torch.tensor([[1, 2]]),
        mask=torch.tensor([[True, False]]),
        type="c",
    )

    sample = scheduler.initial_sample(
        batch_size=1,
        token_length=2,
        device=torch.device("cpu"),
        condition=condition,
    )

    assert torch.equal(torch.argmax(sample, dim=1), torch.tensor([[1, 3]]))


def test_scheduler_restricts_padding_for_conditional_attributes():
    scheduler = LayoutDMScheduler(
        vocab_size=4,
        mask_token_id=3,
        pad_token_id=2,
    )
    condition = LayoutDMCondition(
        input_ids=torch.tensor([[1, 3]]),
        mask=torch.tensor([[True, False]]),
        type="c",
    )
    log_prob = torch.zeros(1, 4, 2)

    output = scheduler.apply_condition(log_prob, condition)

    assert output[0, 2, 1].item() == pytest.approx(math.log(1.0e-30))
    assert output[0, 1, 0].item() == pytest.approx(0.0)
