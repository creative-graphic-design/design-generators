import math

import pytest
import torch
from torch import nn

from canvas_vae.training import (
    CrossEpochBatchSampler,
    KerasAdam,
    clip_gradients_by_norm,
    l2_penalty,
    sequential_batches,
    wrapping_batches,
)


def keras_adam_reference(param, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)
    for step, grad in enumerate(grads, start=1):
        m = m + (grad - m) * (1 - b1)
        v = v + (grad**2 - v) * (1 - b2)
        alpha = lr * math.sqrt(1 - b2**step) / (1 - b1**step)
        param = param - m * alpha / (v.sqrt() + eps)

    return param, m, v


def test_keras_adam_matches_reference():
    generator = torch.Generator().manual_seed(0)
    start = torch.randn(5, generator=generator)
    grads = [torch.randn(5, generator=generator) for _ in range(3)]
    param = nn.Parameter(start.clone())
    optimizer = KerasAdam([param])
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()

    expected, m, v = keras_adam_reference(start, grads)
    torch.testing.assert_close(param.detach(), expected)
    torch.testing.assert_close(optimizer.state[param]["exp_avg"], m)
    torch.testing.assert_close(optimizer.state[param]["exp_avg_sq"], v)
    assert optimizer.state[param]["step"] == 3


def test_keras_adam_closure_and_missing_grad():
    param = nn.Parameter(torch.ones(1))
    other = nn.Parameter(torch.ones(1))
    optimizer = KerasAdam([param, other])
    param.grad = torch.ones(1)
    assert optimizer.step(lambda: torch.tensor(2.0)).item() == 2.0
    assert other.item() == 1.0


def test_clip_is_per_tensor():
    small = nn.Parameter(torch.zeros(2))
    large = nn.Parameter(torch.zeros(2))
    small.grad = torch.tensor([0.3, 0.4])
    large.grad = torch.tensor([3.0, 4.0])
    clip_gradients_by_norm([small, large, nn.Parameter(torch.zeros(1))], 1.0)
    torch.testing.assert_close(small.grad, torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(large.grad, torch.tensor([0.6, 0.8]))


def test_l2_penalty_covers_dense_and_embedding_only():
    model = nn.Sequential(nn.Embedding(2, 2), nn.Linear(2, 1), nn.LayerNorm(1))
    for param in model.parameters():
        nn.init.ones_(param)

    assert l2_penalty(model, 0.5).item() == 0.5 * (4 + 2 + 1)


def test_cross_epoch_sampler_straddles_passes():
    sampler = CrossEpochBatchSampler(5, 2, torch.Generator().manual_seed(0))
    first = list(sampler)
    second = list(sampler)
    assert len(sampler) == 3 and all(len(batch) == 2 for batch in first + second)
    stream = [index for batch in first + second for index in batch]
    assert sorted(stream[:5]) == list(range(5)) and sorted(stream[5:10]) == list(range(5))
    assert len(set(first[2])) == 2


def test_rico_scale_stream_constants():
    assert len(CrossEpochBatchSampler(45222, 1024, torch.Generator())) == 45
    batches = wrapping_batches(5584, 1024)
    assert len(batches) == 6 and batches[-1][-1] == 6 * 1024 - 5584 - 1
    tests = sequential_batches(5623, 1024)
    assert len(tests) == 6 and len(tests[-1]) == 503


@pytest.mark.training
def test_lightning_fit_smoke(rico_dir, tmp_path):
    from traingen.lightning.cli import main

    main([
        "fit",
        "--config", "models/canvas-vae/configs/training/smoke.yaml",
        f"--model.init_args.data_dir={rico_dir}",
        f"--data.init_args.data_dir={rico_dir}",
        f"--trainer.default_root_dir={tmp_path}",
    ])


@pytest.mark.training
def test_training_module_rejects_global_clipping(rico_dir):
    from canvas_vae.training import CanvasVAEDataModule, CanvasVAETrainingModule

    module = CanvasVAETrainingModule(data_dir=str(rico_dir), latent_dim=16, num_heads=2)
    with pytest.raises(ValueError):
        module.configure_gradient_clipping(module.configure_optimizers(), gradient_clip_val=1.0)

    data = CanvasVAEDataModule(data_dir=str(rico_dir), batch_size=4, seed=0)
    with pytest.raises(RuntimeError):
        data.train_dataloader()

    data.setup()
    batch = next(iter(data.train_dataloader()))
    assert batch["element_ids"].shape[0] == 4
    assert sum(len(b["num_elements"]) for b in data.test_dataloader()) == len(data.splits["test"])
