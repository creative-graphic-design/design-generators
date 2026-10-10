import math

import pytest
import torch
from torch import nn

from canvas_vae.processing_canvas_vae import RicoSplit
from canvas_vae.training import (
    CrossEpochBatchSampler,
    clip_gradients_by_norm,
    l2_penalty,
    sequential_batches,
    wrapping_batches,
)
from traingen.optim import KerasAdam


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
    assert optimizer.step(lambda: 2.0) == 2.0
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
    assert sorted(stream[:5]) == list(range(5)) and sorted(stream[5:10]) == list(
        range(5)
    )
    assert len(set(first[2])) == 2


def test_cross_epoch_sampler_default_order_matches_random_stream():
    generator = torch.Generator().manual_seed(0)
    sampler = CrossEpochBatchSampler(5, 2, generator)
    expected_generator = torch.Generator().manual_seed(0)
    expected_stream = []
    while len(expected_stream) < 12:
        expected_stream.extend(torch.randperm(5, generator=expected_generator).tolist())

    expected = [expected_stream[index : index + 2] for index in range(0, 12, 2)]
    assert list(sampler) + list(sampler) == expected


def test_rico_scale_stream_constants():
    assert len(CrossEpochBatchSampler(45222, 1024, torch.Generator())) == 45
    batches = wrapping_batches(5584, 1024)
    assert len(batches) == 6 and batches[-1][-1] == 6 * 1024 - 5584 - 1
    tests = sequential_batches(5623, 1024)
    assert len(tests) == 6 and len(tests[-1]) == 503


@pytest.mark.training
def test_lightning_fit_smoke(rico_dir, tmp_path):
    from traingen.lightning.cli import main
    from canvas_vae.training.checkpoints import validate_final_checkpoint

    main(
        [
            "fit",
            "--config",
            "models/canvas-vae/configs/training/smoke.yaml",
            f"--model.init_args.data_dir={rico_dir}",
            f"--data.init_args.data_dir={rico_dir}",
            f"--trainer.default_root_dir={tmp_path}",
        ]
    )

    assert validate_final_checkpoint(
        tmp_path / "checkpoints" / "last.ckpt",
        max_epochs=1,
        expected_global_step=2,
    ) == (0, 2)


@pytest.mark.training
def test_final_checkpoint_is_saved_when_validation_worsens(tmp_path):
    import torch
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    from canvas_vae.training.checkpoints import validate_final_checkpoint

    class WorseningValidationScore(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(
            self, batch: tuple[torch.Tensor, ...], batch_idx: int
        ) -> torch.Tensor:
            return self.weight.square()

        def validation_step(
            self, batch: tuple[torch.Tensor, ...], batch_idx: int
        ) -> None:
            score = 1.0 - self.current_epoch
            self.log("val/total_score", score)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    loader = DataLoader(TensorDataset(torch.ones(1)), batch_size=1)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path,
                monitor="val/total_score",
                mode="max",
                save_top_k=1,
                save_last=False,
                filename="best",
            ),
            ModelCheckpoint(dirpath=tmp_path, save_top_k=1, save_last=True),
        ],
    )
    module = WorseningValidationScore()
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    best_checkpoint = tmp_path / "best.ckpt"
    final_checkpoint = tmp_path / "last.ckpt"
    assert best_checkpoint.is_file()
    assert validate_final_checkpoint(
        final_checkpoint,
        max_epochs=2,
        expected_global_step=2,
    ) == (1, 2)
    final_state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    torch.testing.assert_close(
        final_state["state_dict"]["weight"], module.weight.detach()
    )
    with pytest.raises(ValueError, match="found epoch=0, global_step=1"):
        validate_final_checkpoint(
            best_checkpoint,
            max_epochs=2,
            expected_global_step=2,
        )


@pytest.mark.training
def test_training_module_rejects_global_clipping(rico_dir):
    from canvas_vae.training.datamodule import CanvasVAEDataModule
    from canvas_vae.training.lightning_module import CanvasVAETrainingModule

    module = CanvasVAETrainingModule(data_dir=str(rico_dir), latent_dim=16, num_heads=2)
    with pytest.raises(ValueError):
        module.configure_gradient_clipping(
            module.configure_optimizers(), gradient_clip_val=1.0
        )

    data = CanvasVAEDataModule(data_dir=str(rico_dir), batch_size=4, seed=0)
    with pytest.raises(RuntimeError):
        data.train_dataloader()

    data.setup()
    batch = next(iter(data.train_dataloader()))
    assert batch["element_ids"].shape[0] == 4
    assert sum(len(b["num_elements"]) for b in data.test_dataloader()) == len(
        data.splits[RicoSplit.test]
    )
