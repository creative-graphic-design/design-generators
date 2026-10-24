from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
from collections.abc import Iterator, Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers.image_processing_utils import BatchFeature
from tfrecord import example_pb2

import pixel_vae.training.datamodule as datamodule
import pixel_vae.training.lightning_module as lightning_module
from pixel_vae import PixelVAEEncoder, PixelVAEOutput
from pixel_vae.training import (
    PixelVAEDataModule,
    PixelVAEEpochTimingCallback,
    PixelVAETrainingModule,
)
from traingen.lightning.cli import lightning_cli_class
from traingen.optim import KerasAdam


def _png(value: int) -> bytes:
    stream = BytesIO()
    Image.new("RGBA", (256, 256), (value, value, value, 255)).save(stream, format="PNG")
    return stream.getvalue()


def _serialized_example(images: bytes | Sequence[bytes] | None) -> bytes:
    example = example_pb2.Example()
    if images is not None:
        values = [images] if isinstance(images, bytes) else images
        example.features.feature["image"].bytes_list.value.extend(values)

    return example.SerializeToString()


def _write_tfrecord(
    path: Path, images: Sequence[bytes | Sequence[bytes] | None]
) -> None:
    with path.open("wb") as stream:
        for image in images:
            payload = _serialized_example(image)
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"\x00" * 4)
            stream.write(payload)
            stream.write(b"\x00" * 4)


def _prepared_data_dir(
    tmp_path: Path,
    counts: object | None = None,
    *,
    train_images: Sequence[bytes | Sequence[bytes] | None] | None = None,
    val_images: Sequence[bytes | Sequence[bytes] | None] | None = None,
) -> Path:
    data_dir = tmp_path / "crello-image"
    data_dir.mkdir()
    (data_dir / "count.json").write_text(
        json.dumps(counts if counts is not None else {"train": 3, "val": 2}),
        encoding="utf-8",
    )
    _write_tfrecord(
        data_dir / "train-00000.tfrecord",
        train_images
        if train_images is not None
        else [_png(value) for value in (1, 2, 3)],
    )
    _write_tfrecord(
        data_dir / "val-00000.tfrecord",
        val_images if val_images is not None else [_png(value) for value in (10, 20)],
    )
    return data_dir


def _legacy_pixel_batches(
    images: Sequence[bytes],
    batch_size: int,
    epoch_count: int,
    selected_epochs: set[int] | None = None,
) -> dict[int, str]:
    def repeat_records() -> Iterator[bytes]:
        while True:
            yield from datamodule.shuffle_iterator(iter(images), len(images))

    stream = repeat_records()
    epoch_size = (len(images) + batch_size - 1) // batch_size * batch_size
    selected_epochs = (
        selected_epochs if selected_epochs is not None else {0, epoch_count - 1}
    )
    hashes: dict[int, str] = {}
    processor = datamodule.PixelVAEImageProcessor()
    for epoch in range(epoch_count):
        epoch_images = list(islice(stream, epoch_size))
        if epoch not in selected_epochs:
            continue

        digest = hashlib.sha256()
        for start in range(0, batch_size * 20, batch_size):
            batch = processor(epoch_images[start : start + batch_size])["pixel_values"]
            digest.update(cast(torch.Tensor, batch).contiguous().numpy().tobytes())

        hashes[epoch] = digest.hexdigest()

    return hashes


def _tensor_hash(batches: Sequence[BatchFeature]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        pixels = cast(torch.Tensor, batch["pixel_values"])
        digest.update(pixels.contiguous().numpy().tobytes())

    return digest.hexdigest()


@pytest.mark.training
def test_datamodule_repeats_full_batches_and_shuffles_only_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _prepared_data_dir(tmp_path)

    def reverse(records: Iterator[int], queue_size: int) -> Iterator[int]:
        assert queue_size == 3
        return iter(reversed(list(records)))

    monkeypatch.setattr(datamodule, "shuffle_iterator", reverse)
    data = PixelVAEDataModule(str(data_dir), batch_size=2)

    data.setup("fit")
    data.setup("fit")
    train_batches = list(data.train_dataloader())
    val_batches = list(data.val_dataloader())

    assert len(train_batches) == 2
    assert all(
        batch["pixel_values"].shape == (2, 4, 256, 256) for batch in train_batches
    )
    repeated_train_batches = list(data.train_dataloader())
    assert len(val_batches) == 1
    assert len(repeated_train_batches) == 2
    val_pixels = cast(BatchFeature, val_batches[0])["pixel_values"]
    assert torch.allclose(val_pixels[:, 0, 0, 0], torch.tensor([10, 20]) / 255)
    train_pixels = cast(BatchFeature, train_batches[0])["pixel_values"]
    assert torch.allclose(train_pixels[:, 0, 0, 0], torch.tensor([3, 2]) / 255)


@pytest.mark.training
def test_map_dataset_matches_legacy_epoch_1_and_11_tensor_hashes(
    tmp_path: Path,
) -> None:
    images = [_png(value) for value in range(41)]
    data_dir = _prepared_data_dir(
        tmp_path,
        {"train": len(images), "val": 2},
        train_images=images,
    )
    original_state = np.random.get_state()
    np.random.seed(0)
    initial_state = np.random.get_state()
    expected = _legacy_pixel_batches(images, batch_size=2, epoch_count=11)
    np.random.set_state(initial_state)

    actual: dict[int, str] = {}
    data = PixelVAEDataModule(
        str(data_dir), batch_size=2, num_workers=6, pin_memory=True
    )
    data.setup("fit")
    loader = data.train_dataloader()
    assert loader.num_workers == 6
    assert loader.pin_memory
    assert loader.persistent_workers
    try:
        for epoch in (0, 10):
            data.samplers["train"].set_epoch(epoch)
            batches = list(islice(iter(loader), 20))
            actual[epoch] = _tensor_hash(cast(Sequence[BatchFeature], batches))
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        np.random.set_state(original_state)

    assert actual == expected


@pytest.mark.training
def test_parallel_loader_preserves_zero_worker_cpu_rng_sequence(
    tmp_path: Path,
) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    images = torch.arange(1)
    legacy_train = DataLoader(TensorDataset(images), batch_size=1)
    legacy_val = DataLoader(TensorDataset(images), batch_size=1)
    torch.manual_seed(73)
    legacy_values: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _epoch in range(11):
        iter(legacy_train)
        train_value = torch.rand(4)
        iter(legacy_val)
        val_value = torch.rand(4)
        legacy_values.append((train_value, val_value))

    torch.manual_seed(73)
    data = PixelVAEDataModule(str(data_dir), batch_size=2, num_workers=1)
    data.setup("fit")
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    module = cast(PixelVAETrainingModule, SimpleNamespace())
    parallel_values: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for epoch in range(11):
            data.samplers["train"].set_epoch(epoch)
            iter(train_loader)
            PixelVAETrainingModule.on_train_epoch_start(module)
            train_value = torch.rand(4)
            iter(val_loader)
            PixelVAETrainingModule.on_validation_start(module)
            val_value = torch.rand(4)
            parallel_values.append((train_value, val_value))
    finally:
        for loader in (train_loader, val_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()

    assert all(
        torch.equal(actual_train, expected_train)
        and torch.equal(actual_val, expected_val)
        for (actual_train, actual_val), (expected_train, expected_val) in zip(
            parallel_values, legacy_values, strict=True
        )
    )


@pytest.mark.training
def test_datamodule_deduplicates_to_crello_v1_split_counts(
    tmp_path: Path,
) -> None:
    counts = {"train": 89_545, "val": 11_690, "test": 11_428}
    train_images = [
        value.to_bytes(4, "big")
        for value in range(counts["train"])
        for _copy in range(2)
    ]
    val_images = [
        value.to_bytes(4, "big") for value in range(counts["val"]) for _copy in range(2)
    ]
    data_dir = _prepared_data_dir(
        tmp_path, counts, train_images=train_images, val_images=val_images
    )
    data = PixelVAEDataModule(str(data_dir), batch_size=64)
    data.setup("fit")

    assert len(data.datasets["train"]) == 89_545
    assert len(data.datasets["val"]) == 11_690
    assert len(data.samplers["train"]) // 64 == 1_400
    assert len(data.samplers["val"]) // 64 == 183

    for split in ("train", "val"):
        count = counts[split]
        indices = list(data.samplers[split])
        assert len(set(indices[:count])) == count
        assert set(indices[:count]) == set(range(count))
        assert len(indices) == math.ceil(count / 64) * 64


@pytest.mark.training
def test_datamodule_requires_setup_before_loader_access(tmp_path: Path) -> None:
    data = PixelVAEDataModule(str(tmp_path))
    with pytest.raises(RuntimeError, match="call setup"):
        data.train_dataloader()


@pytest.mark.training
@pytest.mark.parametrize("batch_size", [0, -1])
def test_datamodule_requires_positive_batch_size(
    tmp_path: Path, batch_size: int
) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        PixelVAEDataModule(str(data_dir), batch_size=batch_size).setup("fit")


@pytest.mark.training
def test_datamodule_requires_non_negative_worker_count(tmp_path: Path) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        PixelVAEDataModule(str(data_dir), num_workers=-1).setup("fit")


@pytest.mark.training
@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([], "split-to-count"),
        ({"train": True, "val": 1}, "positive train count"),
        ({"train": 3, "val": 0}, "positive val count"),
    ],
)
def test_datamodule_rejects_invalid_counts(
    tmp_path: Path, counts: object, message: str
) -> None:
    data_dir = _prepared_data_dir(tmp_path, counts)
    with pytest.raises(ValueError, match=message):
        PixelVAEDataModule(str(data_dir)).setup("fit")


@pytest.mark.training
def test_datamodule_rejects_missing_split_records(tmp_path: Path) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    (data_dir / "train-00000.tfrecord").unlink()
    with pytest.raises(FileNotFoundError, match="no train-\\*\\.tfrecord"):
        PixelVAEDataModule(str(data_dir)).setup("fit")


@pytest.mark.training
def test_datamodule_rejects_missing_count_file(tmp_path: Path) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    (data_dir / "count.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing PixelVAE split counts"):
        PixelVAEDataModule(str(data_dir)).setup("fit")


@pytest.mark.training
@pytest.mark.parametrize(
    ("train_images", "train_count", "message"),
    [
        ([_png(1)], 3, "count.json record count exceeds"),
        (
            [_png(1), _png(2), _png(3), _png(4)],
            3,
            "count.json record count is smaller",
        ),
        (
            [(_png(1), _png(2)), _png(3)],
            2,
            "image feature must contain bytes",
        ),
    ],
)
def test_datamodule_rejects_invalid_record_contents(
    tmp_path: Path,
    train_images: list[bytes | tuple[bytes, ...]],
    train_count: int,
    message: str,
) -> None:
    data_dir = _prepared_data_dir(
        tmp_path,
        {"train": train_count, "val": 2},
        train_images=train_images,
    )
    data = PixelVAEDataModule(str(data_dir), batch_size=2)

    with pytest.raises(ValueError, match=message):
        data.setup("fit")


class _FakePixelVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.sample_values: list[bool | None] = []

    def forward(
        self, pixel_values: torch.Tensor, *, sample: bool | None = None
    ) -> PixelVAEOutput:
        assert pixel_values.ndim == 4
        self.sample_values.append(sample)
        loss = self.weight.square()
        return PixelVAEOutput(
            loss=loss,
            reconstruction_loss=loss + 1,
            kl_divergence=loss + 2,
        )


@pytest.mark.training
def test_training_module_uses_model_losses_and_keras_adam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakePixelVAE()
    monkeypatch.setattr(lightning_module, "PixelVAEModel", lambda _config: fake_model)
    module = PixelVAETrainingModule(learning_rate=1e-4)
    logged: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        module, "log", lambda name, value, **_kwargs: logged.__setitem__(name, value)
    )
    batch = {"pixel_values": torch.zeros(2, 4, 256, 256)}

    loss = module.training_step(batch, 0)
    module.validation_step(batch, 0)
    optimizer = module.configure_optimizers()

    assert loss.item() == 1.0
    assert fake_model.sample_values == [True, True]
    assert set(logged) == {
        "train_loss",
        "train_reconstruction",
        "train_kl_divergence",
        "val_reconstruction",
    }
    assert isinstance(optimizer, KerasAdam)
    assert optimizer.param_groups[0]["lr"] == 1e-4


@pytest.mark.training
def test_training_config_instantiates_package_classes(tmp_path: Path) -> None:
    cli = lightning_cli_class()(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        args=[
            "--config",
            "models/pixel-vae/configs/training/pixel_vae_crello.yaml",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.logger=false",
            f"--data.init_args.data_dir={tmp_path}",
        ],
    )

    assert isinstance(cli.model, PixelVAETrainingModule)
    assert isinstance(cli.datamodule, PixelVAEDataModule)
    assert cli.trainer.max_epochs == 250
    assert cli.datamodule.batch_size == 64
    assert cli.datamodule.num_workers == 6
    assert cli.datamodule.pin_memory
    callbacks = cast(Sequence[Callback], getattr(cli.trainer, "callbacks"))
    assert any(
        isinstance(callback, PixelVAEEpochTimingCallback) for callback in callbacks
    )
    checkpoints = [
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint)
        and callback.monitor == "val_reconstruction"
    ]
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint.monitor == "val_reconstruction"
    assert checkpoint.mode == "min"
    assert checkpoint.save_weights_only
    resumable_checkpoints = [
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint) and callback.save_last
    ]
    assert len(resumable_checkpoints) == 1
    assert not resumable_checkpoints[0].save_weights_only
    assert resumable_checkpoints[0].every_n_epochs == 1


@pytest.mark.training
def test_epoch_timing_callback_records_sampler_epoch_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = [_png(value) for value in range(41)]
    data_dir = _prepared_data_dir(
        tmp_path,
        {"train": len(images), "val": 2},
        train_images=images,
    )
    original_state = np.random.get_state()
    np.random.seed(0)
    initial_state = np.random.get_state()
    expected = _legacy_pixel_batches(
        images, batch_size=2, epoch_count=13, selected_epochs={10, 11, 12}
    )
    np.random.set_state(initial_state)

    output_path = tmp_path / "epoch-times.jsonl"
    monkeypatch.setenv("PIXELVAE_EPOCH_TIMING_FILE", str(output_path))
    callback = PixelVAEEpochTimingCallback()
    module = cast(L.LightningModule, SimpleNamespace())
    callback.setup(cast(L.Trainer, SimpleNamespace()), module, "fit")
    try:
        data = PixelVAEDataModule(str(data_dir), batch_size=2)
        data.setup("fit")
        loader = data.train_dataloader()
        trainer_state = SimpleNamespace(
            current_epoch=10,
            sanity_checking=False,
            datamodule=data,
        )
        trainer = cast(L.Trainer, trainer_state)
        actual: dict[int, str] = {}

        for epoch in (10, 11, 12):
            trainer_state.current_epoch = epoch
            callback.on_train_epoch_start(trainer, module)
            assert data.samplers["train"].epoch == epoch
            batches = list(islice(iter(loader), 20))
            actual[epoch] = _tensor_hash(cast(Sequence[BatchFeature], batches))
            callback.on_validation_end(trainer, module)
    finally:
        np.random.set_state(original_state)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["epoch"] for record in records] == [11, 12, 13]
    assert [record["sampler_epoch"] for record in records] == [10, 11, 12]
    assert all(record["train_and_validation_wall_seconds"] >= 0 for record in records)
    assert actual == expected


@pytest.mark.training
def test_epoch_timing_callback_handles_missing_start_and_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PIXELVAE_EPOCH_TIMING_FILE", raising=False)
    callback = PixelVAEEpochTimingCallback()
    module = cast(L.LightningModule, SimpleNamespace())
    trainer_state = SimpleNamespace(current_epoch=0, sanity_checking=False)
    trainer = cast(L.Trainer, trainer_state)

    trainer_state.sanity_checking = True
    callback.on_validation_end(trainer, module)
    trainer_state.sanity_checking = False
    callback.on_validation_end(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_validation_end(trainer, module)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.training
def test_trained_encoder_export_round_trip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    module = PixelVAETrainingModule(latent_dim=2)
    pixels = torch.rand(1, 4, 256, 256)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=0,
                save_last=True,
                save_weights_only=False,
            )
        ],
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            TensorDataset(pixels),
            batch_size=1,
            collate_fn=lambda items: {
                "pixel_values": torch.stack([sample[0] for sample in items])
            },
        ),
    )
    checkpoint = checkpoint_dir / "last.ckpt"
    output_dir = tmp_path / "encoder"
    exporter_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "export_trained_encoder.py"
    )
    expected = module.model.encoder.eval()(pixels)
    assert expected.posterior_mean is not None
    assert expected.posterior_log_variance is not None

    subprocess.run(
        [
            sys.executable,
            str(exporter_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )
    restored = PixelVAEEncoder.from_pretrained(output_dir).eval()
    actual = restored(pixels)
    assert actual.posterior_mean is not None
    assert actual.posterior_log_variance is not None

    torch.testing.assert_close(actual.posterior_mean, expected.posterior_mean)
    torch.testing.assert_close(
        actual.posterior_log_variance, expected.posterior_log_variance
    )
