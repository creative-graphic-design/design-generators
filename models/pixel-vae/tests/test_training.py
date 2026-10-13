from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import cast

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers.image_processing_utils import BatchFeature

import pixel_vae.training.datamodule as datamodule
import pixel_vae.training.lightning_module as lightning_module
from pixel_vae import PixelVAEEncoder, PixelVAEOutput
from pixel_vae.training import PixelVAEDataModule, PixelVAETrainingModule
from traingen.lightning.cli import lightning_cli_class
from traingen.optim import KerasAdam


def _png(value: int) -> bytes:
    stream = BytesIO()
    Image.new("RGBA", (256, 256), (value, value, value, 255)).save(stream, format="PNG")
    return stream.getvalue()


def _prepared_data_dir(tmp_path: Path, counts: object | None = None) -> Path:
    data_dir = tmp_path / "crello-image"
    data_dir.mkdir()
    (data_dir / "count.json").write_text(
        json.dumps(counts if counts is not None else {"train": 3, "val": 2}),
        encoding="utf-8",
    )
    (data_dir / "train-00000.tfrecord").touch()
    (data_dir / "val-00000.tfrecord").touch()
    return data_dir


def _fake_record_reader(
    records_by_path: dict[str, list[dict[str, bytes]]],
) -> tuple[
    Callable[[str, str | None, dict[str, str]], Iterator[dict[str, bytes]]],
    Callable[[Iterator[Mapping[str, bytes]], int], Iterator[Mapping[str, bytes]]],
]:
    def load(
        path: str, index_path: str | None, description: dict[str, str]
    ) -> Iterator[dict[str, bytes]]:
        assert index_path is None
        assert description == {"image": "byte"}
        return iter(records_by_path[path])

    def shuffle(
        records: Iterator[Mapping[str, bytes]], queue_size: int
    ) -> Iterator[Mapping[str, bytes]]:
        assert queue_size == 3
        return iter(reversed(list(records)))

    return load, shuffle


@pytest.mark.training
def test_datamodule_repeats_full_batches_and_shuffles_only_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    train_images = [_png(value) for value in (1, 2, 3)]
    val_images = [_png(value) for value in (10, 20)]
    records = {
        str(data_dir / "train-00000.tfrecord"): [
            {"image": image} for image in train_images
        ],
        str(data_dir / "val-00000.tfrecord"): [
            {"image": image} for image in val_images
        ],
    }
    load, shuffle = _fake_record_reader(records)
    monkeypatch.setattr(datamodule, "tfrecord_loader", load)
    monkeypatch.setattr(datamodule, "shuffle_iterator", shuffle)
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


@pytest.mark.training
def test_datamodule_deduplicates_to_crello_v1_split_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"train": 89_545, "val": 11_690, "test": 11_428}
    data_dir = _prepared_data_dir(tmp_path, counts)

    def load(
        path: str, index_path: str | None, description: dict[str, str]
    ) -> Iterator[dict[str, bytes]]:
        assert index_path is None
        assert description == {"image": "byte"}
        count = counts[Path(path).name.split("-", maxsplit=1)[0]]
        return (
            {"image": image}
            for value in range(count)
            for image in (value.to_bytes(4, "big"), value.to_bytes(4, "big"))
        )

    monkeypatch.setattr(datamodule, "tfrecord_loader", load)
    monkeypatch.setattr(datamodule, "shuffle_iterator", lambda records, _size: records)
    data = PixelVAEDataModule(str(data_dir), batch_size=64)
    data.setup("fit")

    assert data.datasets["train"].record_count == 89_545
    assert data.datasets["val"].record_count == 11_690
    assert len(data.datasets["train"]) // 64 == 1_400
    assert len(data.datasets["val"]) // 64 == 183

    for split in ("train", "val"):
        count = counts[split]
        records = list(islice(iter(data.datasets[split]), count + 1))

        assert len(set(records[:-1])) == count
        assert records[-1] == records[0]


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
@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([], "split-to-count"),
        ({"train": True, "val": 1}, "positive train count"),
        ({"train": 1, "val": 0}, "positive val count"),
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
    ("records", "message"),
    [
        ([{"image": _png(1)}], "count.json record count exceeds"),
        (
            [
                {"image": _png(1)},
                {"image": _png(2)},
                {"image": _png(3)},
                {"image": _png(4)},
            ],
            "count.json record count is smaller",
        ),
        (
            [
                {"image": cast(bytes, "not bytes")},
                {"image": _png(2)},
                {"image": _png(3)},
            ],
            "image feature must contain bytes",
        ),
    ],
)
def test_datamodule_rejects_invalid_record_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, bytes]],
    message: str,
) -> None:
    data_dir = _prepared_data_dir(tmp_path)
    monkeypatch.setattr(
        datamodule,
        "tfrecord_loader",
        lambda *_args, **_kwargs: records,
    )
    monkeypatch.setattr(datamodule, "shuffle_iterator", lambda items, _size: items)
    data = PixelVAEDataModule(str(data_dir), batch_size=2)
    data.setup("fit")

    with pytest.raises(ValueError, match=message):
        list(data.train_dataloader())


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
    callbacks = cast(Sequence[Callback], getattr(cli.trainer, "callbacks"))
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
