import json

import lightning.pytorch as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from canvas_vae.training.update_trace import (
    TRACE_BATCH_FIELDS,
    TrainingUpdateTrace,
    TrainingUpdateTraceCallback,
    _batch_sha256,
    install_tensorflow_s4_train_stream,
)


def _batch() -> dict[str, np.ndarray]:
    return {
        name: np.arange(6, dtype=np.float32 if name == "image_embedding" else np.int64)
        for name in TRACE_BATCH_FIELDS
    }


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def test_batch_sha256_is_stable_and_tracks_order():
    batch = _batch()
    reordered = {name: values[::-1].copy() for name, values in batch.items()}

    assert _batch_sha256(batch) == _batch_sha256(_batch())
    assert _batch_sha256(batch) != _batch_sha256(reordered)


def test_original_s4_stream_disables_training_shuffle(monkeypatch):
    class DataSpec:
        def make_dataset(self, split, *, shuffle=None, repeat=False):
            return split, shuffle, repeat

    original = DataSpec.make_dataset
    monkeypatch.setattr(DataSpec, "make_dataset", original)
    install_tensorflow_s4_train_stream(DataSpec)

    assert DataSpec().make_dataset("train", shuffle=True, repeat=True) == (
        "train",
        False,
        True,
    )
    assert DataSpec().make_dataset("val", shuffle=True) == ("val", True, False)


def test_update_trace_records_consecutive_objectives_and_batch_hashes(tmp_path):
    path = tmp_path / "updates.jsonl"
    trace = TrainingUpdateTrace(path, system="package", seed=7)
    trace.record(1, np.asarray(2.5), _batch())
    trace.record(2, np.asarray(1.5), _batch())
    trace.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["record"] == "metadata"
    assert rows[0]["seed"] == 7
    assert [row["update"] for row in rows[1:]] == [1, 2]
    assert [row["objective"] for row in rows[1:]] == [2.5, 1.5]
    assert rows[1]["batch_sha256"] == rows[2]["batch_sha256"]
    with pytest.raises(ValueError, match="consecutive"):
        trace.record(4, np.asarray(1.0), _batch())


def test_package_callback_records_training_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = tmp_path / "package.jsonl"
    monkeypatch.setenv("CANVAS_VAE_UPDATE_TRACE", str(path))
    monkeypatch.setenv("CANVAS_VAE_UPDATE_TRACE_SEED", "3")

    class TinyTrainingModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.weight * batch["length"].float().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    samples = [
        {
            name: torch.full(
                (3, 256)
                if name == "image_embedding"
                else (3, 3)
                if name == "color"
                else (3,),
                fill_value=index,
                dtype=torch.float32
                if name in {"image_embedding", "color"}
                else torch.int64,
            )
            for name in TRACE_BATCH_FIELDS
        }
        for index in range(4)
    ]
    expected_hashes = [
        _batch_sha256(
            {
                name: torch.stack(
                    [sample[name] for sample in samples[start : start + 2]]
                )
                for name in TRACE_BATCH_FIELDS
            }
        )
        for start in (0, 2)
    ]
    callback = TrainingUpdateTraceCallback()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(
        TinyTrainingModule(),
        train_dataloaders=DataLoader(_TinyDataset(samples), batch_size=2),
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["update"] for row in rows[1:]] == [1, 2]
    assert [row["batch_sha256"] for row in rows[1:]] == expected_hashes
    assert all(np.isfinite(row["objective"]) for row in rows[1:])


def test_original_callback_records_training_batches(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from canvas_vae.training.update_trace import install_tensorflow_training_trace

    class TinyVAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.projection = tf.keras.layers.Dense(1, use_bias=False)

        def call(self, inputs, training=False):
            del training
            return self.projection(tf.cast(inputs["length"][:, :1], tf.float32))

        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
            del y, sample_weight
            if x is None:
                raise ValueError("training features are required")
            target = tf.cast(x["target"][:, None], tf.float32)
            return tf.reduce_mean(tf.square(y_pred - target))

        def train_step(self, data):
            with tf.GradientTape() as tape:
                prediction = self(data, training=True)
                loss = self.compute_loss(x=data, y=None, y_pred=prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return {"loss": loss}

    batch = {
        name: np.full(
            (4, 2, 256)
            if name == "image_embedding"
            else (4, 2, 3)
            if name == "color"
            else (4, 2),
            index,
            dtype=np.float32 if name in {"image_embedding", "color"} else np.int64,
        )
        for index, name in enumerate(TRACE_BATCH_FIELDS)
    }
    batch["target"] = np.arange(4, dtype=np.float32)
    expected_hashes = [
        _batch_sha256(
            {
                name: values[start : start + 2]
                for name, values in batch.items()
                if name in TRACE_BATCH_FIELDS
            }
        )
        for start in (0, 2)
    ]
    path = tmp_path / "original.jsonl"
    model = TinyVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    install_tensorflow_training_trace(TinyVAE, path, seed=5)
    model.fit(tf.data.Dataset.from_tensor_slices(batch).batch(2), epochs=1, verbose=0)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["update"] for row in rows[1:]] == [1, 2]
    assert [row["batch_sha256"] for row in rows[1:]] == expected_hashes
    assert all(np.isfinite(row["objective"]) for row in rows[1:])
