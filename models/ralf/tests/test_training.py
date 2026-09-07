"""Unit tests for package-local RALF training contracts."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from jaxtyping import Shaped
from PIL import Image

lightning = pytest.importorskip("lightning")

from lightning.pytorch import Trainer  # noqa: E402

from ralf import RalfConfig, RalfForConditionalLayoutGeneration  # noqa: E402
from ralf.training.config import RalfTrainingStage  # noqa: E402
from ralf.training.datamodule import (  # noqa: E402
    RalfDataModule,
    RalfSampleValue,
    RalfTrainingDataset,
    _as_image,
    _normalize_for_config,
    collate_training_batch,
    _sorted_layout,
    encode_training_sample,
)
from ralf.training.lightning_module import RalfTrainingModule  # noqa: E402


class _OptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler


def _small_config(*, max_seq_length: int = 2, top_k: int = 1) -> RalfConfig:
    return RalfConfig(
        dataset_name="cgl",
        max_seq_length=max_seq_length,
        num_bin=8,
        d_model=32,
        decoder_d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        top_k=top_k,
    )


def _sample(
    sample_id: str = "sample-0",
) -> dict[str, RalfSampleValue | Shaped[torch.Tensor, ...]]:
    return {
        "id": sample_id,
        "image": torch.zeros(3, 64, 64),
        "saliency": torch.zeros(1, 64, 64),
        "label": [0],
        "center_x": [0.5],
        "center_y": [0.5],
        "width": [0.25],
        "height": [0.25],
    }


def test_package_declares_runtime_jaxtyping_dependency() -> None:
    package = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert "jaxtyping>=0.3" in package["project"]["dependencies"]


def test_training_stage_order_is_strict() -> None:
    assert tuple(RalfTrainingStage) == (
        RalfTrainingStage.s0,
        RalfTrainingStage.s1,
        RalfTrainingStage.s2,
        RalfTrainingStage.s3,
        RalfTrainingStage.s4,
        RalfTrainingStage.s5,
    )


def test_training_image_normalization_and_channel_adaptation() -> None:
    grayscale = _as_image(torch.full((2, 2), 255), channels=3)
    assert grayscale.shape == (3, 2, 2)
    assert torch.allclose(grayscale, torch.ones_like(grayscale))

    pil_image = _as_image(Image.new("L", (2, 2), color=128), channels=1)
    assert pil_image.shape == (1, 2, 2)
    assert torch.allclose(pil_image, torch.full_like(pil_image, 128 / 255))

    expanded = _as_image(torch.ones(1, 2, 2), channels=3)
    truncated = _as_image(torch.ones(4, 2, 2), channels=3)
    assert expanded.shape == (3, 2, 2)
    assert truncated.shape == (3, 2, 2)

    with pytest.raises(TypeError, match="tensor or PIL image"):
        _as_image("not-an-image", channels=2)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        _as_image(torch.ones(1, 2, 2, 2), channels=1)


def test_training_normalization_maps_configured_string_labels() -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    config.id2label = {0: "logo", 1: "text"}
    sample = {
        "label": ["text"],
        "center_x": [0.5],
        "center_y": [0.5],
        "width": [0.2],
        "height": [0.2],
    }

    normalized = _normalize_for_config(sample, config)

    assert normalized["labels"].tolist() == [1]

    config.id2label = None
    with pytest.raises(RuntimeError, match="id2label"):
        _normalize_for_config(sample, config)


def test_encode_training_sample_matches_teacher_forcing_contract() -> None:
    config = RalfConfig(
        dataset_name="cgl",
        max_seq_length=2,
        num_bin=8,
        d_model=32,
        decoder_d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        top_k=1,
    )
    sample = {
        "id": "sample-0",
        "image": torch.zeros(3, 64, 64),
        "saliency": torch.zeros(1, 64, 64),
        "label": [0],
        "center_x": [0.5],
        "center_y": [0.5],
        "width": [0.25],
        "height": [0.25],
    }
    encoded = encode_training_sample(
        sample,
        config=config,
        retrieval_indexes=[0],
        retrieval_samples=[sample],
    )
    assert encoded["input_ids"].shape == (config.max_token_length,)
    assert encoded["labels"].shape == (config.max_token_length,)
    assert encoded["attention_mask"].shape == (config.max_token_length,)
    assert encoded["pixel_values"].shape == (3, 64, 64)
    assert encoded["saliency"].shape == (1, 64, 64)
    indexes = encoded["retrieved"].indexes
    assert indexes is not None
    assert indexes.tolist() == [[0]]


def test_sorted_layout_handles_annotations_and_empty_layouts() -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    annotated = {
        "annotations": {"bbox": [[0.0, 0.0, 10.0, 20.0]], "category": [1]},
        "width": 100,
        "height": 200,
    }
    empty = {"annotations": {"bbox": [], "category": []}, "width": 100, "height": 200}

    sorted_layout = _sorted_layout(annotated, config)
    empty_layout = _sorted_layout(empty, config)

    assert sorted_layout["labels"].tolist() == [1, 0]
    assert sorted_layout["mask"].tolist() == [True, False]
    assert torch.allclose(
        sorted_layout["bbox"][0], torch.tensor([0.05, 0.05, 0.1, 0.1])
    )
    assert empty_layout["labels"].tolist() == [0, 0]
    assert empty_layout["mask"].tolist() == [True, False]


def test_training_dataset_and_dataloaders_load_split_and_retrieval_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    sample = _sample()
    data_root = tmp_path / "data" / "cgl"
    data_root.mkdir(parents=True)
    (data_root / "train-000.parquet").touch()
    (data_root / "val-000.parquet").touch()
    retrieval_path = tmp_path / "train.pt"
    validation_retrieval_path = tmp_path / "validation.pt"
    torch.save({"sample-0": [0]}, retrieval_path)
    torch.save({"sample-0": [0]}, validation_retrieval_path)
    calls: list[tuple[str, str, str]] = []

    def fake_load_dataset(
        name: str, *, data_files: dict[str, list[str]], split: str
    ) -> list[dict[str, RalfSampleValue | Shaped[torch.Tensor, ...]]]:
        calls.append((name, data_files[split][0], split))
        return [sample]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    module = RalfDataModule(
        config=config,
        data_root=str(tmp_path / "data"),
        retrieval_index_path=str(retrieval_path),
        validation_retrieval_index_path=str(validation_retrieval_path),
        batch_size=1,
    )

    module.setup("test")
    assert module.train_dataset is None
    module.setup("fit")
    assert module.train_dataset is not None
    assert module.validation_dataset is not None
    train_batch = next(iter(module.train_dataloader()))
    validation_batch = next(iter(module.val_dataloader()))

    assert train_batch["input_ids"].shape == (1, config.max_token_length)
    assert validation_batch["retrieved"].indexes is not None
    assert {item[2] for item in calls} == {"train", "val"}


def test_training_dataset_and_collator_reject_incomplete_retrieval() -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    sample = _sample()
    dataset = RalfTrainingDataset(
        samples=[sample],
        config=config,
        retrieval_table={"sample-0": []},
        retrieval_samples=[sample],
    )

    assert len(dataset) == 1
    with pytest.raises(ValueError, match="no complete row"):
        _ = dataset[0]

    encoded = encode_training_sample(
        sample,
        config=config,
        retrieval_indexes=[0],
        retrieval_samples=[sample],
    )
    encoded["retrieved"].indexes = None
    with pytest.raises(ValueError, match="retrieval indexes"):
        collate_training_batch([encoded])


def test_s0_package_training_module_owns_package_model() -> None:
    config = RalfConfig(
        dataset_name="cgl",
        max_seq_length=1,
        num_bin=8,
        d_model=32,
        decoder_d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        top_k=1,
    )
    model = RalfForConditionalLayoutGeneration(config)
    module = RalfTrainingModule(
        config=config,
        model=model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        scheduler="none",
    )
    assert module.model is model
    assert all(
        parameter.requires_grad
        for name, parameter in module.model.named_parameters()
        if not name.startswith("layout_encoer.")
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in module.model.named_parameters()
        if name.startswith("layout_encoer.")
    )


def test_training_module_runs_forward_steps_optimizer_and_scheduler() -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    sample = _sample()
    encoded = encode_training_sample(
        sample,
        config=config,
        retrieval_indexes=[0],
        retrieval_samples=[sample],
    )
    batch = collate_training_batch([encoded])
    module = RalfTrainingModule(
        config=config,
        model=RalfForConditionalLayoutGeneration(config),
        learning_rate=1e-4,
        weight_decay=1e-4,
        clip_max_norm=0.05,
        epochs=30,
        scheduler="multi_step",
        scheduler_milestones=(0.7,),
    )

    forward_output = module.forward(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        saliency=batch["saliency"],
        retrieved=batch["retrieved"],
    )
    assert forward_output.logits is not None
    assert forward_output.logits.shape[:2] == batch["input_ids"].shape
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert set(module.latest_step_trace) == {"train_loss", "logits"}
    validation_loss = module.validation_step(batch, 0)
    assert validation_loss.ndim == 0

    configured_value = module.configure_optimizers()
    assert isinstance(configured_value, dict)
    configured = cast(_OptimizerConfig, configured_value)
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert set(scheduler.milestones) == {21}
    assert {group["lr"] for group in optimizer.param_groups} == {1e-5, 1e-4}

    loss.backward()
    calls: list[int] = []

    class Hook:
        def on_package_gradients_clipped(self, pl_module: RalfTrainingModule) -> None:
            assert pl_module is module
            calls.append(1)

    module._gradient_trace_hook = Hook()
    module.configure_gradient_clipping(optimizer)
    assert calls == [1]

    module.scheduler = "none"
    assert isinstance(module.configure_optimizers(), torch.optim.Optimizer)


def test_scheduler_milestone_follows_the_trainer_epoch_count() -> None:
    config = _small_config(max_seq_length=2, top_k=1)
    module = RalfTrainingModule(
        config=config,
        model=RalfForConditionalLayoutGeneration(config),
        epochs=70,
        scheduler="multi_step",
        scheduler_milestones=(0.7,),
    )

    detached = cast(_OptimizerConfig, module.configure_optimizers())["lr_scheduler"]
    assert isinstance(detached, torch.optim.lr_scheduler.MultiStepLR)
    assert set(detached.milestones) == {49}

    module.trainer = Trainer(
        max_epochs=30,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    attached = cast(_OptimizerConfig, module.configure_optimizers())["lr_scheduler"]

    assert isinstance(attached, torch.optim.lr_scheduler.MultiStepLR)
    assert set(attached.milestones) == {21}


def test_training_dataset_rejects_missing_retrieval_rows() -> None:
    with pytest.raises(ValueError, match="retrieval"):
        RalfTrainingDataset(
            samples=[],
            config=RalfConfig(max_seq_length=1, top_k=1),
            retrieval_table={},
            retrieval_samples=[],
        )


def test_sorted_layout_matches_vendor_transform_order() -> None:
    config = RalfConfig(dataset_name="cgl", max_seq_length=4)
    sample = {
        "label": [2, 3, 2],
        "center_x": [0.5, 0.5, 0.5],
        "center_y": [0.8, 0.1, 0.2],
        "width": [0.1, 0.1, 0.1],
        "height": [0.1, 0.1, 0.1],
    }

    sorted_layout = _sorted_layout(sample, config)

    assert sorted_layout["labels"].tolist() == [3, 2, 2, 0]


def test_sorted_layout_uses_vendor_source_float_order_for_ties() -> None:
    config = RalfConfig(dataset_name="cgl", max_seq_length=2)
    sample = {
        "label": [2, 2],
        "center_x": [0.4, 0.4],
        "center_y": [0.398, 0.3993333333333333],
        "width": [0.1, 0.1],
        "height": [0.028, 0.0306666666666667],
    }

    sorted_layout = _sorted_layout(sample, config)

    assert torch.equal(
        sorted_layout["bbox"],
        torch.tensor(
            [
                [0.4, 0.3993333333333333, 0.1, 0.0306666666666667],
                [0.4, 0.398, 0.1, 0.028],
            ],
            dtype=torch.float32,
        ),
    )
