import sys
from pathlib import Path
from typing import cast
from types import SimpleNamespace

import pytest
import torch

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.conversion import remap_denoiser_key, split_original_state_dict
from layout_dm.training.datamodule import LayoutDMDataModule
from layout_dm.training.config import (
    LayoutDMSeedMode,
    LayoutDMTimeSampler,
    LayoutDMTrainingDatasetName,
    LayoutDMTrainingDatasetSource,
    LayoutDMTrainingScheduler,
)
from layout_dm.training.dataset import (
    LayoutDMDataset,
    LayoutDMProcessedDataset,
    LayoutDMSyntheticDataset,
    _extract_layout,
)
from layout_dm.training.losses import (
    log_categorical,
    mean_except_batch,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
)
from layout_dm.training.seed import apply_layout_dm_seed_mode


def test_loss_helpers_and_time_sampling_are_finite() -> None:
    log_probs = torch.log_softmax(torch.randn(2, 4, 3), dim=1)
    assert torch.allclose(multinomial_kl(log_probs, log_probs), torch.zeros(2, 3))
    assert mean_except_batch(torch.ones(2, 3, 4)).tolist() == [1.0, 1.0]
    assert log_categorical(log_probs, log_probs).shape == (2, 3)

    t, pt = sample_time_uniform(
        5,
        num_timesteps=7,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    assert t.shape == pt.shape == (5,)
    assert torch.allclose(pt, torch.full((5,), 1 / 7))

    history = torch.arange(1, 8, dtype=torch.float32)
    count = torch.full((7,), 11.0)
    t_imp, pt_imp = sample_time_importance(
        5,
        num_timesteps=7,
        lt_history=history,
        lt_count=count,
        generator=torch.Generator().manual_seed(2),
    )
    assert t_imp.shape == pt_imp.shape == (5,)
    assert torch.isfinite(pt_imp).all()


def test_synthetic_dataset_and_hf_sample_normalization() -> None:
    config = LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=4,
        num_bin_bboxes=8,
        bbox_quantization="linear",
    )
    synthetic = LayoutDMSyntheticDataset(config=config, size=2, elements=2)
    item = synthetic[0]
    input_ids = item["input_ids"]
    attention_mask = item["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.shape == (20,)
    assert attention_mask.sum() == 10

    sample = {
        "annotations": [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "category": "text"},
            {"bbox": [0.4, 0.5, 0.2, 0.2], "category": "figure"},
        ],
        "id": "doc-1",
    }
    bbox, labels, canvas_size = _extract_layout(sample, {"text": 0, "figure": 4})
    assert bbox.shape == (2, 4)
    assert labels.tolist() == [0, 4]
    assert canvas_size is None

    dataset = LayoutDMDataset.__new__(LayoutDMDataset)
    dataset.dataset_name = "publaynet"
    dataset.split = "train"
    dataset.config = config
    dataset.tokenizer = synthetic.tokenizer
    dataset.processor = synthetic.processor
    dataset.max_seq_length = 4
    dataset.box_format = "xywh"
    dataset.normalized = True
    dataset.label2id = {"text": 0, "figure": 4}
    encoded = dataset._encode_sample(sample)
    assert encoded["input_ids"].shape == (20,)
    assert encoded["id"] == "doc-1"


def test_processed_dataset_and_datamodule_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "processed-data"
    processed_dir = data_root / "publaynet-max4" / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "torch_geometric", SimpleNamespace())
    data = SimpleNamespace(
        x=torch.tensor(
            [
                [0.125, 0.25, 0.375, 0.5],
                [0.5, 0.625, 0.25, 0.125],
                [0.25, 0.125, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ),
        y=torch.tensor([1, 2, 3], dtype=torch.long),
        attr={"name": ["row-0", "row-1"]},
    )
    slices = {"x": torch.tensor([0, 2, 3]), "y": torch.tensor([0, 2, 3])}
    for split_name in ["train", "val", "test"]:
        torch.save((data, slices), processed_dir / f"{split_name}.pt")

    config = LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=4,
        num_bin_bboxes=8,
        bbox_quantization="linear",
    )
    dataset = LayoutDMProcessedDataset(
        dataset_name="publaynet",
        config=config,
        processed_data_dir=data_root,
        split="validation",
    )
    assert len(dataset) == 2
    assert dataset[0]["id"] == "row-0"
    assert cast(torch.Tensor, dataset[0]["input_ids"]).shape == (20,)
    assert cast(torch.Tensor, dataset[1]["attention_mask"]).sum() == 5

    dm = LayoutDMDataModule(
        dataset_name="publaynet",
        config=config,
        batch_size=2,
        max_seq_length=4,
        num_workers=0,
        dataset_source="processed",
        processed_data_dir=str(data_root),
    )
    dm.setup("fit")
    assert len(dm.train_dataloader()) == 1
    lazy_dm = LayoutDMDataModule(
        dataset_name="publaynet",
        config=config,
        batch_size=2,
        max_seq_length=4,
        num_workers=0,
        dataset_source="processed",
        processed_data_dir=str(data_root),
    )
    assert len(lazy_dm.train_dataloader()) == 1
    assert len(lazy_dm.val_dataloader()) == 1
    assert len(lazy_dm.test_dataloader()) == 1
    with pytest.raises(RuntimeError):
        lazy_dm._loader(None, shuffle=False)

    with pytest.raises(ValueError):
        LayoutDMDataModule(
            dataset_name="publaynet",
            config=config,
            dataset_source="processed",
            num_workers=0,
        ).setup("fit")


def test_seed_modes_public_options_and_conversion_key_maps() -> None:
    assert LayoutDMSeedMode("default") is LayoutDMSeedMode.default
    apply_layout_dm_seed_mode("default", seed=1)
    apply_layout_dm_seed_mode("deterministic", seed=1)
    assert LayoutDMTrainingDatasetName.__args__ == ("rico25", "publaynet")
    assert LayoutDMTrainingDatasetSource.__args__ == ("hf", "processed")
    assert LayoutDMTrainingScheduler.__args__ == ("reduce_on_plateau",)
    assert LayoutDMTimeSampler.__args__ == ("importance", "uniform")

    assert (
        remap_denoiser_key("model.module.transformer.emb.weight")
        == "transformer.emb.weight"
    )
    assert (
        remap_denoiser_key("model.transformer.emb.weight") == "transformer.emb.weight"
    )
    assert remap_denoiser_key("model.backbone.emb.weight") == "emb.weight"
    state = {
        "model.transformer.emb.weight": torch.ones(1),
        "lt_history": torch.zeros(1),
    }
    converted = split_original_state_dict(state)
    assert set(converted) == {"transformer.emb.weight"}
    assert torch.equal(converted["transformer.emb.weight"], torch.ones(1))
    with pytest.raises(KeyError):
        remap_denoiser_key("unexpected.weight")
