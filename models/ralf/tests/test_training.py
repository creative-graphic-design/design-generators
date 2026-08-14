"""Unit tests for package-local RALF training contracts."""

from __future__ import annotations

import pytest
import torch

lightning = pytest.importorskip("lightning")

from ralf import RalfConfig, RalfForConditionalLayoutGeneration  # noqa: E402
from ralf.training.config import (  # noqa: E402
    RalfTrainingConfig,
    RalfTrainingStage,
)
from ralf.training.datamodule import (  # noqa: E402
    RalfTrainingDataset,
    _sorted_layout,
    encode_training_sample,
)
from ralf.training.lightning_module import RalfTrainingModule  # noqa: E402


def test_training_stage_order_is_strict() -> None:
    assert tuple(RalfTrainingStage) == (
        RalfTrainingStage.s0,
        RalfTrainingStage.s1,
        RalfTrainingStage.s2,
        RalfTrainingStage.s3,
        RalfTrainingStage.s4,
        RalfTrainingStage.s5,
    )
    assert RalfTrainingConfig.stage_order() == tuple(RalfTrainingStage)


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
    assert encoded["retrieved"].indexes.tolist() == [[0]]


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
