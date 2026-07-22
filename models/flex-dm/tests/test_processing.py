"""Flex-DM processor tests."""

import pytest
import torch

from flex_dm import FlexDmProcessor
from flex_dm.modeling_flex_dm import FlexDmModelOutput

from flex_dm.testing import tiny_config


def test_processor_encodes_public_layout_and_feature_alias() -> None:
    """Processor converts public xywh to internal ltwh bins and masks."""
    processor = FlexDmProcessor.from_config(tiny_config())
    encoded = processor(
        condition_type="pos",
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        labels=torch.tensor([[1]]),
        mask=torch.tensor([[True]]),
    )

    inputs = encoded["inputs"]
    masks = encoded["masks"]
    assert inputs["left"].shape == (1, 1, 1)
    assert masks["left"].item() is True
    assert encoded["condition_type"] == "completion"
    assert encoded["feature_group"] == "pos"


def test_processor_post_process_document_returns_intermediates() -> None:
    """Post-processing returns common schema and stores attributes."""
    config = tiny_config()
    processor = FlexDmProcessor.from_config(config)
    encoded = processor(
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        labels=torch.tensor([[1]]),
        mask=torch.tensor([[True]]),
        feature_group="type",
        return_tensors="pt",
    )
    inputs = encoded["inputs"]
    logits = {}
    for key, column in config.input_columns.items():
        if not column["is_sequence"]:
            continue
        if column["type"] == "categorical":
            logits[key] = torch.zeros(
                1, 1, int(column["shape"][-1]), int(column["input_dim"])
            )
        else:
            logits[key] = torch.zeros(1, 1, int(column["shape"][-1]))
    output = processor.post_process_document(
        FlexDmModelOutput(logits=logits),
        original_inputs=inputs,
        masks=encoded["masks"],
        return_intermediates=True,
    )

    assert output.bbox.shape == (1, 1, 4)
    assert output.mask.tolist() == [[True]]
    assert "attributes" in output.intermediates


def test_processor_save_load_from_vocabulary_and_error_branches(tmp_path) -> None:
    """Processor serialization and validation branches work locally."""
    processor = FlexDmProcessor.from_vocabulary(
        dataset_name="crello",
        vocabulary={"type": ["a", "b"]},
        checkpoint_variant="ours-exp",
    )
    processor.save_pretrained(tmp_path)
    loaded = FlexDmProcessor.from_pretrained(tmp_path)

    assert loaded.config.checkpoint_variant == "ours-exp"
    assert loaded.config.id2label[1] == "b"
    encoded = loaded(
        condition_type="content_image",
        num_elements=2,
        batch_size=1,
    )
    assert encoded["feature_group"] == "img"
    with pytest.raises(ValueError, match="return_tensors"):
        loaded(return_tensors="np")
    output = loaded.post_process_document(
        FlexDmModelOutput(logits={}),
        original_inputs=encoded["inputs"],
        masks=encoded["masks"],
        output_type="dict",
    )
    assert set(output) >= {"bbox", "labels", "mask", "id2label"}
    with pytest.raises(ValueError, match="output_type"):
        loaded.post_process_document(
            FlexDmModelOutput(logits={}),
            original_inputs=encoded["inputs"],
            masks=encoded["masks"],
            output_type="bad",
        )
