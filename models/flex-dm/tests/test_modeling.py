"""Flex-DM model tests."""

import torch

from flex_dm import FlexDmForMaskedDocumentModeling

from flex_dm.testing import tiny_config


def _inputs(config):
    return {
        key: torch.zeros(
            (1, 2, int(column["shape"][-1])),
            dtype=torch.float32 if column["type"] == "numerical" else torch.long,
        )
        for key, column in config.input_columns.items()
    } | {"length": torch.tensor([[1]])}


def test_forward_shapes_and_no_layout_generation_method() -> None:
    """Model forward returns per-head outputs without layout-level APIs."""
    config = tiny_config()
    model = FlexDmForMaskedDocumentModeling(config)
    output = model(inputs=_inputs(config), output_hidden_states=True)

    assert output.logits["left"].shape == (1, 2, 1, 64)
    assert output.logits["image_embedding"].shape == (1, 2, 4)
    assert output.hidden_states.shape == (1, 2, 16)
    assert not hasattr(model, "generate_layout")


def test_save_pretrained_roundtrip(tmp_path) -> None:
    """Standard Transformers save/load works for the model class."""
    config = tiny_config()
    model = FlexDmForMaskedDocumentModeling(config)
    model.save_pretrained(tmp_path)

    loaded = FlexDmForMaskedDocumentModeling.from_pretrained(tmp_path)

    assert loaded.config.model_type == "flex-dm"


def test_forward_tuple_and_loss() -> None:
    """Forward supports tuple output and reconstruction loss."""
    config = tiny_config()
    model = FlexDmForMaskedDocumentModeling(config)
    inputs = _inputs(config)
    labels = {key: value.clone() for key, value in inputs.items() if key != "length"}

    logits, loss = model(inputs=inputs, labels=labels, return_dict=False)

    assert "left" in logits
    assert loss is not None
