from typing import cast
from collections.abc import Callable

import torch
import pytest

from laygen.common.bbox import BoxFormat
from laygen.common.testing import assert_layout_output_schema
from laygen.outputs.transformers import LayoutGenerationOutput
from layoutganpp import (
    ConditionType,
    LayoutGANPPConfig,
    LayoutGANPPModel,
    OutputType,
    normalize_condition_type,
    normalize_output_type,
)


def make_model() -> LayoutGANPPModel:
    config = LayoutGANPPConfig(
        dataset_name="publaynet",
        latent_size=4,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    return LayoutGANPPModel(config).eval()


def test_forward_bbox_range_and_shape():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    latents = torch.zeros(1, 3, model.config.latent_size)
    out = model(latents=latents, labels=labels)
    assert out.bbox.shape == (1, 3, 4)
    assert torch.all((0 <= out.bbox) & (out.bbox <= 1))
    assert torch.equal(out.mask, torch.ones(1, 3, dtype=torch.bool))
    tuple_out = model(latents=latents, labels=labels, return_dict=False)
    assert tuple_out[0].shape == (1, 3, 4)


def test_attention_mask_and_padding_mask_are_equivalent():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    latents = torch.randn(1, 3, model.config.latent_size)
    attention_mask = torch.tensor([[True, True, False]])
    out_attention = model(latents, labels, attention_mask=attention_mask)
    out_padding = model(latents, labels, padding_mask=~attention_mask)
    torch.testing.assert_close(out_attention.bbox, out_padding.bbox)


def test_generate_seed_is_reproducible():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    out1 = model.generate(labels=labels, seed=123)
    out2 = model.generate(labels=labels, seed=123)
    assert isinstance(out1, LayoutGenerationOutput)
    assert isinstance(out2, LayoutGenerationOutput)
    assert_layout_output_schema(out1, batch_size=1)
    torch.testing.assert_close(out1.bbox, out2.bbox)


def test_explicit_latents_are_deterministic():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    latents = torch.randn(1, 3, model.config.latent_size)
    out1 = model.generate(labels=labels, latents=latents, seed=1)
    out2 = model.generate(labels=labels, latents=latents, seed=999)
    assert isinstance(out1, LayoutGenerationOutput)
    assert isinstance(out2, LayoutGenerationOutput)
    torch.testing.assert_close(out1.bbox, out2.bbox)


def test_invalid_inputs_raise():
    model = make_model()
    with pytest.raises(ValueError, match="labels are required"):
        model.generate()
    with pytest.raises(ValueError, match="unconditional"):
        model.generate(
            labels=torch.tensor([[0]]), condition_type=ConditionType.unconditional
        )
    with pytest.raises(ValueError, match="outside"):
        model.generate(labels=torch.tensor([[999]]), latents=torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="shape"):
        model(latents=torch.zeros(1, 1, 4), labels=torch.tensor([0]))
    with pytest.raises(ValueError, match="shape"):
        model(latents=torch.zeros(1, 1, 4), labels=torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="last dimension"):
        model(latents=torch.zeros(1, 1, 2), labels=torch.tensor([[0]]))
    unsupported_kwargs: dict[str, object] = {"unsupported": True}
    generate = cast(Callable[..., object], model.generate)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        generate(labels=torch.tensor([[0]]), **unsupported_kwargs)
    with pytest.raises(ValueError, match="Unknown condition_type"):
        normalize_condition_type("unknown")
    with pytest.raises(ValueError, match="Unsupported output_type"):
        normalize_output_type("unknown")


def test_generate_enums_masks_and_dict_output():
    model = make_model()
    labels = torch.tensor([0, 1])
    latents = torch.zeros(1, 2, model.config.latent_size)
    out = model.generate(
        labels=labels,
        attention_mask=torch.tensor([True, False]),
        box_format=BoxFormat.xywh,
        output_type=OutputType.dict,
        return_intermediates=True,
        latents=latents,
    )
    assert isinstance(out, dict)
    intermediates = out["intermediates"]
    assert isinstance(intermediates, dict)
    intermediates = cast(dict[str, object], intermediates)
    bbox = out["bbox"]
    mask = out["mask"]
    assert isinstance(bbox, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert bbox.shape == (1, 2, 4)
    assert mask.tolist() == [[True, False]]
    assert intermediates["condition_type"] == ConditionType.label
