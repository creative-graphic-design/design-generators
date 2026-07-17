import torch
import pytest

from layout_generation_common.testing import assert_layout_output_schema
from const_layout import ConstLayoutConfig, ConstLayoutForGeneration


def make_model() -> ConstLayoutForGeneration:
    config = ConstLayoutConfig(
        dataset_name="publaynet",
        latent_size=4,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    return ConstLayoutForGeneration(config).eval()


def test_forward_bbox_range_and_shape():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    latents = torch.zeros(1, 3, model.config.latent_size)
    out = model(latents=latents, labels=labels)
    assert out.bbox.shape == (1, 3, 4)
    assert torch.all((0 <= out.bbox) & (out.bbox <= 1))
    assert torch.equal(out.mask, torch.ones(1, 3, dtype=torch.bool))


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
    assert_layout_output_schema(out1, batch_size=1)
    torch.testing.assert_close(out1.bbox, out2.bbox)


def test_explicit_latents_are_deterministic():
    model = make_model()
    labels = torch.tensor([[0, 1, 2]])
    latents = torch.randn(1, 3, model.config.latent_size)
    out1 = model.generate(labels=labels, latents=latents, seed=1)
    out2 = model.generate(labels=labels, latents=latents, seed=999)
    torch.testing.assert_close(out1.bbox, out2.bbox)


def test_invalid_inputs_raise():
    model = make_model()
    with pytest.raises(ValueError, match="labels are required"):
        model.generate()
    with pytest.raises(ValueError, match="unconditional"):
        model.generate(labels=torch.tensor([[0]]), condition_type="unconditional")
    with pytest.raises(ValueError, match="outside"):
        model.generate(labels=torch.tensor([[999]]), latents=torch.zeros(1, 1, 4))
