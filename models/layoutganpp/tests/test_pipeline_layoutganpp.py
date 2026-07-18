import torch
import pytest

from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.common.testing import assert_layout_output_schema
from layoutganpp import (
    ConditionType,
    LayoutGANPPConfig,
    LayoutGANPPModel,
    LayoutGANPPPipeline,
    OutputType,
)


def test_pipeline_contract_and_save_load(tmp_path):
    config = LayoutGANPPConfig(
        dataset_name="publaynet",
        latent_size=4,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    model = LayoutGANPPModel(config).eval()
    pipeline = LayoutGANPPPipeline(model)
    out = pipeline(labels=[["text", "figure"]], seed=0)
    assert isinstance(out, LayoutGenerationOutput)
    assert_layout_output_schema(out, batch_size=1)
    pipeline.save_pretrained(str(tmp_path))
    loaded_pipeline = LayoutGANPPPipeline.from_pretrained(str(tmp_path))
    loaded = loaded_pipeline(labels=[["text", "figure"]], seed=0)
    assert isinstance(loaded, LayoutGenerationOutput)
    assert loaded.bbox.shape == out.bbox.shape


def test_pipeline_preprocess_forward_postprocess_and_tensor_call():
    model = LayoutGANPPModel(
        LayoutGANPPConfig(
            dataset_name="publaynet",
            latent_size=4,
            d_model=16,
            nhead=4,
            num_layers=1,
        )
    ).eval()
    pipeline = LayoutGANPPPipeline(model)
    encoded = pipeline.preprocess(["text"], seed=0, output_type=OutputType.dict)
    assert encoded["labels"].shape == (1, 1)
    forwarded = pipeline._forward(dict(encoded))
    assert isinstance(forwarded, dict)
    bbox = forwarded["bbox"]
    assert isinstance(bbox, torch.Tensor)
    assert bbox.shape == (1, 1, 4)
    assert pipeline.postprocess(forwarded, ignored=True) is forwarded
    out = pipeline(
        labels=torch.tensor([0, 1]),
        mask=torch.tensor([True, False]),
        condition_type=ConditionType.label,
    )
    assert isinstance(out, LayoutGenerationOutput)
    assert out.mask.tolist() == [[True, False]]
    with pytest.raises(ValueError, match="labels are required"):
        pipeline.preprocess()
    with pytest.raises(ValueError, match="labels are required"):
        pipeline()


def test_generator_wins_over_seed():
    model = LayoutGANPPModel(
        LayoutGANPPConfig(
            dataset_name="publaynet",
            latent_size=4,
            d_model=16,
            nhead=4,
            num_layers=1,
        )
    ).eval()
    labels = torch.tensor([[0, 1]])
    generator1 = torch.Generator().manual_seed(7)
    generator2 = torch.Generator().manual_seed(7)
    out1 = model.generate(labels=labels, generator=generator1, seed=1)
    out2 = model.generate(labels=labels, generator=generator2, seed=999)
    torch.testing.assert_close(out1.bbox, out2.bbox)
