import torch

from laygen.common.testing import assert_layout_output_schema
from layoutganpp import (
    LayoutGANPPConfig,
    LayoutGANPPModel,
    LayoutGANPPPipeline,
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
    assert_layout_output_schema(out, batch_size=1)
    pipeline.save_pretrained(str(tmp_path))
    loaded_pipeline = LayoutGANPPPipeline.from_pretrained(str(tmp_path))
    loaded = loaded_pipeline(labels=[["text", "figure"]], seed=0)
    assert loaded.bbox.shape == out.bbox.shape


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
