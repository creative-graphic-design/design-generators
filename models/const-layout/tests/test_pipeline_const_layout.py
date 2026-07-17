import torch

from layout_generation_common.testing import assert_layout_output_schema
from const_layout import (
    ConstLayoutConfig,
    ConstLayoutForGeneration,
    ConstLayoutPipeline,
)


def test_pipeline_contract_and_save_load(tmp_path):
    config = ConstLayoutConfig(
        dataset_name="publaynet",
        latent_size=4,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    model = ConstLayoutForGeneration(config).eval()
    pipeline = ConstLayoutPipeline(model)
    out = pipeline(labels=[["text", "figure"]], seed=0)
    assert_layout_output_schema(out, batch_size=1)
    pipeline.save_pretrained(str(tmp_path))
    loaded_pipeline = ConstLayoutPipeline.from_pretrained(str(tmp_path))
    loaded = loaded_pipeline(labels=[["text", "figure"]], seed=0)
    assert loaded.bbox.shape == out.bbox.shape


def test_generator_wins_over_seed():
    model = ConstLayoutForGeneration(
        ConstLayoutConfig(
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
