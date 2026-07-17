import tempfile

import torch

from layout_generation_common.outputs_diffusers import LayoutGenerationOutput
from layout_generation_common.testing import assert_layout_output_schema
from layout_flow import (
    LayoutFlowConfig,
    LayoutFlowEulerScheduler,
    LayoutFlowPipeline,
    LayoutFlowTransformerModel,
)


def tiny_pipeline() -> LayoutFlowPipeline:
    config = LayoutFlowConfig(
        dataset_name="publaynet",
        max_length=3,
        latent_dim=8,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
        inference_steps=3,
    )
    model = LayoutFlowTransformerModel(
        num_labels=config.num_labels,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        num_layers=config.num_layers,
    )
    return LayoutFlowPipeline(
        model=model,
        scheduler=LayoutFlowEulerScheduler(num_inference_steps=3),
        config=config,
    )


def test_pipeline_unconditional_smoke_and_seed_reproducibility() -> None:
    pipe = tiny_pipeline()
    out1 = pipe(batch_size=1, num_elements=2, seed=0, num_inference_steps=3)
    out2 = pipe(batch_size=1, num_elements=2, seed=0, num_inference_steps=3)
    assert isinstance(out1, LayoutGenerationOutput)
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.equal(out1.labels, out2.labels)
    assert torch.allclose(out1.bbox, out2.bbox)


def test_pipeline_label_conditioning_and_dict_output() -> None:
    pipe = tiny_pipeline()
    out = pipe(
        condition_type="label",
        labels=[[1, 2]],
        bbox=[[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]],
        mask=[[True, True]],
        num_inference_steps=3,
        output_type="dict",
    )
    assert out["labels"].tolist()[0][:2] == [1, 2]


def test_pipeline_save_pretrained_round_trip() -> None:
    pipe = tiny_pipeline()
    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = LayoutFlowPipeline.from_pretrained(tmp)
        out = loaded(batch_size=1, num_elements=1, seed=1, num_inference_steps=2)
    assert_layout_output_schema(out, batch_size=1)
