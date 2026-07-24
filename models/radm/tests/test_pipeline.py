import tempfile
from typing import cast

import torch
from PIL import Image

from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from radm import RADMConfig, RADMDenoiser, RADMPipeline, RADMProcessor, RADMScheduler


def tiny_pipeline() -> RADMPipeline:
    config = RADMConfig(
        num_proposals=3,
        hidden_dim=8,
        text_feature_dim=4,
        image_size=32,
        num_train_timesteps=10,
        inference_steps=2,
    )
    return RADMPipeline(
        denoiser=RADMDenoiser(num_classes=5, hidden_dim=8, text_feature_dim=4),
        scheduler=RADMScheduler(num_train_timesteps=10, num_inference_steps=2),
        config=config,
        processor=RADMProcessor(config=config),
    )


def test_pipeline_smoke_seed_reproducibility_and_intermediates() -> None:
    image = Image.new("RGB", (16, 16), "white")
    pipe = tiny_pipeline()
    out1 = cast(
        LayoutGenerationOutput,
        pipe(image, seed=0, num_inference_steps=2, return_intermediates=True),
    )
    out2 = cast(LayoutGenerationOutput, pipe(image, seed=0, num_inference_steps=2))
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.equal(out1.labels, out2.labels)
    assert torch.allclose(out1.bbox, out2.bbox)
    assert out1.trajectory is not None


def test_pipeline_dict_output_and_generator_precedence() -> None:
    image = Image.new("RGB", (16, 16), "white")
    pipe = tiny_pipeline()
    generator = torch.Generator().manual_seed(1)
    out = cast(
        dict[str, torch.Tensor],
        pipe(image, seed=0, generator=generator, output_type="dict"),
    )
    assert isinstance(out, dict)
    assert out["bbox"].shape[-1] == 4


def test_pipeline_save_pretrained_round_trip() -> None:
    image = Image.new("RGB", (16, 16), "white")
    pipe = tiny_pipeline()
    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = RADMPipeline.from_pretrained(tmp)
        out = cast(LayoutGenerationOutput, loaded(image, seed=0, num_inference_steps=2))
    assert_layout_output_schema(out, batch_size=1)
