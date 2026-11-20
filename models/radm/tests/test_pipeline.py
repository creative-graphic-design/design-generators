import tempfile
from typing import cast

import torch
from PIL import Image

from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from radm import RADMConfig, RADMDenoiser, RADMPipeline
from radm.conversion import build_pipeline


def tiny_config() -> RADMConfig:
    return RADMConfig(
        num_proposals=3,
        hidden_dim=8,
        text_feature_dim=4,
        image_size=32,
        size_divisibility=16,
        pixel_mean=(100.0, 110.0, 120.0),
        pixel_std=(10.0, 20.0, 30.0),
        class_threshold=0.42,
        nms_threshold=0.23,
        num_train_timesteps=10,
        inference_steps=2,
        backbone_depth=18,
        processor_subfolder="tiny_processor",
    )


def tiny_pipeline() -> RADMPipeline:
    return build_pipeline(tiny_config())


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


def test_pipeline_to_and_save_pretrained_round_trip() -> None:
    image = Image.new("RGB", (16, 16), "white")
    config = tiny_config()
    pipe = tiny_pipeline().to("cpu")
    components = pipe.components
    assert set(components) == {"denoiser", "scheduler"}
    assert components["denoiser"] is pipe.denoiser
    assert components["scheduler"] is pipe.scheduler

    with tempfile.TemporaryDirectory() as tmp:
        pipe.save_pretrained(tmp)
        loaded = RADMPipeline.from_pretrained(tmp)
        out = cast(LayoutGenerationOutput, loaded(image, seed=0, num_inference_steps=2))

    processor_fields = (
        "image_size",
        "size_divisibility",
        "pixel_mean",
        "pixel_std",
        "class_threshold",
        "nms_threshold",
        "processor_subfolder",
    )
    expected_processor_config = {
        field: getattr(config, field) for field in processor_fields
    }
    assert {
        field: getattr(pipe.processor.config, field) for field in processor_fields
    } == expected_processor_config
    assert {
        field: getattr(loaded.processor.config, field) for field in processor_fields
    } == expected_processor_config
    assert_layout_output_schema(out, batch_size=1)


def test_denoiser_save_pretrained_round_trips_sample_step() -> None:
    config = RADMConfig(
        num_proposals=2,
        hidden_dim=8,
        text_feature_dim=4,
        image_size=32,
        num_train_timesteps=10,
        inference_steps=2,
        sample_step=7,
        backbone_depth=18,
    )
    denoiser = RADMDenoiser(config=config)

    with tempfile.TemporaryDirectory() as tmp:
        denoiser.save_pretrained(tmp)
        loaded = RADMDenoiser.from_pretrained(tmp)

    assert loaded.config.sample_step == 7
