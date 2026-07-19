import torch
from typing import cast

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline

from ds_gan import DSGANConfig, DSGANModel, DSGANPipeline, DSGANProcessor
from ds_gan.pipeline_ds_gan import normalize_condition_type


def tiny_config() -> DSGANConfig:
    return DSGANConfig(
        backbone="resnet18",
        max_elem=4,
        hidden_size=32,
        num_layers=2,
        image_size=(64, 64),
        backbone_feature_size=16,
    )


def test_pipeline_subclasses_shared_base():
    assert issubclass(DSGANPipeline, LayoutGenerationPipeline)


def test_condition_aliases_and_rejections():
    assert str(normalize_condition_type("content")) == "content_image"

    try:
        normalize_condition_type("unconditional")
    except ValueError as exc:
        assert "Unsupported DS-GAN condition_type" in str(exc)
    else:
        raise AssertionError("expected unsupported condition to raise")

    try:
        normalize_condition_type("unknown")
    except ValueError as exc:
        assert "Unknown condition_type" in str(exc)
    else:
        raise AssertionError("expected unknown condition to raise")


def test_pipeline_call_returns_layout_schema():
    config = tiny_config()
    pipe = DSGANPipeline(
        DSGANModel(config).eval(),
        processor=DSGANProcessor(image_size=(64, 64)),
    )
    pixel_values = torch.zeros(1, 4, 64, 64)

    output = pipe(pixel_values=pixel_values, seed=0)

    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.shape == (1, 4, 4)
    assert output.labels.shape == output.mask.shape
    assert_layout_output_schema(output)


def test_pipeline_dict_fixed_layout_and_intermediates():
    config = tiny_config()
    pipe = DSGANPipeline(
        DSGANModel(config).eval(),
        processor=DSGANProcessor(image_size=(64, 64)),
    )

    output = pipe(
        pixel_values=torch.zeros(1, 4, 64, 64),
        bbox=[[[0.5, 0.5, 0.2, 0.2]]],
        labels=[[0]],
        output_type="dict",
        return_intermediates=True,
    )

    output = cast(dict[str, object], output)
    bbox = cast(torch.Tensor, output["bbox"])
    intermediates = cast(dict[str, torch.Tensor], output["intermediates"])
    assert bbox.shape == (1, 4, 4)
    assert intermediates["initial_layout"].shape == (1, 4, 2, 4)


def test_pipeline_images_path_and_error_paths():
    config = tiny_config()
    pipe = DSGANPipeline(
        DSGANModel(config).eval(),
        processor=DSGANProcessor(image_size=(64, 64)),
    )
    image = torch.zeros(3, 64, 64)

    output = pipe(images=image, saliency=torch.zeros(64, 64), seed=0)
    assert isinstance(output, LayoutGenerationOutput)

    try:
        pipe()
    except ValueError as exc:
        assert "images or pixel_values are required" in str(exc)
    else:
        raise AssertionError("expected missing image inputs to raise")

    try:
        pipe(pixel_values=torch.zeros(1, 4, 64, 64), output_type="bad")
    except ValueError as exc:
        assert "Unsupported output_type" in str(exc)
    else:
        raise AssertionError("expected bad output type to raise")


def test_pipeline_save_pretrained_from_pretrained_smoke(tmp_path):
    config = tiny_config()
    pipe = DSGANPipeline(
        DSGANModel(config).eval(),
        processor=DSGANProcessor(image_size=(64, 64)),
    )

    pipe.save_pretrained(tmp_path)
    restored = DSGANPipeline.from_pretrained(tmp_path, local_files_only=True)
    output = restored(pixel_values=torch.zeros(1, 4, 64, 64), seed=0)

    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.shape[-1] == 4
    assert output.id2label == {0: "text", 1: "logo", 2: "underlay"}
