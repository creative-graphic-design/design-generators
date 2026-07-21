import torch
from PIL import Image, ImageFont
import pytest

from laygen.common.testing import assert_layout_output_schema
from smarttext import (
    SmartTextBASNet,
    SmartTextConfig,
    SmartTextPipeline,
    SmartTextProcessor,
    SmartTextScorer,
)


def _pipeline():
    config = SmartTextConfig(
        align_size=3,
        reduction_dim=4,
        grid_num=16,
        min_font_size=10,
        max_font_size=20,
        font_inc_unit=10,
        min_text_area_coef=2,
        max_text_area_coef=50,
        candi_res=1,
    )
    return SmartTextPipeline(
        SmartTextScorer(config),
        SmartTextBASNet(config),
        SmartTextProcessor(config=config),
        config=config,
    )


def test_pipeline_smoke_with_injected_saliency():
    pipe = _pipeline()

    output = pipe(
        Image.new("RGB", (96, 96), "white"),
        prompt="ICME",
        saliency=torch.zeros(96, 96),
        font=ImageFont.load_default(),
        return_intermediates=True,
    )

    assert_layout_output_schema(output)
    assert "raw_scorer_boxes" in output.intermediates


def test_text_spacing_override_does_not_leak_to_later_calls():
    pipe = _pipeline()
    original_spacing = pipe.config.text_spacing
    image = Image.new("RGB", (96, 96), "white")
    saliency = torch.zeros(96, 96)
    font = ImageFont.load_default()

    def first_candidate_line_gap(output):
        lines = output.intermediates["candidates"][0].lines
        return lines[1].bbox_ltrb_px[1] - lines[0].bbox_ltrb_px[3]

    override_output = pipe(
        image,
        prompt="ICME\n2020",
        saliency=saliency,
        font=font,
        text_spacing=1,
        return_intermediates=True,
    )
    default_output = pipe(
        image,
        prompt="ICME\n2020",
        saliency=saliency,
        font=font,
        return_intermediates=True,
    )

    assert pipe.config.text_spacing == original_spacing
    assert first_candidate_line_gap(override_output) == 1
    assert first_candidate_line_gap(default_output) == original_spacing


def test_pipeline_output_dict_and_unsupported_mode():
    pipe = _pipeline()

    output = pipe(
        Image.new("RGB", (96, 96), "white"),
        prompt="ICME",
        saliency=torch.zeros(96, 96),
        font=ImageFont.load_default(),
        output_type="dict",
    )

    assert "bbox" in output
    with pytest.raises(NotImplementedError):
        pipe(Image.new("RGB", (96, 96), "white"), prompt="ICME", condition_type="text")


def test_pipeline_save_pretrained_round_trip(tmp_path):
    pipe = _pipeline()
    pipe.save_pretrained(tmp_path)

    loaded = SmartTextPipeline.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, SmartTextPipeline)


def test_pipeline_basnet_path_and_batch_validation():
    pipe = _pipeline()
    gradient = torch.arange(96 * 96 * 3, dtype=torch.float32).reshape(96, 96, 3)
    gradient = (gradient % 255).to(torch.uint8).numpy()

    output = pipe(
        Image.fromarray(gradient, mode="RGB"),
        prompt="ICME",
        font=ImageFont.load_default(),
        seed=0,
    )

    assert_layout_output_schema(output)
    with pytest.raises(ValueError):
        pipe(
            [Image.new("RGB", (96, 96), "white"), Image.new("RGB", (96, 96), "white")],
            prompt=["A", "B"],
            batch_size=2,
            saliency=torch.zeros(96, 96),
        )


def test_pipeline_rejects_unknown_condition():
    pipe = _pipeline()

    with pytest.raises(ValueError):
        pipe(
            Image.new("RGB", (96, 96), "white"), prompt="ICME", condition_type="unknown"
        )
