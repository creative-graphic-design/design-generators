import numpy as np
from PIL import Image, ImageFont
import pytest

from smarttext.candidate_generation import (
    candidate_from_reference_json,
    candidate_to_reference_json,
    generate_candidates,
    prepare_scorer_batch,
    split_prompt_lines,
)
from smarttext.configuration_smarttext import SmartTextConfig


def _system_font_path() -> str:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            ImageFont.truetype(path, size=10)
        except OSError:
            continue
        return path
    pytest.skip("No system TrueType font available for font-path coverage.")


def test_split_prompt_lines_rejects_empty_prompt():
    assert split_prompt_lines("A\nB", (1.0, 0.8)) == ("A", "B")
    with pytest.raises(ValueError):
        split_prompt_lines("\n ", (1.0,))


def test_candidate_generation_and_reference_json_round_trip():
    config = SmartTextConfig(
        grid_num=16,
        min_font_size=10,
        max_font_size=20,
        font_inc_unit=10,
        min_text_area_coef=2,
        max_text_area_coef=50,
    )
    image = Image.new("RGB", (96, 96), "white")
    saliency = np.zeros((96, 96), dtype=np.float32)

    candidates = generate_candidates(
        image,
        saliency,
        prompt="ICME",
        font=ImageFont.load_default(),
        config=config,
    )

    assert candidates
    restored = candidate_from_reference_json(candidate_to_reference_json(candidates[0]))
    assert restored == candidates[0]


def test_candidate_generation_accepts_font_path_and_resizes_saliency():
    config = SmartTextConfig(
        grid_num=16,
        min_font_size=10,
        max_font_size=10,
        min_text_area_coef=2,
        max_text_area_coef=500,
    )
    image = Image.new("RGB", (96, 96), "white")

    candidates = generate_candidates(
        image,
        np.zeros((16, 16), dtype=np.float32),
        prompt="ICME",
        font=_system_font_path(),
        config=config,
    )

    assert candidates


def test_prepare_scorer_batch_uses_expanded_regions_by_default():
    config = SmartTextConfig(
        grid_num=16,
        min_font_size=10,
        max_font_size=10,
        min_text_area_coef=2,
        max_text_area_coef=50,
    )
    image = Image.new("RGB", (96, 96), "white")
    candidates = generate_candidates(
        image,
        np.zeros((96, 96), dtype=np.float32),
        prompt="ICME",
        font=ImageFont.load_default(),
        config=config,
    )[:2]

    pixel_values, boxes, kept = prepare_scorer_batch(image, candidates, config=config)

    assert pixel_values.shape[0] == len(candidates)
    assert boxes[:, 0].tolist() == [float(index) for index in range(len(candidates))]
    assert kept == candidates
