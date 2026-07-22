from PIL import Image
import numpy as np

from smarttext.color import choose_text_color, contrast_rate, rgb_to_hex
from smarttext.model_card import build_smarttext_model_card


def test_choose_text_color_matches_vendor_fallbacks():
    np.random.seed(0)
    assert (
        choose_text_color(
            Image.new("RGB", (8, 8), "white"), (0, 0, 8, 8), contrast_threshold=5
        )
        == "#000000"
    )
    assert (
        choose_text_color(
            Image.new("RGB", (8, 8), "white"), (0, 0, 0, 0), contrast_threshold=5
        )
        == "#000000"
    )


def test_vendor_color_helpers_round_like_upstream():
    assert rgb_to_hex([255.9, 0.0, 16.2]) == "#FF0010"
    assert contrast_rate([0, 0, 0], [255, 255, 255]) == 21.0


def test_build_smarttext_model_card_contains_usage():
    card = build_smarttext_model_card(parity_results={"cases": 0})

    assert "SmartText" in card
    assert "SmartTextPipeline" in card
