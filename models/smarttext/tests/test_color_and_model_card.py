from PIL import Image

from smarttext.color import choose_text_color
from smarttext.model_card import build_smarttext_model_card


def test_choose_text_color_handles_empty_and_bright_crops():
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


def test_build_smarttext_model_card_contains_usage():
    card = build_smarttext_model_card(parity_results={"cases": 0})

    assert "SmartText" in card
    assert "SmartTextPipeline" in card
