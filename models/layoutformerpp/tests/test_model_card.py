from pathlib import Path

from layoutformerpp.conversion import (
    layoutformerpp_hub_id,
    layoutformerpp_model_card,
    write_layoutformerpp_model_card,
)


def test_layoutformerpp_model_card_sections() -> None:
    card = layoutformerpp_model_card(dataset="rico", task="gen_t")
    metadata = card.data.to_dict()
    text = str(card)

    assert (
        layoutformerpp_hub_id("rico", "gen_t")
        == "creative-graphic-design/layoutformerpp-rico-gen-t"
    )
    assert metadata["license"] == "mit"
    assert metadata["library_name"] == "transformers"
    assert metadata["datasets"] == ["creative-graphic-design/Rico"]
    assert "LayoutFormerPPForConditionalGeneration.from_pretrained" in text
    assert "LayoutFormer++" in text
    assert "## Parity Summary" in text
    assert "0" in text
    assert "## Citation" in text
    assert "https://github.com/microsoft/LayoutGeneration" in text


def test_write_layoutformerpp_model_card(tmp_path: Path) -> None:
    readme = write_layoutformerpp_model_card(tmp_path, dataset="publaynet", task="ugen")
    text = readme.read_text()

    assert readme.name == "README.md"
    assert "creative-graphic-design/layoutformerpp-publaynet-ugen" in text
    assert "creative-graphic-design/PubLayNet" in text
