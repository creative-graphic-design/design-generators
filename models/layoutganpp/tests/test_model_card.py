from layoutganpp.model_card import (
    layoutganpp_model_card,
    write_layoutganpp_model_card,
)


def test_layoutganpp_model_card_metadata_and_sections():
    card = layoutganpp_model_card("rico")
    metadata = card.data.to_dict()
    text = str(card)

    assert metadata["license"] == "agpl-3.0"
    assert metadata["library_name"] == "transformers"
    assert metadata["datasets"] == ["creative-graphic-design/rico"]
    assert "layoutganpp" in metadata["tags"]
    assert "LayoutGANPPPipeline.from_pretrained" in text
    assert "creative-graphic-design/layoutganpp-rico" in text
    assert "bbox tensors with shape (3, 9, 4)" in text
    assert "max_abs=0 and max_rel=0" in text
    assert "https://github.com/ktrk115/const_layout" in text
    assert "10.1145/3474085.3475497" in text


def test_write_layoutganpp_model_card_smoke(tmp_path):
    readme_path = write_layoutganpp_model_card(tmp_path, "publaynet")
    text = readme_path.read_text(encoding="utf-8")

    assert readme_path.name == "README.md"
    assert "LayoutGAN++ publaynet" in text
    assert "creative-graphic-design/layoutganpp-publaynet" in text
    assert "bbox exact" in text
