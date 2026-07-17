import tempfile

from layout_flow.model_card import layoutflow_model_card, save_layoutflow_model_card


def test_layoutflow_model_card_contains_required_metadata() -> None:
    card = str(layoutflow_model_card("publaynet"))

    assert "creative-graphic-design/layout-flow-publaynet" in card
    assert "creative-graphic-design/PubLayNet" in card
    assert "LayoutFlow: Flow Matching For Layout Generation" in card
    assert "guerreiro2024layoutflow" in card
    assert "| publaynet | n/a | Euler exact | 0 | 0 |" in card


def test_save_layoutflow_model_card_writes_readme() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        readme = save_layoutflow_model_card(tmp, dataset="rico25")

        assert readme.name == "README.md"
        assert "creative-graphic-design/layout-flow-rico25" in readme.read_text()
