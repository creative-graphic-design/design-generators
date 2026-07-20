import tempfile

from layout_flow.model_card import layoutflow_model_card, save_layoutflow_model_card


def test_layoutflow_model_card_contains_required_metadata() -> None:
    model_card = layoutflow_model_card("publaynet")
    card = str(model_card)
    metadata = model_card.data.to_dict()

    assert metadata["license"] == "mit"
    assert metadata["library_name"] == "diffusers"
    assert metadata["pipeline_tag"] == "other"
    assert metadata["language"] == ["en"]
    assert metadata["datasets"] == ["creative-graphic-design/PubLayNet"]
    assert "creative-graphic-design/layout-flow-publaynet" in card
    assert "creative-graphic-design/PubLayNet" in card
    assert "LayoutFlow: Flow Matching For Layout Generation" in card
    assert "## Uses" in card
    assert "### Direct Use" in card
    assert "### Out-of-Scope Use" in card
    assert "## Bias, Risks, and Limitations" in card
    assert "## How to Get Started with the Model" in card
    assert "## Training Details" in card
    assert "## Evaluation" in card
    assert "## Technical Specifications" in card
    assert "guerreiro2024layoutflow" in card
    assert (
        "| publaynet | n/a | Euler trajectory not measured by parity test | 0 | 0 |"
        in card
    )
    assert "LayoutDM" not in card
    assert "CyberAgentAILab" not in card
    assert "2303.08137" not in card
    assert "[More Information Needed]" not in card
    model_card.validate()


def test_save_layoutflow_model_card_writes_readme() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        readme = save_layoutflow_model_card(tmp, dataset="rico25")

        assert readme.name == "README.md"
        assert "creative-graphic-design/layout-flow-rico25" in readme.read_text()
