from layout_corrector.model_card import MODEL_CARD_TEMPLATE, layout_corrector_model_card


def test_static_model_card_template_metadata():
    assert "license: mit" in MODEL_CARD_TEMPLATE
    assert "pipeline_tag: unconditional-layout-generation" in MODEL_CARD_TEMPLATE
    assert "Layout-Corrector" in MODEL_CARD_TEMPLATE


def test_layout_corrector_model_card_metadata():
    card = layout_corrector_model_card(dataset="crello-bbox")

    data = card.data.to_dict()
    assert data["license"] == "mit"
    assert data["library_name"] == "diffusers"
    assert data["pipeline_tag"] == "unconditional-layout-generation"
    assert data["datasets"] == ["cyberagent/crello"]
    assert "Layout-Corrector crello" in str(card)
