from layout_corrector.model_card import MODEL_CARD_TEMPLATE


def test_static_model_card_template_metadata():
    assert "license: mit" in MODEL_CARD_TEMPLATE
    assert "pipeline_tag: unconditional-layout-generation" in MODEL_CARD_TEMPLATE
    assert "Layout-Corrector" in MODEL_CARD_TEMPLATE
