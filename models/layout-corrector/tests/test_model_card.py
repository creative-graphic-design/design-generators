import pytest

from layout_corrector.model_card import MODEL_CARD_TEMPLATE, layout_corrector_model_card


def test_static_model_card_template_metadata():
    assert "license: mit" in MODEL_CARD_TEMPLATE
    assert "pipeline_tag: unconditional-layout-generation" in MODEL_CARD_TEMPLATE
    assert "Layout-Corrector" in MODEL_CARD_TEMPLATE
    assert (
        "Training-free Layout-Corrector module for LayoutDM-style"
        in MODEL_CARD_TEMPLATE
    )


@pytest.mark.parametrize(
    ("dataset", "expected_name", "expected_dataset_id"),
    [
        ("rico25", "rico25", "creative-graphic-design/Rico"),
        ("publaynet", "publaynet", "creative-graphic-design/PubLayNet"),
        ("crello-bbox", "crello", "cyberagent/crello"),
    ],
)
def test_layout_corrector_model_card_metadata(
    dataset: str, expected_name: str, expected_dataset_id: str
) -> None:
    card = layout_corrector_model_card(dataset=dataset)

    data = card.data.to_dict()
    assert data["license"] == "mit"
    assert data["library_name"] == "diffusers"
    assert data["pipeline_tag"] == "unconditional-layout-generation"
    assert data["datasets"] == [expected_dataset_id]
    text = str(card)
    assert f"Layout-Corrector {expected_name}" in text
    assert expected_dataset_id in text
    assert "creative-graphic-design/rico25" not in text
    assert "creative-graphic-design/publaynet" not in text
    assert (
        "training-free corrector module for discrete diffusion layout generators"
        in text
    )
    assert "LayoutCorrectorPipeline(layout_dm=layout_dm, corrector=corrector)" in text
