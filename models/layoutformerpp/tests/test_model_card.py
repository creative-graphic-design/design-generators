from pathlib import Path

from laygen.common import ConditionType, DatasetName

from layoutformerpp import LayoutFormerPPTask
from layoutformerpp.conversion import (
    layoutformerpp_hub_id,
    layoutformerpp_model_card,
    write_layoutformerpp_model_card,
)


def test_layoutformerpp_model_card_sections() -> None:
    card = layoutformerpp_model_card(
        dataset=DatasetName.rico25, task=LayoutFormerPPTask.gen_t
    )
    metadata = card.data.to_dict()
    text = str(card)

    assert (
        layoutformerpp_hub_id(DatasetName.rico25, ConditionType.label)
        == "creative-graphic-design/layoutformerpp-rico-label"
    )
    assert metadata["license"] == "mit"
    assert metadata["library_name"] == "transformers"
    assert metadata["datasets"] == ["creative-graphic-design/Rico"]
    assert metadata["language"] == ["en"]
    assert card.validate() is None
    assert "LayoutFormerPPPipeline.from_pretrained" in text
    assert "LayoutFormer++" in text
    assert "## Model Details" in text
    assert "## Uses" in text
    assert "## Bias, Risks, and Limitations" in text
    assert "## How to Get Started with the Model" in text
    assert "## Training Details" in text
    assert "## Evaluation" in text
    assert "## Technical Specifications" in text
    assert "0" in text
    assert "## Citation" in text
    assert "every public `rico` and `publaynet`" in text
    assert "full constrained-decoding generation parity" not in text
    assert "More Information Needed" not in text
    assert "https://github.com/microsoft/LayoutGeneration" in text


def test_write_layoutformerpp_model_card(tmp_path: Path) -> None:
    readme = write_layoutformerpp_model_card(tmp_path, dataset="publaynet", task="ugen")
    text = readme.read_text()

    assert readme.name == "README.md"
    assert "creative-graphic-design/layoutformerpp-publaynet-unconditional" in text
    assert "creative-graphic-design/PubLayNet" in text
    assert "## Evaluation" in text
