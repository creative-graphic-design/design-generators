from pathlib import Path


def test_readme_has_required_model_card_sections():
    text = Path("models/ds-gan/README.md").read_text(encoding="utf-8")

    assert "## Parity Results" in text
    assert "## Reproducibility" in text
    assert "creative-graphic-design/ds-gan-pku-posterlayout" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
