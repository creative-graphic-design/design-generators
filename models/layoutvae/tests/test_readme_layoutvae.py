from pathlib import Path


def test_readme_has_required_sections():
    text = Path("models/layoutvae/README.md").read_text(encoding="utf-8")
    assert "# Model Card for LayoutVAE" in text
    assert "## Reproducibility" in text
    assert "### Parity Results" in text
    assert (
        "layoutvae @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layoutvae"
        in text
    )
