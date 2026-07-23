from pathlib import Path


def test_readme_contains_parity_results_and_install_snippet():
    text = Path("models/layout-fid/README.md").read_text(encoding="utf-8")
    assert "### Parity Results" in text
    assert (
        "layout-fid @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layout-fid"
        in text
    )
