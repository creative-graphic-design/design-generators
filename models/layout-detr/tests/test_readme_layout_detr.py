from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_required_layout_detr_contracts():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "creative-graphic-design/layout-detr-ad-banner" in text
    assert "### Parity Results" in text
    assert "tolerance-verified" in text
    assert "852 loaded keys" in text
    assert "max_abs=1.49e-7" in text
    assert "max_abs=1.19e-7" in text
    assert "https://github.com/salesforce/LayoutDETR" in text
    assert "```bibtex" in text


def test_reproducing_has_ordered_commands():
    text = (ROOT / "REPRODUCING.md").read_text(encoding="utf-8")

    assert (
        "Workflow order: download assets, generate references, run parity checks, convert checkpoints"
        in text
    )
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "uv run --package layout-detr" in text
    assert "from_pretrained" in text
