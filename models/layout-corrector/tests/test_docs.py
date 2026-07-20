import ast
from pathlib import Path
from typing import Final


ROOT: Final[Path] = Path(__file__).parents[3]


def test_readme_includes_reproducible_vendor_parity_commands():
    model_dir = ROOT / "models" / "layout-corrector"
    text = (model_dir / "README.md").read_text(encoding="utf-8")
    reproducing = (model_dir / "REPRODUCING.md").read_text(encoding="utf-8")

    assert "## Reproducibility" in text
    assert (
        "Layout-Corrector refines candidate layouts by running a training-free "
        "correction stage on top of [LayoutDM](../layout-dm/)" in text
    )
    assert (
        "pipe = LayoutCorrectorPipeline(layout_dm=layout_dm, corrector=corrector)"
        in text
    )
    assert "return_intermediates=True" in text
    assert "Layout-Corrector confidence score" in text
    assert (
        "https://github.com/creative-graphic-design/design-generators/blob/main/"
        "models/layout-corrector/REPRODUCING.md" in text
    )
    assert (
        "This guide reproduces the original-implementation agreement checks"
        in reproducing
    )
    assert "models/layout-corrector/scripts/download_original.py" in reproducing
    assert (
        "git submodule update --init vendor/layout-corrector vendor/layout-dm" in text
        or "git submodule update --init vendor/layout-corrector vendor/layout-dm"
        in reproducing
    )
    assert "models/layout-corrector/tests/vendor_parity" in reproducing
    assert (
        "CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-corrector --extra vendor pytest"
        in reproducing
    )
    assert (
        "uv run --package layout-corrector python "
        "models/layout-corrector/scripts/smoke_from_pretrained.py" in reproducing
    )


def test_scripts_have_module_docstrings():
    scripts_dir = ROOT / "models" / "layout-corrector" / "scripts"
    for path in scripts_dir.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(module), path
