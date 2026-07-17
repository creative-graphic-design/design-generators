import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_readme_includes_reproducible_vendor_parity_commands():
    text = (ROOT / "models" / "layout-corrector" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "## Reproducing Vendor Parity" in text
    assert "models/layout-corrector/scripts/download_original.py" in text
    assert "models/layout-corrector/tests/vendor_parity" in text
    assert (
        "CUDA_VISIBLE_DEVICES=5 uv run --package layout-corrector --extra vendor pytest"
        in text
    )
    assert "uv run --package layout-corrector python - <<'PY'" in text


def test_scripts_have_module_docstrings():
    scripts_dir = ROOT / "models" / "layout-corrector" / "scripts"
    for path in scripts_dir.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(module), path
