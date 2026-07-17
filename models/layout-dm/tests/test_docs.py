import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_readme_includes_reproducible_vendor_parity_commands():
    text = (ROOT / "models" / "layout-dm" / "README.md").read_text(encoding="utf-8")

    assert "## Reproducing Vendor Parity" in text
    assert "scripts/download_original.py" in text
    assert "models/layout-dm/tests/vendor_parity" in text
    assert "CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm pytest" in text
    assert "uv run --package layout-dm python - <<'PY'" in text


def test_scripts_have_module_docstrings():
    scripts_dir = ROOT / "models" / "layout-dm" / "scripts"
    for path in scripts_dir.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(module), path
