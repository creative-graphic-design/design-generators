"""Tests for generated API reference discovery rules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_REF_PAGES = REPO_ROOT / "scripts" / "gen_ref_pages.py"


def _load_gen_ref_pages() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_ref_pages", GEN_REF_PAGES)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_imported_public_modules_accepts_absolute_self_imports(tmp_path: Path) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    package = tmp_path / "layout_gpt"
    package.mkdir()
    init_file = package / "__init__.py"
    init_file.write_text(
        "\n".join(
            [
                "from layout_gpt.agent import LayoutGPTAgent",
                "from layout_gpt.enums import ICLType",
                "from .schema import LayoutGPTOutput",
                "import layout_gpt.types",
            ]
        ),
        encoding="utf-8",
    )

    assert gen_ref_pages.imported_public_modules(init_file) == {
        "agent",
        "enums",
        "schema",
        "types",
    }


def test_model_conversion_modules_are_documented(tmp_path: Path) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    package = tmp_path / "layout_dm"
    package.mkdir()
    conversion = package / "conversion.py"
    conversion.write_text('"""Conversion helpers."""\n', encoding="utf-8")

    assert gen_ref_pages.should_document_source(
        conversion,
        package,
        "Models",
        imported_modules=set(),
    )
