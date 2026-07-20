"""Documentation generation tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
from pytest import MonkeyPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_REF_PAGES = REPO_ROOT / "scripts/gen_ref_pages.py"


def _load_gen_ref_pages() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_ref_pages", GEN_REF_PAGES)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gen_ref_pages_writes_standalone_api_tree(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "[project]\nname = 'fake-project'\n",
        encoding="utf-8",
    )
    (member_dir / "README.md").write_text("# Fake Project\n", encoding="utf-8")
    (member_dir / "REPRODUCING.md").write_text(
        "# Reproducing Fake Project\n\nRun parity checks.\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text(
        "from .public import PublicThing\n",
        encoding="utf-8",
    )
    (package_dir / "public.py").write_text(
        "class PublicThing:\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gen_ref_pages, "ROOT", tmp_path)
    monkeypatch.setattr(gen_ref_pages, "GENERATED_API_DIR", tmp_path / "docs" / "api")

    gen_ref_pages.main()

    assert (tmp_path / "docs/api/index.md").is_file()
    assert (tmp_path / "docs/api/models/fake-project/index.md").is_file()
    package_index = (tmp_path / "docs/api/models/fake-project/index.md").read_text(
        encoding="utf-8"
    )
    assert (
        "[Reproduce original-implementation parity](reproducing.md)"
        in package_index
    )
    assert (tmp_path / "docs/api/models/fake-project/reproducing.md").read_text(
        encoding="utf-8"
    ) == "# Reproducing Fake Project\n\nRun parity checks.\n"
    assert (tmp_path / "docs/api/models/fake-project/package.md").read_text(
        encoding="utf-8"
    ) == "# `fake_pkg`\n\n::: fake_pkg\n"
    assert (tmp_path / "docs/api/models/fake-project/public.md").read_text(
        encoding="utf-8"
    ) == "# `fake_pkg.public`\n\n::: fake_pkg.public\n"
    assert not (tmp_path / "docs/api/SUMMARY.md").exists()


def test_gen_ref_pages_requires_reproducing_for_model_packages(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "[project]\nname = 'fake-project'\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(gen_ref_pages, "ROOT", tmp_path)
    monkeypatch.setattr(gen_ref_pages, "GENERATED_API_DIR", tmp_path / "docs" / "api")

    with pytest.raises(
        FileNotFoundError,
        match=r"Model package models/fake must include REPRODUCING\.md",
    ):
        gen_ref_pages.main()
