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


def _write_shared_enum_sources(root: Path) -> None:
    """Write the minimal shared enum sources needed by docs generation tests."""
    laygen_common = root / "lib" / "laygen" / "src" / "laygen" / "common"
    posgen_common = root / "lib" / "posgen" / "src" / "posgen" / "common"
    laygen_common.mkdir(parents=True)
    posgen_common.mkdir(parents=True)
    (laygen_common / "conditions.py").write_text(
        "\n".join(
            [
                "from enum import StrEnum, auto",
                "",
                "class ConditionType(StrEnum):",
                "    unconditional = auto()",
                "    label = auto()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (laygen_common / "labels.py").write_text(
        "\n".join(
            [
                "from enum import StrEnum, auto",
                "",
                "class DatasetName(StrEnum):",
                "    rico25 = auto()",
                "    publaynet = auto()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (posgen_common / "labels.py").write_text(
        "\n".join(
            [
                "from enum import StrEnum",
                "",
                "class DatasetName(StrEnum):",
                "    crello = 'crello'",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _patch_docs_generator_root(
    gen_ref_pages: ModuleType,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Point the docs generator module at a temporary repository root."""
    monkeypatch.setattr(gen_ref_pages, "ROOT", tmp_path)
    monkeypatch.setattr(gen_ref_pages, "GENERATED_API_DIR", tmp_path / "docs" / "api")
    monkeypatch.setattr(
        gen_ref_pages,
        "GENERATED_MKDOCS_CONFIG",
        tmp_path / "mkdocs.generated.yml",
    )


def _write_minimal_fake_model(
    tmp_path: Path,
    *,
    pyproject: str,
    with_reproducing: bool = True,
) -> None:
    """Write a minimal fake model package fixture."""
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    if with_reproducing:
        (member_dir / "REPRODUCING.md").write_text("# Reproducing\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: fake\nnav:\n  - Overview: index.md\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")


def test_gen_ref_pages_writes_standalone_api_tree(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "name = 'fake-project'",
                "",
                "[tool.design-generators]",
                "framework = 'transformers'",
                "task = ['content-agnostic-layout-generation', 'content-aware-layout-generation']",
                "conditions = ['unconditional', 'label']",
                "datasets = ['rico25', 'publaynet']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (member_dir / "README.md").write_text(
        "---\nmodel-index:\n  - name: FakeProject\n---\n\n# Model Card for FakeProject\n",
        encoding="utf-8",
    )
    (member_dir / "REPRODUCING.md").write_text(
        "# Reproducing Fake Project\n\nRun parity checks.\n",
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "\n".join(
            [
                "site_name: fake",
                "nav:",
                "  - Overview: index.md",
                "  - API Reference: api/",
                "markdown_extensions:",
                "  - toc",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# Fake Repo",
                "",
                "[Model](models/fake/README.md)",
                "[Guide](models/fake/REPRODUCING.md)",
                "[License](LICENSE)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("Fake license.\n", encoding="utf-8")
    (package_dir / "__init__.py").write_text(
        "from .public import PublicThing\n",
        encoding="utf-8",
    )
    (package_dir / "public.py").write_text(
        "class PublicThing:\n    pass\n",
        encoding="utf-8",
    )

    _patch_docs_generator_root(gen_ref_pages, tmp_path, monkeypatch)

    gen_ref_pages.main()

    assert (tmp_path / "docs/index.md").read_text(encoding="utf-8") == "\n".join(
        [
            "---",
            "icon: lucide/layout-template",
            "tags:",
            "  - Overview",
            "  - Documentation",
            "---",
            "",
            "# Fake Repo",
            "",
            "[Model](api/models/fake-project/index.md)",
            "[Guide](https://github.com/creative-graphic-design/design-generators/blob/main/models/fake/REPRODUCING.md)",
            "[License](https://github.com/creative-graphic-design/design-generators/blob/main/LICENSE)",
            "",
        ]
    )
    assert (tmp_path / "docs/api/index.md").is_file()
    assert (tmp_path / "docs/models.md").is_file()
    assert (tmp_path / "docs/api/models/index.md").is_file()
    assert (tmp_path / "docs/api/models/fake-project/index.md").is_file()
    assert "- [FakeProject](models/fake-project/index.md)" in (
        tmp_path / "docs/api/index.md"
    ).read_text(encoding="utf-8")
    assert "- [FakeProject](fake-project/index.md)" in (
        tmp_path / "docs/api/models/index.md"
    ).read_text(encoding="utf-8")
    package_index = (tmp_path / "docs/api/models/fake-project/index.md").read_text(
        encoding="utf-8"
    )
    assert package_index.startswith(
        "---\ntags:\n  - transformers\n  - content-agnostic-layout-generation\n  - content-aware-layout-generation\n"
    )
    assert (
        "  - unconditional\n  - label\n  - rico25\n  - publaynet\n---\n"
        in package_index
    )
    assert "model-index:" not in package_index
    assert (
        "**Reproducing parity:** [Open the model reproducing guide](reproducing.md)."
        in package_index
    )
    assert "## Reproducing Guide" not in package_index
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
    models_overview = (tmp_path / "docs/models.md").read_text(encoding="utf-8")
    assert (
        "| [FakeProject](api/models/fake-project/index.md) | `transformers` | "
        "`content-agnostic-layout-generation`, `content-aware-layout-generation` | `unconditional`, `label` | "
        "`rico25`, `publaynet` |"
    ) in models_overview
    generated_config = (tmp_path / "mkdocs.generated.yml").read_text(encoding="utf-8")
    assert "  - Models: models.md" in generated_config
    assert (
        "      - Models:\n          - Overview: api/models/index.md" in generated_config
    )
    assert "  - Getting Started: getting-started.md" in generated_config
    assert "  - Models: models.md" in generated_config
    assert (
        "          - FakeProject: api/models/fake-project/index.md" in generated_config
    )
    assert "api/models/fake-project/reproducing.md" not in generated_config
    assert "api/models/fake-project/public.md" not in generated_config


def test_gen_ref_pages_requires_reproducing_for_model_packages(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "name = 'fake-project'",
                "",
                "[tool.design-generators]",
                "framework = 'transformers'",
                "task = 'content-agnostic-layout-generation'",
                "conditions = ['unconditional']",
                "datasets = ['rico25']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: fake\nnav:\n  - Overview: index.md\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Fake Repo\n", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    _patch_docs_generator_root(gen_ref_pages, tmp_path, monkeypatch)

    with pytest.raises(
        FileNotFoundError,
        match=r"Model package models/fake must include REPRODUCING\.md",
    ):
        gen_ref_pages.main()


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


def test_generated_overview_matches_readme_with_rewritten_links() -> None:
    gen_ref_pages = _load_gen_ref_pages()
    expected = gen_ref_pages.rewrite_repo_relative_links(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8").rstrip()
    )

    gen_ref_pages.main()

    assert (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8") == (
        f"{gen_ref_pages.OVERVIEW_FRONTMATTER}\n{expected}\n"
    )


def test_gen_ref_pages_rejects_unknown_model_metadata_values(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    member_dir = tmp_path / "models" / "fake"
    package_dir = member_dir / "src" / "fake_pkg"
    package_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "name = 'fake-project'",
                "",
                "[tool.design-generators]",
                "framework = 'transformers'",
                "task = 'content-agnostic-layout-generation'",
                "conditions = ['gen_t']",
                "datasets = ['rico25']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (member_dir / "REPRODUCING.md").write_text("# Reproducing\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: fake\nnav:\n  - Overview: index.md\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(gen_ref_pages, "ROOT", tmp_path)
    monkeypatch.setattr(gen_ref_pages, "GENERATED_API_DIR", tmp_path / "docs" / "api")
    monkeypatch.setattr(
        gen_ref_pages,
        "GENERATED_MKDOCS_CONFIG",
        tmp_path / "mkdocs.generated.yml",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"models/fake \[tool\.design-generators\] conditions: "
            r"has unknown values: \['gen_t'\].*Required keys: framework, task, "
            r"conditions, datasets.*Example: \[tool\.design-generators\]"
        ),
    ):
        gen_ref_pages.main()


def test_gen_ref_pages_requires_model_metadata_table(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    _write_minimal_fake_model(
        tmp_path,
        pyproject="[project]\nname = 'fake-project'\n",
    )
    _patch_docs_generator_root(gen_ref_pages, tmp_path, monkeypatch)

    with pytest.raises(
        KeyError,
        match=(
            r"models/fake \[tool\.design-generators\] table: is required.*"
            r"Required keys: framework, task, conditions, datasets.*"
            r"Example: \[tool\.design-generators\]"
        ),
    ):
        gen_ref_pages.main()


def test_gen_ref_pages_requires_model_metadata_keys(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    _write_minimal_fake_model(
        tmp_path,
        pyproject="\n".join(
            [
                "[project]",
                "name = 'fake-project'",
                "",
                "[tool.design-generators]",
                "framework = 'transformers'",
                "task = 'content-agnostic-layout-generation'",
                "datasets = ['rico25']",
                "",
            ]
        ),
    )
    _patch_docs_generator_root(gen_ref_pages, tmp_path, monkeypatch)

    with pytest.raises(
        KeyError,
        match=(
            r"models/fake \[tool\.design-generators\] conditions: is required.*"
            r"Required keys: framework, task, conditions, datasets.*"
            r"Example: \[tool\.design-generators\]"
        ),
    ):
        gen_ref_pages.main()


def test_gen_ref_pages_rejects_empty_model_metadata_values(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    gen_ref_pages = _load_gen_ref_pages()
    _write_shared_enum_sources(tmp_path)
    _write_minimal_fake_model(
        tmp_path,
        pyproject="\n".join(
            [
                "[project]",
                "name = 'fake-project'",
                "",
                "[tool.design-generators]",
                "framework = 'transformers'",
                "task = 'content-agnostic-layout-generation'",
                "conditions = []",
                "datasets = ['rico25']",
                "",
            ]
        ),
    )
    _patch_docs_generator_root(gen_ref_pages, tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match=(
            r"models/fake \[tool\.design-generators\] conditions: "
            r"must be a non-empty list of strings.*"
            r"Required keys: framework, task, conditions, datasets.*"
            r"Example: \[tool\.design-generators\]"
        ),
    ):
        gen_ref_pages.main()
