"""Documentation generation tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest
from pytest import MonkeyPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_REF_PAGES = REPO_ROOT / "scripts/gen_ref_pages.py"
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.S)


def _docs_markdown_pages(root: Path = REPO_ROOT) -> list[Path]:
    return [
        path
        for path in sorted((root / "docs").rglob("*.md"))
        if "stylesheets" not in path.relative_to(root / "docs").parts
    ]


def _assert_docs_page_frontmatter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    assert match is not None, f"{path}: missing YAML frontmatter"
    frontmatter = match.group("body")
    assert re.search(r"(?m)^icon:\s+lucide/", frontmatter), (
        f"{path}: frontmatter must include a lucide icon"
    )
    assert re.search(r"(?m)^tags:\n(?:  - .+\n?)+", frontmatter), (
        f"{path}: frontmatter must include non-empty tags"
    )


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
                "    label_size = auto()",
                "    content_image = auto()",
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


def _non_generated_file_snapshot(root: Path) -> dict[Path, bytes]:
    """Snapshot files the API generator is not allowed to create or modify."""
    generated_api_dir = root / "docs" / "api"
    generated_config = root / "mkdocs.generated.yml"
    snapshot: dict[Path, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path == generated_config or generated_api_dir in path.parents:
            continue
        snapshot[path.relative_to(root)] = path.read_bytes()
    return snapshot


def _nav_entries(nav_lines: list[str]) -> list[list[str]]:
    """Return top-level nav entries, retaining their nested lines."""
    entries: list[list[str]] = []
    current: list[str] = []
    for line in nav_lines:
        if line.startswith("  - "):
            if current:
                entries.append(current)
            current = [line]
            continue
        if current:
            current.append(line)
    if current:
        entries.append(current)
    return entries


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


def test_shields_static_badge_messages_use_query_encoding() -> None:
    gen_ref_pages = _load_gen_ref_pages()

    assert "message=label_size" in gen_ref_pages.render_model_overview_badge(
        "label_size", axis="conditions"
    )
    assert "message=content_image" in gen_ref_pages.render_model_overview_badge(
        "content_image", axis="conditions"
    )
    assert (
        "message=content-agnostic-layout-generation"
        in gen_ref_pages.render_model_overview_badge(
            "content-agnostic-layout-generation", axis="task"
        )
    )
    assert gen_ref_pages.render_model_overview_badge(
        "transformers", axis="framework"
    ) == (
        "![framework: transformers]"
        "(https://img.shields.io/static/v1?label=framework&message=transformers"
        "&color=blue&style=flat-square&logo=huggingface&logoColor=white)"
    )
    assert gen_ref_pages.render_model_overview_badge("rico25", axis="datasets") == (
        "![dataset: rico25]"
        "(https://img.shields.io/static/v1?label=dataset&message=rico25"
        "&color=orange&style=flat-square&logo=huggingface&logoColor=white)"
    )


def test_docs_markdown_pages_have_icon_and_tags_frontmatter() -> None:
    gen_ref_pages = _load_gen_ref_pages()
    gen_ref_pages.main()

    for path in _docs_markdown_pages():
        _assert_docs_page_frontmatter(path)


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
                "conditions = ['unconditional', 'label_size']",
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
    (member_dir / "TRAINING.md").write_text(
        "# Training Fake Project\n\nRun training.\n",
        encoding="utf-8",
    )
    (tmp_path / "mkdocs.yml").write_text(
        "\n".join(
            [
                "site_name: fake",
                "nav:",
                "  - Overview: index.md",
                "  - Getting Started: getting-started.md",
                "  - Models: models.md",
                "  - Conventions: conventions.md",
                "  - Architecture: architecture.md",
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
                "[Extending](docs/extending.md)",
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

    for path in _docs_markdown_pages(tmp_path):
        _assert_docs_page_frontmatter(path)

    assert (tmp_path / "docs/api/index.md").is_file()
    assert (tmp_path / "docs/api/models/index.md").is_file()
    assert (tmp_path / "docs/api/models/fake-project/index.md").is_file()
    assert "- [FakeProject](models/fake-project/)" in (
        tmp_path / "docs/api/index.md"
    ).read_text(encoding="utf-8")
    assert "- [FakeProject](fake-project/)" in (
        tmp_path / "docs/api/models/index.md"
    ).read_text(encoding="utf-8")
    package_index = (tmp_path / "docs/api/models/fake-project/index.md").read_text(
        encoding="utf-8"
    )
    assert package_index.startswith(
        "---\nicon: lucide/package\ntags:\n  - Models\n  - API Reference\n  - transformers\n  - content-agnostic-layout-generation\n  - content-aware-layout-generation\n"
    )
    assert (
        "  - unconditional\n  - label_size\n  - rico25\n  - publaynet\n---\n"
        in package_index
    )
    assert "model-index:" not in package_index
    assert (
        "**Reproducing parity:** [Open the model reproducing guide](reproducing/)."
        in package_index
    )
    assert "**Training:** [Open the model training guide](training/)." in package_index
    assert "## Reproducing Guide" not in package_index
    assert (tmp_path / "docs/api/models/fake-project/reproducing.md").read_text(
        encoding="utf-8"
    ) == "\n".join(
        [
            "---",
            "icon: lucide/refresh-cw",
            "tags:",
            "  - Reproducibility",
            "  - Models",
            "---",
            "",
            "# Reproducing Fake Project",
            "",
            "Run parity checks.",
            "",
        ]
    )
    assert (tmp_path / "docs/api/models/fake-project/training.md").read_text(
        encoding="utf-8"
    ) == "\n".join(
        [
            "---",
            "icon: lucide/dumbbell",
            "tags:",
            "  - Training",
            "  - Models",
            "---",
            "",
            "# Training Fake Project",
            "",
            "Run training.",
            "",
        ]
    )
    assert (tmp_path / "docs/api/models/fake-project/package.md").read_text(
        encoding="utf-8"
    ) == "\n".join(
        [
            "---",
            "icon: lucide/file-code",
            "tags:",
            "  - API Reference",
            "  - Models",
            "---",
            "",
            "# `fake_pkg`",
            "",
            "::: fake_pkg",
            "",
        ]
    )
    assert (tmp_path / "docs/api/models/fake-project/public.md").read_text(
        encoding="utf-8"
    ) == "\n".join(
        [
            "---",
            "icon: lucide/file-code",
            "tags:",
            "  - API Reference",
            "  - Models",
            "---",
            "",
            "# `fake_pkg.public`",
            "",
            "::: fake_pkg.public",
            "",
        ]
    )
    assert not (tmp_path / "docs/api/SUMMARY.md").exists()
    generated_config = (tmp_path / "mkdocs.generated.yml").read_text(encoding="utf-8")
    assert "  - Models: models.md" in generated_config
    assert (
        "      - Models:\n          - Overview: api/models/index.md" in generated_config
    )
    assert "  - Getting Started: getting-started.md" in generated_config
    assert "  - Models: models.md" in generated_config
    assert "          - FakeProject:" in generated_config
    assert (
        "              - Overview: api/models/fake-project/index.md" in generated_config
    )
    assert (
        "              - Reproducing: api/models/fake-project/reproducing.md"
        in generated_config
    )
    assert (
        "              - Training: api/models/fake-project/training.md"
        in generated_config
    )
    assert "              - API Modules:" in generated_config
    assert (
        "                  - fake_pkg: api/models/fake-project/package.md"
        in generated_config
    )
    assert (
        "                  - fake_pkg.public: api/models/fake-project/public.md"
        in generated_config
    )


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


def test_gen_ref_pages_preserves_handwritten_files() -> None:
    gen_ref_pages = _load_gen_ref_pages()
    snapshots = _non_generated_file_snapshot(REPO_ROOT)

    gen_ref_pages.main()

    assert _non_generated_file_snapshot(REPO_ROOT) == snapshots


def test_generated_nav_preserves_handwritten_mkdocs_entries() -> None:
    gen_ref_pages = _load_gen_ref_pages()
    source = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
    nav_start = source.index("nav:")
    nav_end = nav_start + 1
    while nav_end < len(source):
        line = source[nav_end]
        if line and not line.startswith((" ", "-")):
            break
        nav_end += 1
    source_entries = _nav_entries(source[nav_start + 1 : nav_end])
    handwritten_entries = [
        entry
        for entry in source_entries
        if not entry[0].startswith("  - API Reference:")
    ]

    gen_ref_pages.main()

    generated = (
        (REPO_ROOT / "mkdocs.generated.yml").read_text(encoding="utf-8").splitlines()
    )
    generated_nav_start = generated.index("nav:")
    generated_nav_end = generated_nav_start + 1
    while generated_nav_end < len(generated):
        line = generated[generated_nav_end]
        if line and not line.startswith((" ", "-")):
            break
        generated_nav_end += 1
    generated_entries = _nav_entries(
        generated[generated_nav_start + 1 : generated_nav_end]
    )
    generated_handwritten_entries = [
        entry
        for entry in generated_entries
        if not entry[0].startswith("  - API Reference:")
    ]
    assert generated_handwritten_entries == handwritten_entries


def test_repo_root_relative_docs_links_are_rewritten_for_site() -> None:
    gen_ref_pages = _load_gen_ref_pages()

    assert (
        gen_ref_pages.site_page_for_repo_link("docs/training-reproduction.md")
        == "training-reproduction/"
    )
    assert (
        gen_ref_pages.rewrite_repo_relative_links(
            "[training](docs/training-reproduction.md)"
        )
        == "[training](training-reproduction/)"
    )
    assert (
        gen_ref_pages.rewrite_repo_relative_links(
            "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&message=ckpt)]"
            "(models/posterllama/REPRODUCING.md)"
        )
        == "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&message=ckpt)]"
        "(api/models/posterllama/reproducing/)"
    )


def test_generated_training_pages_rewrite_repo_relative_links() -> None:
    gen_ref_pages = _load_gen_ref_pages()

    gen_ref_pages.main()

    training_page = REPO_ROOT / "docs" / "api" / "models" / "layout-dm" / "training.md"
    text = training_page.read_text(encoding="utf-8")
    assert "[training reproduction protocol](training-reproduction/)" in text
    assert "](docs/training-reproduction.md)" not in text


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
