"""Tests for the small API-page generator and docs-page frontmatter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_API_PAGES = REPO_ROOT / "scripts/gen_api_pages.py"
PUBLISH_DOC_GUIDES = REPO_ROOT / "scripts/publish_doc_guides.py"
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.S)


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_api_pages", GEN_API_PAGES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_guide_publisher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "publish_doc_guides", PUBLISH_DOC_GUIDES
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_frontmatter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    assert match is not None, f"{path}: missing YAML frontmatter"
    body = match.group("body")
    assert re.search(r"(?m)^icon:\s+lucide/", body), path
    assert re.search(r"(?m)^tags:\n(?:  - .+\n?)+", body), path


def _write_workspace(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["lib/*", "models/*"]\n',
        encoding="utf-8",
    )


def _write_member(root: Path, parent: str, directory: str, project: str) -> Path:
    member = root / parent / directory
    package = member / "src" / directory.replace("-", "_")
    package.mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        f"[project]\nname = '{project}'\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    return package


def test_module_discovery_uses_workspace_members_and_excludes_private_files(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    _write_workspace(tmp_path)
    package = _write_member(tmp_path, "models", "fake", "fake-project")
    (package / "public.py").write_text("", encoding="utf-8")
    (package / "_private.py").write_text("", encoding="utf-8")
    (package / "testing.py").write_text("", encoding="utf-8")
    (package / "vendor_parity.py").write_text("", encoding="utf-8")
    (package / "vendor_state.py").write_text("", encoding="utf-8")
    (package / "vendor_state_dict.py").write_text("", encoding="utf-8")
    nested = package / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "item.py").write_text("", encoding="utf-8")
    (nested / "__main__.py").write_text("", encoding="utf-8")

    pages = generator.discover_modules(tmp_path)

    assert [page[0] for page in pages] == [
        "fake",
        "fake.nested",
        "fake.nested.item",
        "fake.public",
    ]
    assert {page[3].as_posix() for page in pages} == {
        "models/fake-project/index.md",
        "models/fake-project/nested/index.md",
        "models/fake-project/nested/item.md",
        "models/fake-project/public.md",
    }


def test_summary_contains_section_index_pages(tmp_path: Path) -> None:
    generator = _load_generator()
    _write_workspace(tmp_path)
    package = _write_member(tmp_path, "lib", "fake-lib", "fake-lib")
    (package / "common").mkdir()
    (package / "common" / "__init__.py").write_text("", encoding="utf-8")
    (package / "common" / "bbox.py").write_text("", encoding="utf-8")

    generator.generate(tmp_path)
    summary = (tmp_path / "docs/api/SUMMARY.md").read_text(encoding="utf-8")

    assert "[fake-lib](libraries/fake-lib/index.md)" in summary
    assert "[common](libraries/fake-lib/common/index.md)" in summary
    assert "[bbox](libraries/fake-lib/common/bbox.md)" in summary


def test_publishes_guides_and_rewrites_repository_links(tmp_path: Path) -> None:
    generator = _load_generator()
    publisher = _load_guide_publisher()
    _write_workspace(tmp_path)
    package = _write_member(tmp_path, "models", "fake", "fake")
    (package / "training").mkdir()
    (package / "training" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/training-reproduction.md").write_text(
        "# Guide\n", encoding="utf-8"
    )
    member = package.parents[1]
    (member / "README.md").write_text(
        "# Fake\n\n[Training](models/fake/TRAINING.md)\n"
        "[Reproducing](models/fake/REPRODUCING.md#quick)\n"
        "[Docs](docs/training-reproduction.md)\n",
        encoding="utf-8",
    )
    (member / "REPRODUCING.md").write_text(
        "[README](models/fake/README.md)\n", encoding="utf-8"
    )
    (member / "TRAINING.md").write_text("# Training\n", encoding="utf-8")

    generator.generate(tmp_path)
    publisher.publish(tmp_path)

    readme = (tmp_path / "docs/models/fake/index.md").read_text(encoding="utf-8")
    reproducing = (tmp_path / "docs/models/fake/reproducing.md").read_text(
        encoding="utf-8"
    )
    assert "](training/)" in readme
    assert "](reproducing/#quick)" in readme
    assert "](../../training-reproduction/)" in readme
    assert "](./)" in reproducing
    assert (tmp_path / "docs/models/fake/training.md").is_file()
    assert (tmp_path / "docs/api/models/fake/training/index.md").is_file()
    summary = (tmp_path / "docs/models/SUMMARY.md").read_text(encoding="utf-8")
    assert "* [fake](fake/index.md)" in summary
    assert "  * [Training](fake/training.md)" in summary


def test_generator_writes_only_to_generated_doc_roots(tmp_path: Path) -> None:
    generator = _load_generator()
    publisher = _load_guide_publisher()
    _write_workspace(tmp_path)
    package = _write_member(tmp_path, "models", "fake", "fake")
    (package.parents[1] / "README.md").write_text("# Fake\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("hand-written\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text("hand-written config\n", encoding="utf-8")
    generated_roots = {Path("docs/api"), Path("docs/models"), Path("docs/libraries")}
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and not any(
            root in path.relative_to(tmp_path).parents for root in generated_roots
        )
    }

    generator.generate(tmp_path)
    publisher.publish(tmp_path)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and not any(
            root in path.relative_to(tmp_path).parents for root in generated_roots
        )
    }
    assert after == before
    assert (tmp_path / "docs/api/SUMMARY.md").is_file()
    assert (tmp_path / "docs/models/SUMMARY.md").is_file()


def test_docs_markdown_pages_have_icon_and_tags_frontmatter() -> None:
    generator = _load_generator()
    publisher = _load_guide_publisher()
    generator.generate(REPO_ROOT)
    publisher.publish(REPO_ROOT)

    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(REPO_ROOT / "docs")
        if {"stylesheets", "plans"} & set(
            relative.parts
        ) or relative.name == "SUMMARY.md":
            continue
        _assert_frontmatter(path)
