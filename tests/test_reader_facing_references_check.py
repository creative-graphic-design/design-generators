"""Reader-facing Markdown reference checker tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_READER_REFERENCES = REPO_ROOT / "scripts" / "check_reader_facing_references.py"


def load_checker() -> ModuleType:
    """Load the checker script as a module."""
    spec = importlib.util.spec_from_file_location(
        "check_reader_facing_references", CHECK_READER_REFERENCES
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_doc(root: Path, relative_path: str, text: str) -> Path:
    """Write a reader-facing document under a temporary repository root."""
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_rejects_unlinked_references_in_headings_and_body(tmp_path: Path) -> None:
    checker = load_checker()
    path = write_doc(
        tmp_path,
        "docs/example.md",
        "# Results for issue #12\n\nThe pull request #34 changed the protocol.\n",
    )

    violations = checker.current_violations(tmp_path)

    assert [
        (item.path, item.line, item.location, item.reference) for item in violations
    ] == [
        (path, 1, "heading", "#12"),
        (path, 3, "body", "#34"),
    ]
    assert checker.check_reader_facing_references(tmp_path) == 1


def test_accepts_contextual_markdown_links_and_autolinks(tmp_path: Path) -> None:
    checker = load_checker()
    write_doc(
        tmp_path,
        "docs/example.md",
        "See [issue #12](https://github.com/creative-graphic-design/design-generators/issues/148).\n"
        "See <https://github.com/creative-graphic-design/design-generators/issues/148#issue-148>.\n"
        "See [issue #56][issue-ref].\n\n"
        "See [issue #78].\n\n"
        "[issue-ref]: /issues/56\n"
        "[issue #78]: /issues/78\n",
    )

    assert checker.current_violations(tmp_path) == []
    assert checker.check_reader_facing_references(tmp_path) == 0


def test_rejects_escaped_and_undefined_reference_links(tmp_path: Path) -> None:
    checker = load_checker()
    write_doc(
        tmp_path,
        "docs/example.md",
        r"\[issue #12](#issue-12)" + "\n[issue #34][missing]\n",
    )

    violations = checker.current_violations(tmp_path)

    assert [(item.line, item.reference) for item in violations] == [
        (1, "#12"),
        (2, "#34"),
    ]


def test_checks_inline_code_but_ignores_fenced_examples(tmp_path: Path) -> None:
    checker = load_checker()
    path = write_doc(
        tmp_path,
        "models/example/TRAINING.md",
        "Use `#12` or \\#13 only as a linked reference.\n\n"
        "```text\n#34 is an example value.\n```\n",
    )

    violations = checker.current_violations(tmp_path)

    assert len(violations) == 2
    assert violations[0].path == path
    assert violations[0].line == 1
    assert violations[0].location == "body"
    assert violations[0].reference == "#12"
    assert violations[1].reference == "#13"


def test_rejects_exact_english_and_japanese_empty_markers(tmp_path: Path) -> None:
    checker = load_checker()
    path = write_doc(
        tmp_path,
        "docs/example.md",
        "This document describes.\n\n"
        "In this section we will\n\n"
        "本ドキュメントでは。\n\n"
        "In conclusion.\n\n"
        "結論として\n\n"
        "This document describes the protocol.\n\n"
        "In conclusion, the protocol passes.\n",
    )

    violations = checker.slop_violations_for_document(path)

    assert [(item.line, item.category, item.marker) for item in violations] == [
        (1, "empty introduction marker", "This document describes"),
        (3, "empty introduction marker", "In this section we will"),
        (5, "empty introduction marker", "本ドキュメントでは"),
        (7, "empty conclusion marker", "In conclusion"),
        (9, "empty conclusion marker", "結論として"),
    ]


def test_rejects_duplicate_headings_but_allows_unique_headings(tmp_path: Path) -> None:
    checker = load_checker()
    path = write_doc(
        tmp_path,
        "docs/example.md",
        "# Guide\n\n## Results\n\n### Results ##\n\n## Other\n",
    )

    violations = checker.slop_violations_for_document(path)

    assert len(violations) == 1
    assert violations[0].path == path
    assert violations[0].line == 5
    assert violations[0].category == "duplicate heading"
    assert violations[0].marker == "Results"
    assert violations[0].detail == "first occurrence at line 3"


def test_ignores_slop_looking_text_in_fenced_examples(tmp_path: Path) -> None:
    checker = load_checker()
    path = write_doc(
        tmp_path,
        "docs/example.md",
        "```text\nThis document describes\n\n## Results\n\nIn conclusion\n```\n",
    )

    assert checker.slop_violations_for_document(path) == []


def test_current_reader_facing_documents_pass() -> None:
    checker = load_checker()

    assert checker.check_reader_facing_references(REPO_ROOT) == 0
