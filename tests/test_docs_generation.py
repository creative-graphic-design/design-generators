"""Tests for documentation-page frontmatter."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.S)


def test_docs_markdown_pages_have_icon_and_tags_frontmatter() -> None:
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(REPO_ROOT / "docs")
        if "plans" in relative.parts:
            continue

        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        assert match is not None, f"{path}: missing YAML frontmatter"
        body = match.group("body")
        assert re.search(r"(?m)^icon:\s+lucide/", body), path
        assert re.search(r"(?m)^tags:\n(?:  - .+\n?)+", body), path
