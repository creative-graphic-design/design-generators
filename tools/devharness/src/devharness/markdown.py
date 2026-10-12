"""Shared Markdown parsing primitives for training-document checkers."""

from __future__ import annotations

import re
from collections.abc import Iterable

FENCE_START_RE = re.compile(r"^\s*(```|~~~)")


def split_markdown_row(line: str) -> list[str]:
    """Split a simple Markdown table row into stripped cells."""
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line.strip().strip("|"):
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if escaped:
        current.append("\\")

    cells.append("".join(current).strip())
    return cells


def is_table_delimiter(line: str) -> bool:
    """Return whether a Markdown table row is a delimiter row."""
    cells = split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def iter_unfenced_lines(text: str) -> Iterable[str]:
    """Yield Markdown lines outside fenced code blocks."""
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        match = FENCE_START_RE.match(line)

        if match:
            marker = match.group(1)

            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""

            continue

        if not in_fence:
            yield line


def iter_heading_sections_with_level(
    text: str,
) -> Iterable[tuple[int, str, list[str]]]:
    """Yield Markdown heading level, text, and section lines outside fences."""
    current_heading: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []

    for line in iter_unfenced_lines(text):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current_heading is not None:
                assert current_level is not None
                yield current_level, current_heading, current_lines
                current_lines = []

            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            continue

        if current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        assert current_level is not None
        yield current_level, current_heading, current_lines


def iter_heading_sections(text: str) -> Iterable[tuple[str, list[str]]]:
    """Yield Markdown heading text with the lines inside that heading."""
    for _, current_heading, current_lines in iter_heading_sections_with_level(text):
        yield current_heading, current_lines
