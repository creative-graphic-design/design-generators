"""Reject unlinked issue and pull-request references in reader-facing docs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RE = re.compile(r"(?<!\w)#\d+\b")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
REFERENCE_DEFINITION_RE = re.compile(r"^\s{0,3}\[([^\]\n]+)\]:\s*(?:<[^>\n]+>|\S+)")
TRAILING_PUNCTUATION = ".!?;:,。！？；：、…"

# Include only exact standalone English intro fragments: a prefix match would
# flag useful prose such as "This document describes the stage protocol".
EMPTY_INTRO_MARKERS_EN = (
    "This document describes",
    "In this section we will",
)
# Keep the Japanese list separate so additions can be reviewed by language;
# the exact-standalone rule excludes informative Japanese sentences too.
EMPTY_INTRO_MARKERS_JA = ("本ドキュメントでは",)

# Keep conclusion markers exact and standalone: a sentence such as "In
# conclusion, the package passes S0" carries information and is excluded.
EMPTY_CONCLUSION_MARKERS_EN = ("In conclusion",)
EMPTY_CONCLUSION_MARKERS_JA = ("結論として", "まとめると")


@dataclass(frozen=True)
class ReferenceViolation:
    """An unlinked issue or pull-request reference in reader-facing prose."""

    path: Path
    line: int
    reference: str
    location: str

    def format(self, root: Path) -> str:
        """Return a stable human-readable violation line."""
        relative_path = self.path.relative_to(root).as_posix()
        return (
            f"{relative_path}:{self.line}: {self.location} contains "
            f"unlinked issue/PR reference {self.reference}"
        )


@dataclass(frozen=True)
class SlopViolation:
    """A deterministic, curated slop marker in reader-facing prose."""

    path: Path
    line: int
    category: str
    marker: str
    detail: str = ""

    def format(self, root: Path) -> str:
        """Return a stable human-readable violation line."""
        relative_path = self.path.relative_to(root).as_posix()
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{relative_path}:{self.line}: {self.category}: {self.marker}{suffix}"


def reader_facing_paths(root: Path = ROOT) -> list[Path]:
    """Return Markdown files covered by the reader-facing documentation rule."""
    paths = sorted((root / "docs").glob("*.md"))
    paths.extend(sorted((root / "models").glob("*/TRAINING.md")))
    return [path for path in paths if path.is_file()]


def _closing_parenthesis(line: str, opening: int) -> int | None:
    """Return the closing parenthesis for a simple Markdown link target."""
    escaped = False
    for index in range(opening + 1, len(line)):
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ")":
            return index
    return None


def _reference_key(value: str) -> str:
    """Normalize a Markdown reference label for definition matching."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def reference_definitions(text: str) -> set[str]:
    """Return defined Markdown reference-link labels outside code fences."""
    definitions: set[str] = set()

    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        fence_match = FENCE_RE.match(line)

        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue

        if match := REFERENCE_DEFINITION_RE.match(line):
            definitions.add(_reference_key(match.group(1)))
    return definitions


def mask_markdown_links(line: str, definitions: set[str] | None = None) -> str:
    """Replace Markdown link and autolink spans with spaces.

    Keeping the line length unchanged lets callers report the original line
    number and reference while allowing issue numbers in link labels or URLs.
    """
    characters = list(line)
    index = 0
    while index < len(line):
        opening = index
        if line[index] == "!" and index + 1 < len(line) and line[index + 1] == "[":
            opening = index + 1

        if line[opening] == "[" and (opening == 0 or line[opening - 1] != "\\"):
            label_end = line.find("]", opening + 1)
            if label_end != -1:
                end = label_end + 1

                valid_link = False
                if end < len(line) and line[end] == "(":
                    target_end = _closing_parenthesis(line, end)
                    if target_end is not None:
                        end = target_end + 1
                        valid_link = True

                elif end < len(line) and line[end] == "[":
                    reference_end = line.find("]", end + 1)
                    reference_label = (
                        line[end + 1 : reference_end] or line[opening + 1 : label_end]
                        if reference_end != -1
                        else ""
                    )
                    if (
                        reference_end != -1
                        and definitions is not None
                        and _reference_key(reference_label) in definitions
                    ):
                        end = reference_end + 1
                        valid_link = True
                elif (
                    definitions is not None
                    and _reference_key(line[opening + 1 : label_end]) in definitions
                ):
                    valid_link = True
                if valid_link:
                    characters[index:end] = [" "] * (end - index)
                    index = end
                    continue
        if line[index] == "<":
            autolink_end = line.find(">", index + 1)
            if autolink_end != -1 and re.match(
                r"(?:https?://|mailto:)", line[index + 1 : autolink_end]
            ):
                characters[index : autolink_end + 1] = [" "] * (
                    autolink_end + 1 - index
                )
                index = autolink_end + 1
                continue
        index += 1
    return "".join(characters)


def _heading_text(line: str) -> str | None:
    """Return heading text without an optional ATX closing marker."""
    match = HEADING_RE.match(line)
    if match is None:
        return None
    return re.sub(r"\s+#+\s*$", "", line[match.end() :].strip())


def _normalized_heading(line: str) -> str | None:
    """Return a case-insensitive heading key, or ``None`` for other lines."""
    heading = _heading_text(line)
    if heading is None:
        return None
    return re.sub(r"\s+", " ", heading).casefold()


def _empty_marker(paragraph: str) -> tuple[str, str] | None:
    """Return the category and marker for an exact empty marker paragraph."""
    normalized = re.sub(r"\s+", " ", paragraph).strip()
    marker = normalized.rstrip(TRAILING_PUNCTUATION).strip()
    english_marker = marker.casefold()

    for candidate in EMPTY_INTRO_MARKERS_EN:
        if english_marker == candidate.casefold():
            return "empty introduction marker", candidate

    for candidate in EMPTY_INTRO_MARKERS_JA:
        if marker == candidate:
            return "empty introduction marker", candidate

    for candidate in EMPTY_CONCLUSION_MARKERS_EN:
        if english_marker == candidate.casefold():
            return "empty conclusion marker", candidate

    for candidate in EMPTY_CONCLUSION_MARKERS_JA:
        if marker == candidate:
            return "empty conclusion marker", candidate

    return None


def _unfenced_lines(text: str) -> list[tuple[int, str | None]]:
    """Return lines outside fences, with ``None`` marking a fence boundary."""
    lines: list[tuple[int, str | None]] = []
    in_fence = False
    fence_marker = ""

    for line_number, line in enumerate(text.splitlines(), start=1):
        fence_match = FENCE_RE.match(line)

        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""

            lines.append((line_number, None))
            continue

        if not in_fence:
            lines.append((line_number, line))
    return lines


def slop_violations_for_document(path: Path) -> list[SlopViolation]:
    """Return curated deterministic slop markers in one document."""
    text = path.read_text(encoding="utf-8")
    definitions = reference_definitions(text)
    lines = _unfenced_lines(text)
    violations: list[SlopViolation] = []
    headings: dict[str, int] = {}
    paragraph_lines: list[tuple[int, str]] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = "\n".join(line for _, line in paragraph_lines)
        if marker := _empty_marker(mask_markdown_links(paragraph, definitions)):
            category, marker_text = marker
            violations.append(
                SlopViolation(path, paragraph_lines[0][0], category, marker_text)
            )
        paragraph_lines.clear()

    for line_number, line in lines:
        if line is None:
            flush_paragraph()
            continue
        heading_match = HEADING_RE.match(line)
        heading_text = _heading_text(line)
        heading = _normalized_heading(line)
        if heading is not None:
            assert heading_match is not None
            assert heading_text is not None
            flush_paragraph()
            if heading in headings:
                violations.append(
                    SlopViolation(
                        path,
                        line_number,
                        "duplicate heading",
                        heading_text,
                        f"first occurrence at line {headings[heading]}",
                    )
                )
            else:
                headings[heading] = line_number
            continue
        if line.strip():
            paragraph_lines.append((line_number, line))
        else:
            flush_paragraph()
    flush_paragraph()
    return violations


def violations_for_document(path: Path) -> list[ReferenceViolation]:
    """Return unlinked issue and pull-request references in one document."""
    text = path.read_text(encoding="utf-8")
    definitions = reference_definitions(text)
    violations: list[ReferenceViolation] = []
    in_fence = False
    fence_marker = ""

    for line_number, line in enumerate(text.splitlines(), start=1):
        fence_match = FENCE_RE.match(line)

        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True

                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        if REFERENCE_DEFINITION_RE.match(line):
            continue

        location = "heading" if HEADING_RE.match(line) else "body"
        masked_line = mask_markdown_links(line, definitions)

        for match in REFERENCE_RE.finditer(masked_line):
            violations.append(
                ReferenceViolation(path, line_number, match.group(), location)
            )
    return violations


def current_violations(root: Path = ROOT) -> list[ReferenceViolation | SlopViolation]:
    """Return reader-facing reference and slop violations under the root."""
    violations: list[ReferenceViolation | SlopViolation] = []
    for path in reader_facing_paths(root):
        violations.extend(violations_for_document(path))
        violations.extend(slop_violations_for_document(path))
    return violations


def check_reader_facing_references(root: Path = ROOT) -> int:
    """Check reader-facing docs for references and curated slop markers."""
    violations = current_violations(root)
    if not violations:
        return 0
    print("Reader-facing documentation violations:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation.format(root)}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the reader-facing reference checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    return check_reader_facing_references(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
