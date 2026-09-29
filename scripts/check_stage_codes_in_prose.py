"""Reject training-reproduction stage codes in source-code prose.

The checker scans comments and module, class, and function docstrings only.
Ordinary string literals, such as JSON data values, skip reasons, and
argparse help, are deliberately outside its scope.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("lib", "models")
STAGE_CODE_RE = re.compile(r"\bS[0-5](?:-S[0-5])?\b")
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def python_targets(root: Path) -> list[Path]:
    """Return Python files covered by the source-code prose check."""
    targets: list[Path] = []
    for top_level in SOURCE_DIRS:
        source_dir = root / top_level
        if not source_dir.is_dir():
            continue

        targets.extend(
            path
            for path in source_dir.rglob("*.py")
            if path.is_file() and "vendor" not in path.relative_to(root).parts
        )

    return sorted(targets)


def _character_column(source_lines: list[str], line: int, byte_column: int) -> int:
    """Convert an AST UTF-8 byte column to a tokenization character column."""
    return len(source_lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))


def _docstring_spans(
    tree: ast.AST, source: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return source spans for the first string statement of each docstring owner."""
    source_lines = source.splitlines(keepends=True)
    spans: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS):
            continue

        body = getattr(node, "body", ())
        if not body:
            continue

        first = body[0]
        if not isinstance(first, ast.Expr):
            continue

        if not isinstance(first.value, ast.Constant) or not isinstance(
            first.value.value, str
        ):
            continue

        if first.end_lineno is None or first.end_col_offset is None:
            continue

        start = (
            first.lineno,
            _character_column(source_lines, first.lineno, first.col_offset),
        )
        end = (
            first.end_lineno,
            _character_column(source_lines, first.end_lineno, first.end_col_offset),
        )
        spans.append((start, end))

    return spans


def _in_span(
    start: tuple[int, int],
    end: tuple[int, int],
    span: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    """Return whether a token range is contained in an AST docstring span."""
    return span[0] <= start and end <= span[1]


def check_file(root: Path, path: Path) -> list[tuple[Path, int, str]]:
    """Return stage-code matches in comments and docstrings for one file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [(path.relative_to(root), exc.lineno or 1, f"syntax error: {exc.msg}")]

    docstring_spans = _docstring_spans(tree, source)
    reports: list[tuple[Path, int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        is_docstring = token.type == tokenize.STRING and any(
            _in_span(token.start, token.end, span) for span in docstring_spans
        )
        if token.type != tokenize.COMMENT and not is_docstring:
            continue

        for match in STAGE_CODE_RE.finditer(token.string):
            line = token.start[0] + token.string[: match.start()].count("\n")
            reports.append((path.relative_to(root), line, match.group()))

    return reports


def check_stage_codes_in_prose(root: Path) -> int:
    """Run the source-code prose stage-code check."""
    reports = [
        report for path in python_targets(root) for report in check_file(root, path)
    ]
    if not reports:
        return 0

    print("Training-reproduction stage codes in source-code prose:", file=sys.stderr)
    for path, line, match in reports:
        print(f"{path}:{line}: {match}", file=sys.stderr)

    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the source-code prose stage-code checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to scan",
    )
    args = parser.parse_args(argv)
    return check_stage_codes_in_prose(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
