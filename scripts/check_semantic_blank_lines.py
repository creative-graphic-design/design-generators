"""Check semantic blank-line conventions in package and script source.

The consecutive-statement check is intentionally a crude heuristic: more than
15 consecutive non-blank statement lines inside one function body may indicate
that semantic units need separation, but the checker does not judge semantics.
The raise-block check is exact: after a ``raise`` statement, ordinary code on a
later line requires a blank line unless that code is a block-continuation
keyword (``except``, ``elif``, ``else``, ``finally``, or ``case``).
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "semantic_blank_lines_baseline.txt"
MAX_CONSECUTIVE_STATEMENTS = 15
CONTINUATION_KEYWORDS = frozenset({"except", "elif", "else", "finally", "case"})
SCAN_GLOBS = (
    "models/*/src/**/*.py",
    "lib/*/src/**/*.py",
    "scripts/**/*.py",
)


@dataclass(frozen=True)
class Violation:
    """A source location reported by one of the checker rules."""

    path: str
    line: int
    rule: str
    detail: str

    def as_baseline_entry(self) -> str:
        """Return the stable tab-separated baseline representation."""
        return f"{self.path}\t{self.line}\t{self.rule}\t{self.detail}"


def source_files(root: Path) -> list[Path]:
    """Return Python files covered by this check."""
    files: set[Path] = set()
    for pattern in SCAN_GLOBS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def _collect_statement_lines(node: ast.AST, lines: set[int]) -> None:
    """Collect statement-start lines while treating nested functions as leaves."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        lines.add(node.lineno)
        return
    if isinstance(node, ast.stmt):
        lines.add(node.lineno)
    for child in ast.iter_child_nodes(node):
        _collect_statement_lines(child, lines)


def _function_statement_lines(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[int]:
    """Return sorted statement-start lines in a function body."""
    lines: set[int] = set()
    for statement in function.body:
        _collect_statement_lines(statement, lines)
    return sorted(lines)


def _consecutive_runs(lines: Iterable[int]) -> Iterator[list[int]]:
    """Yield runs of statement lines with no physical line gaps."""
    run: list[int] = []
    for line in lines:
        if run and line != run[-1] + 1:
            yield run
            run = []
        run.append(line)
    if run:
        yield run


def _heuristic_violations(tree: ast.Module, relative_path: str) -> Iterator[Violation]:
    """Yield long consecutive statement runs inside function bodies."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for run in _consecutive_runs(_function_statement_lines(node)):
            if len(run) > MAX_CONSECUTIVE_STATEMENTS:
                yield Violation(
                    relative_path,
                    run[0],
                    "heuristic",
                    f"{node.name}: {len(run)} consecutive statement lines",
                )


def _first_code_tokens(source: str) -> dict[int, tokenize.TokenInfo]:
    """Return the first non-layout token on each physical source line."""
    first_tokens: dict[int, tokenize.TokenInfo] = {}
    ignored = {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type in ignored:
                continue
            first_tokens.setdefault(token.start[0], token)
    except (IndentationError, tokenize.TokenError):
        # AST parsing already reports malformed Python. This fallback keeps the
        # exact rule useful for a valid AST even if tokenization is incomplete.
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line.strip() and not line.lstrip().startswith("#"):
                first_tokens.setdefault(
                    line_number,
                    tokenize.TokenInfo(
                        tokenize.NAME,
                        line.strip().split(maxsplit=1)[0],
                        (line_number, 0),
                        (line_number, len(line)),
                        line,
                    ),
                )
    return first_tokens


def _raise_violations(
    tree: ast.Module, source: str, relative_path: str
) -> Iterator[Violation]:
    """Yield exact missing blank lines after raises before ordinary code."""
    lines = source.splitlines()
    first_tokens = _first_code_tokens(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        end_line = node.end_lineno or node.lineno
        next_line = next(
            (
                line_number
                for line_number in range(end_line + 1, len(lines) + 1)
                if line_number in first_tokens
            ),
            None,
        )
        if next_line is None:
            continue
        if first_tokens[next_line].string in CONTINUATION_KEYWORDS:
            continue
        if end_line < len(lines) and not lines[end_line].strip():
            continue
        yield Violation(
            relative_path,
            end_line,
            "raise-block",
            f"ordinary code follows raise on line {next_line} without a blank line",
        )


def violations_for_file(root: Path, path: Path) -> Iterator[Violation]:
    """Yield checker violations for one source file."""
    relative_path = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        yield Violation(
            relative_path,
            exc.lineno or 1,
            "syntax-error",
            f"{exc.msg}",
        )
        return
    yield from _heuristic_violations(tree, relative_path)
    yield from _raise_violations(tree, source, relative_path)


def current_entries(root: Path) -> set[str]:
    """Return all current violations as baseline-compatible strings."""
    return {
        violation.as_baseline_entry()
        for path in source_files(root)
        for violation in violations_for_file(root, path)
    }


def baseline_entries(path: Path) -> set[str]:
    """Read non-comment entries from a shrink-only baseline file."""
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def check_semantic_blank_lines(root: Path, baseline: Path) -> int:
    """Report current violations that are not present in the baseline."""
    new_entries = sorted(current_entries(root) - baseline_entries(baseline))
    if not new_entries:
        return 0
    print("New semantic blank-line violations:", file=sys.stderr)
    for entry in new_entries:
        print(entry, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the semantic blank-line checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root to scan",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="shrink-only baseline file",
    )
    args = parser.parse_args(argv)
    return check_semantic_blank_lines(args.root, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
