"""Check semantic blank-line conventions in package and script source.

The consecutive-statement check is intentionally a crude heuristic: more than
15 consecutive non-blank statement lines inside one function body may indicate
that semantic units need separation, but the checker does not judge semantics.
The raise-block check is exact: after a ``raise`` statement, ordinary code on a
later line requires a blank line unless that code is a block-continuation
keyword (``except``, ``elif``, ``else``, ``finally``, or an AST-confirmed
``match`` ``case`` clause).
The block-end check is also exact: after an ``if``, ``for``, ``while``,
``try``, or ``with`` suite, ordinary code at the compound statement's
enclosing indentation requires one blank line. Continuation clauses and the
end of a suite or file are exempt.

The baseline is a shrink-only ratchet keyed by ``(file, rule)``. Each
tab-separated entry stores a violation count rather than a line number, so
source edits that move a violation do not churn the baseline. This deliberately
accepts a same-file swap (one residual violation fixed and one added) when the
file/rule count is unchanged; that is the granularity of this heuristic.

The checker fails when the working baseline counts differ from the current scan.
Once a count-format baseline exists in committed ``HEAD``, it also fails if a
working count is added or increased relative to that commit. This permits only
decreasing or removing residual counts and prevents hiding a new violation by
appending it to the baseline. A missing or legacy location-format baseline in
``HEAD`` is treated as an initial baseline, which permits the merge that first
introduces this checker to establish counts for content inherited from main.
An ``# Established rule: <rule>`` marker records that a rule has been
introduced even after its residual count reaches zero. Established markers are
also shrink-only: removing one relative to committed ``HEAD`` fails.
When a new rule is introduced, its first count-format baseline is also allowed
through the ``INTRODUCED_RULES`` exception below; later changes to that rule
are ratcheted against its committed counts like every existing rule.
"""

from __future__ import annotations

import argparse
import ast
import io
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "semantic_blank_lines_baseline.txt"
BASELINE_RELATIVE_PATH = Path("scripts/semantic_blank_lines_baseline.txt")
MAX_CONSECUTIVE_STATEMENTS = 15
CONTINUATION_KEYWORDS = frozenset({"except", "elif", "else", "finally"})
ESTABLISHED_RULE_PREFIX = "# Established rule:"
INTRODUCED_RULES = frozenset({"block-end"})
COMPOUND_STATEMENT_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
)
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


def source_files(root: Path) -> list[Path]:
    """Return Python files covered by this check."""
    files: set[Path] = set()
    for pattern in SCAN_GLOBS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def _collect_statement_lines(node: ast.AST, lines: set[int]) -> None:
    """Collect statement-start lines while treating nested functions as leaves."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
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


def _match_case_lines(
    tree: ast.Module, first_tokens: dict[int, tokenize.TokenInfo]
) -> set[int]:
    """Return lines whose ``case`` token starts an AST ``match`` clause."""
    case_lines: set[int] = set()
    for match in ast.walk(tree):
        if not isinstance(match, ast.Match):
            continue
        match_end = match.end_lineno or match.lineno
        candidates = sorted(
            line
            for line, token in first_tokens.items()
            if token.string == "case"
            and match.lineno < line <= match_end
            and token.start[1] > match.col_offset
        )
        for match_case in match.cases:
            pattern_line = getattr(match_case.pattern, "lineno", None)
            if pattern_line is None:
                continue
            preceding = [line for line in candidates if line <= pattern_line]
            if preceding:
                case_lines.add(max(preceding))
    return case_lines


def _next_code_line(
    first_tokens: dict[int, tokenize.TokenInfo], end_line: int, line_count: int
) -> int | None:
    """Return the next physical line containing a non-layout token."""
    return next(
        (
            line_number
            for line_number in range(end_line + 1, line_count + 1)
            if line_number in first_tokens
        ),
        None,
    )


def _is_continuation(
    token: tokenize.TokenInfo, line_number: int, match_case_lines: set[int]
) -> bool:
    """Return whether a token starts an exempt compound-statement continuation."""
    return token.string in CONTINUATION_KEYWORDS or (
        token.string == "case" and line_number in match_case_lines
    )


def _has_blank_line_after(lines: list[str], end_line: int) -> bool:
    """Return whether the physical line immediately after ``end_line`` is blank."""
    return end_line < len(lines) and not lines[end_line].strip()


def _raise_violations(
    tree: ast.Module, source: str, relative_path: str
) -> Iterator[Violation]:
    """Yield exact missing blank lines after raises before ordinary code."""
    lines = source.splitlines()
    first_tokens = _first_code_tokens(source)
    match_case_lines = _match_case_lines(tree, first_tokens)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue

        end_line = node.end_lineno or node.lineno
        next_line = _next_code_line(first_tokens, end_line, len(lines))

        if next_line is None:
            continue

        next_token = first_tokens[next_line]
        if _is_continuation(next_token, next_line, match_case_lines):
            continue

        if _has_blank_line_after(lines, end_line):
            continue

        yield Violation(
            relative_path,
            end_line,
            "raise-block",
            f"ordinary code follows raise on line {next_line} without a blank line",
        )


def _block_end_violations(
    tree: ast.Module, source: str, relative_path: str
) -> Iterator[Violation]:
    """Yield missing blank lines after compound suites at their enclosing indent."""
    lines = source.splitlines()
    first_tokens = _first_code_tokens(source)
    match_case_lines = _match_case_lines(tree, first_tokens)
    seen_boundaries: set[tuple[int, int]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, COMPOUND_STATEMENT_TYPES):
            continue

        end_line = node.end_lineno or node.lineno
        next_line = _next_code_line(first_tokens, end_line, len(lines))

        if next_line is None:
            continue

        next_token = first_tokens[next_line]
        if _is_continuation(next_token, next_line, match_case_lines):
            continue

        if next_token.start[1] != node.col_offset:
            continue

        if _has_blank_line_after(lines, end_line):
            continue

        boundary = (end_line, node.col_offset)
        if boundary in seen_boundaries:
            continue

        seen_boundaries.add(boundary)
        yield Violation(
            relative_path,
            end_line,
            "block-end",
            f"ordinary code follows {type(node).__name__} suite on line "
            f"{next_line} at the enclosing indentation without a blank line",
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
    yield from _block_end_violations(tree, source, relative_path)


def current_counts(root: Path) -> dict[tuple[str, str], int]:
    """Return current violation counts keyed by file and rule."""
    return dict(
        Counter(
            (violation.path, violation.rule)
            for path in source_files(root)
            for violation in violations_for_file(root, path)
        )
    )


def _parse_baseline_text(
    text: str,
) -> tuple[dict[tuple[str, str], int], set[str]]:
    """Parse count-format entries and durable established-rule markers."""
    counts: dict[tuple[str, str], int] = {}
    established_rules: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith(ESTABLISHED_RULE_PREFIX):
            rule = line.removeprefix(ESTABLISHED_RULE_PREFIX).strip()
            if not rule:
                raise ValueError(
                    f"invalid established semantic blank-line rule: {line}"
                )

            if rule in established_rules:
                raise ValueError(
                    f"duplicate established semantic blank-line rule: {line}"
                )

            established_rules.add(rule)
            continue

        if line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"invalid semantic blank-line baseline entry: {line}")

        path, rule, count_text = fields
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"invalid semantic blank-line count: {line}") from exc

        if count < 0:
            raise ValueError(f"negative semantic blank-line count: {line}")

        key = (path, rule)
        if key in counts:
            raise ValueError(f"duplicate semantic blank-line baseline key: {line}")

        counts[key] = count
    return counts, established_rules


def _baseline_state(path: Path) -> tuple[dict[tuple[str, str], int], set[str]]:
    """Read count entries and established-rule markers from a baseline path."""
    if not path.exists():
        return {}, set()

    return _parse_baseline_text(path.read_text(encoding="utf-8"))


def baseline_counts(path: Path) -> dict[tuple[str, str], int]:
    """Read file/rule violation counts from a shrink-only baseline file."""
    counts, _ = _baseline_state(path)
    return counts


def baseline_established_rules(path: Path) -> set[str]:
    """Read durable established-rule markers from a baseline file."""
    _, established_rules = _baseline_state(path)
    return established_rules


def committed_baseline_state(
    root: Path,
) -> tuple[dict[tuple[str, str], int], set[str]] | None:
    """Return committed counts and markers, or ``None`` before count format."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"HEAD:{BASELINE_RELATIVE_PATH.as_posix()}",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    non_comment_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if non_comment_lines and all(
        len(line.split("\t")) == 4 for line in non_comment_lines
    ):
        return None

    return _parse_baseline_text(result.stdout)


def committed_baseline_counts(root: Path) -> dict[tuple[str, str], int] | None:
    """Return committed counts, or ``None`` before the count-format baseline."""
    state = committed_baseline_state(root)
    return state[0] if state is not None else None


def check_semantic_blank_lines(root: Path, baseline: Path) -> int:
    """Enforce an exact, shrink-only match between scan and baseline."""
    current = current_counts(root)
    baseline_current, established_rules = _baseline_state(baseline)
    mismatches = sorted(
        (
            path,
            rule,
            baseline_current.get((path, rule), 0),
            current.get((path, rule), 0),
        )
        for path, rule in baseline_current.keys() | current.keys()
        if baseline_current.get((path, rule), 0) != current.get((path, rule), 0)
    )

    committed_state = committed_baseline_state(root)
    if committed_state is None:
        committed = None
        committed_rules: set[str] = set()
        removed_established_rules: list[str] = []
    else:
        committed, committed_established_rules = committed_state
        committed_rules = {rule for _, rule in committed} | committed_established_rules
        removed_established_rules = sorted(
            committed_established_rules - established_rules
        )

    increased_counts = (
        sorted(
            (
                path,
                rule,
                committed.get((path, rule), 0),
                count,
            )
            for (path, rule), count in baseline_current.items()
            if count > committed.get((path, rule), 0)
            and not (rule in INTRODUCED_RULES and rule not in committed_rules)
        )
        if committed is not None
        else []
    )
    if not mismatches and not increased_counts and not removed_established_rules:
        return 0

    print("Semantic blank-line baseline is not shrink-only:", file=sys.stderr)
    if mismatches:
        print("Baseline counts differ from current scan:", file=sys.stderr)
        for path, rule, baseline_count, current_count in mismatches:
            print(
                f"{path}\t{rule}\tbaseline={baseline_count}\tcurrent={current_count}",
                file=sys.stderr,
            )

    if increased_counts:
        print("Baseline counts added or increased relative to HEAD:", file=sys.stderr)
        for path, rule, committed_count, current_count in increased_counts:
            print(
                f"{path}\t{rule}\tHEAD={committed_count}\tcurrent={current_count}",
                file=sys.stderr,
            )

    if removed_established_rules:
        print("Established rule markers removed relative to HEAD:", file=sys.stderr)
        for rule in removed_established_rules:
            print(f"# Established rule: {rule}", file=sys.stderr)

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
