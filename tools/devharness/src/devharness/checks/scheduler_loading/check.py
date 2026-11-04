"""Reject post-load Diffusers scheduler replacement assignments."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Final

from devharness.repo import find_repo_root


PYTHON_FENCE_START: Final = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>`{3,}|~{3,})"
    r"[ \t]*(?P<info>[^\r\n]*?)[ \t]*(?:\r?\n|$)"
)


@dataclass(frozen=True)
class _PythonCodeFence:
    """A Python Markdown fence, or the reason its closing fence is missing."""

    opening_line: int
    code: str | None
    error: str | None


def _in_scope_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """Return in-scope Markdown and Python source files."""
    markdown_paths = [
        *root.glob("models/*/README.md"),
        *root.glob("models/*/REPRODUCING.md"),
        *root.glob("docs/*.md"),
    ]
    source_paths = [
        path
        for source_root in (*root.glob("models/*/src"), *root.glob("lib/*/src"))
        for path in source_root.rglob("*.py")
        if path.is_file()
        and not ({"tests", "vendor"} & set(path.relative_to(root).parts))
    ]

    return sorted(markdown_paths), sorted(source_paths)


def _python_code_fences(text: str) -> list[_PythonCodeFence]:
    """Extract Python fences and retain unterminated fences as parse errors."""
    lines = text.splitlines(keepends=True)
    fences: list[_PythonCodeFence] = []
    index = 0
    while index < len(lines):
        opening = PYTHON_FENCE_START.match(lines[index])
        if opening is None:
            index += 1
            continue

        info = opening.group("info").strip(" \t")
        language = re.split(r"[ \t]+", info, maxsplit=1)[0].lower() if info else ""
        if language not in {"python", "py"}:
            closing = _closing_fence_pattern(opening.group("marker"))
            index = _after_closing_fence(lines, index + 1, closing)
            continue

        opening_line = index + 1
        closing = _closing_fence_pattern(opening.group("marker"))
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and closing.match(lines[cursor]) is None:
            line = lines[cursor]
            indent = len(opening.group("indent"))
            if indent and line.startswith(" " * indent):
                line = line[indent:]
            body.append(line)
            cursor += 1

        if cursor == len(lines):
            fences.append(
                _PythonCodeFence(
                    opening_line=opening_line,
                    code=None,
                    error="unterminated Python code fence",
                )
            )
            break

        fences.append(
            _PythonCodeFence(
                opening_line=opening_line,
                code="".join(body),
                error=None,
            )
        )
        index = cursor + 1

    return fences


def _closing_fence_pattern(marker: str) -> re.Pattern[str]:
    """Match a Markdown closing fence using the opener's character and length."""
    return re.compile(
        rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\r?\n|$)"
    )


def _after_closing_fence(lines: list[str], start: int, closing: re.Pattern[str]) -> int:
    """Skip a non-Python fence and return the first line after its closer."""
    for index in range(start, len(lines)):
        if closing.match(lines[index]):
            return index + 1

    return len(lines)


def _target_has_scheduler_attribute(target: ast.expr) -> bool:
    """Return whether an assignment target contains a `.scheduler` attribute."""
    if isinstance(target, ast.Attribute):
        return target.attr == "scheduler"

    if isinstance(target, ast.Starred):
        return _target_has_scheduler_attribute(target.value)

    if isinstance(target, (ast.List, ast.Tuple)):
        return any(_target_has_scheduler_attribute(item) for item in target.elts)

    return False


def _is_scheduler_factory_call(value: ast.expr | None) -> bool:
    """Return whether an expression calls a supported scheduler factory."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in {"from_config", "from_pretrained"}
    )


def _assignment_violations(
    tree: ast.Module, relative_path: str, line_offset: int = 0
) -> list[str]:
    """Return forbidden scheduler assignments from a parsed Python module."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
            value = node.value
        else:
            continue

        if not _is_scheduler_factory_call(value):
            continue
        if not any(_target_has_scheduler_attribute(target) for target in targets):
            continue

        violations.append(f"{relative_path}:{node.lineno + line_offset}")

    return violations


def _parse_error(path: str, line: int, context: str, message: str) -> str:
    """Format a syntax error from an in-scope Python file or code fence."""
    return f"{path}:{line}: cannot parse {context}: {message}"


def find_scheduler_assignment_violations(root: Path) -> list[str]:
    """Return scheduler assignments and parse failures in the policy scope."""
    violations: list[str] = []
    markdown_paths, source_paths = _in_scope_paths(root)

    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(text, filename=relative_path)
        except SyntaxError as exc:
            violations.append(
                _parse_error(
                    relative_path,
                    exc.lineno or 1,
                    "Python source",
                    exc.msg,
                )
            )
            continue

        violations.extend(_assignment_violations(tree, relative_path))

    for path in markdown_paths:
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        for fence in _python_code_fences(text):
            if fence.error is not None:
                violations.append(
                    _parse_error(
                        relative_path,
                        fence.opening_line,
                        "Python code fence",
                        fence.error,
                    )
                )
                continue

            try:
                tree = ast.parse(
                    fence.code or "",
                    filename=f"{relative_path}:{fence.opening_line}",
                )
            except SyntaxError as exc:
                violations.append(
                    _parse_error(
                        relative_path,
                        fence.opening_line + (exc.lineno or 1),
                        "Python code fence",
                        exc.msg,
                    )
                )
                continue

            violations.extend(
                _assignment_violations(tree, relative_path, fence.opening_line)
            )

    return sorted(violations)


def check(root: Path) -> int:
    """Run the scheduler loading policy check."""
    violations = find_scheduler_assignment_violations(root)
    if not violations:
        return 0

    print(
        "Pass Diffusers pipeline schedulers to `from_pretrained(..., scheduler=...)`; "
        "do not assign them after loading, and keep scanned Python syntax valid:",
        file=sys.stderr,
    )
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)

    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the scheduler loading checker from the current repository."""
    parser = argparse.ArgumentParser(prog="devharness check scheduler-loading")
    parser.parse_args([] if argv is None else argv)
    try:
        root = find_repo_root(Path.cwd())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    return check(root)
