"""Reject post-load Diffusers scheduler replacement assignments."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Final

from devharness.repo import find_repo_root


HORIZONTAL_WHITESPACE: Final = r"[ \t]*"
SCHEDULER_ASSIGNMENT: Final = re.compile(
    rf"\b[A-Za-z_]\w*(?:{HORIZONTAL_WHITESPACE}\."
    rf"{HORIZONTAL_WHITESPACE}[A-Za-z_]\w*)*{HORIZONTAL_WHITESPACE}\."
    rf"{HORIZONTAL_WHITESPACE}scheduler{HORIZONTAL_WHITESPACE}="
    rf"{HORIZONTAL_WHITESPACE}(?:[A-Za-z_]\w*{HORIZONTAL_WHITESPACE}\."
    rf"{HORIZONTAL_WHITESPACE})+from_(?:config|pretrained)"
    rf"{HORIZONTAL_WHITESPACE}\(",
)


def _in_scope_paths(root: Path) -> list[Path]:
    """Return policy documentation and runtime source files in scope."""
    paths = [
        *root.glob("models/*/README.md"),
        *root.glob("models/*/REPRODUCING.md"),
        *root.glob("docs/*.md"),
    ]
    for source_root in (*root.glob("models/*/src"), *root.glob("lib/*/src")):
        paths.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".pyi"}
            and not ({"tests", "vendor"} & set(path.relative_to(root).parts))
        )

    return sorted(paths)


def find_scheduler_assignment_violations(root: Path) -> list[str]:
    """Return relative path and line entries for forbidden assignments."""
    violations = []
    for path in _in_scope_paths(root):
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        for match in SCHEDULER_ASSIGNMENT.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            violations.append(f"{relative_path}:{line_number}")

    return violations


def check(root: Path) -> int:
    """Run the scheduler loading policy check."""
    violations = find_scheduler_assignment_violations(root)
    if not violations:
        return 0

    print(
        "Pass Diffusers pipeline schedulers to `from_pretrained(..., scheduler=...)`; "
        "do not assign them after loading:",
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
