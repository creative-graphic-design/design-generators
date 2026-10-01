"""Shared set-based baseline primitives for repository checkers."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


def read_entry_baseline(path: Path) -> set[str]:
    """Read non-empty, non-comment entries from a UTF-8 baseline file."""
    return {
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def write_entry_baseline(path: Path, entries: Iterable[str]) -> None:
    """Write sorted entries with one trailing newline when non-empty."""
    content = "\n".join(sorted(entries))
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def diff_entry_baseline(
    current: set[str], baseline: set[str]
) -> tuple[list[str], list[str]]:
    """Return sorted unexpected and stale baseline entries."""
    return sorted(current - baseline), sorted(baseline - current)


def print_entries(header: str, marker: str, entries: list[str]) -> None:
    """Print formatted baseline entries to stderr."""
    if not entries:
        return

    print(header, file=sys.stderr)
    for entry in entries:
        print(f"  {marker} {entry}", file=sys.stderr)
