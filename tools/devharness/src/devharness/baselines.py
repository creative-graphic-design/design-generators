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


def _leading_comment_header(path: Path) -> bytes:
    """Return the existing contiguous leading comment lines unchanged."""
    if not path.exists():
        return b""

    header = bytearray()
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.startswith(b"#"):
            break

        header.extend(line)

    return bytes(header)


def write_entry_baseline(path: Path, entries: Iterable[str]) -> None:
    """Preserve a leading comment header and write sorted entries."""
    header = _leading_comment_header(path)
    content = "\n".join(sorted(entries))
    if not content:
        path.write_bytes(header)
        return

    if header and not header.endswith((b"\r", b"\n")):
        header += b"\n"

    path.write_bytes(header + f"{content}\n".encode())


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
