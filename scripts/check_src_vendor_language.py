"""Reject new vendor-language references in main package source files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "src_vendor_language_baseline.txt"
SCAN_GLOBS = ("models/*/src/**/*.py", "lib/*/src/**/*.py")
VENDOR_RE = re.compile(r"\bvendor\b", re.IGNORECASE)


def is_excluded(path: Path) -> bool:
    """Return whether a source file owns conversion or vendor-state plumbing."""
    name = path.name
    return name.startswith("conversion") or name.startswith("vendor_state")


def source_files() -> list[Path]:
    """Return package source files covered by this check."""
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(path for path in files if not is_excluded(path))


def current_entries() -> set[str]:
    """Return normalized baseline entries for current source matches."""
    entries: set[str] = set()
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for path in source_files():
        rel_path = path.relative_to(ROOT).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if VENDOR_RE.search(line) is None:
                continue
            normalized = " ".join(line.strip().split())
            key = (rel_path, normalized)
            occurrences[key] += 1
            entries.add(f"{rel_path}\t{occurrences[key]}\t{normalized}")
    return entries


def baseline_entries() -> set[str]:
    """Return committed baseline entries."""
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(BASELINE_PATH)
    return {
        line
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def main() -> int:
    """Compare source matches against the shrink-only baseline."""
    if sys.argv[1:] == ["--write-baseline"]:
        BASELINE_PATH.write_text(
            "\n".join(sorted(current_entries())) + "\n", encoding="utf-8"
        )
        return 0
    current = current_entries()
    baseline = baseline_entries()
    unexpected = sorted(current - baseline)
    stale = sorted(baseline - current)
    if not unexpected and not stale:
        return 0
    if unexpected:
        print("New vendor language in package source:", file=sys.stderr)
        for entry in unexpected:
            print(f"  + {entry}", file=sys.stderr)
    if stale:
        print("Stale vendor language baseline entries:", file=sys.stderr)
        for entry in stale:
            print(f"  - {entry}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
