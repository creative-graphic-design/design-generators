"""Reject host-specific absolute paths in committed files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SLASH = "/"
BACKSLASH = "\\"
ROOT_HOME_PREFIX = SLASH + "root" + SLASH
LINUX_HOME_PREFIX = SLASH + "home" + SLASH
MACOS_HOME_PREFIX = SLASH + "Users" + SLASH
WINDOWS_HOME_PATTERN = (
    r"[A-Za-z]:" + re.escape(BACKSLASH) + re.escape("Users") + re.escape(BACKSLASH)
)
GHQ_MARKER = SLASH.join(("ghq", "github.com"))
GHQ_ABSOLUTE_PREFIX = r"""(?:^|(?<=[\s'"`]))"""

EXCLUDED_FILES = {"uv.lock"}
EXCLUDED_DIRS = {"vendor"}
PATH_CHARS = r"""[^\s'"`<>]*"""
USERNAME = r"""[^\s/'"`<>:\\]+"""


@dataclass(frozen=True)
class PathViolation:
    """Host-specific path occurrence in a tracked file."""

    path: str
    line: int
    kind: str
    match: str

    def format(self) -> str:
        """Return the stable human-readable report line."""
        return f"{self.path}:{self.line}: {self.kind}: {self.match}"


@dataclass(frozen=True)
class PatternSpec:
    """Compiled path pattern and the capture group to report."""

    kind: str
    pattern: re.Pattern[str]
    match_group: str | int = 0


PATTERNS = (
    PatternSpec(
        kind="root home path",
        pattern=re.compile(re.escape(ROOT_HOME_PREFIX) + PATH_CHARS),
    ),
    PatternSpec(
        kind="Linux user home path",
        pattern=re.compile(
            re.escape(LINUX_HOME_PREFIX) + USERNAME + re.escape(SLASH) + PATH_CHARS
        ),
    ),
    PatternSpec(
        kind="macOS user home path",
        pattern=re.compile(
            re.escape(MACOS_HOME_PREFIX) + USERNAME + re.escape(SLASH) + PATH_CHARS
        ),
    ),
    PatternSpec(
        kind="Windows user home path",
        pattern=re.compile(WINDOWS_HOME_PATTERN + USERNAME + r"(?:\\[^\s'\"`<>]*)?"),
    ),
    PatternSpec(
        kind="absolute ghq checkout path",
        pattern=re.compile(
            GHQ_ABSOLUTE_PREFIX
            + r"(?P<path>"
            + re.escape(SLASH)
            + PATH_CHARS
            + re.escape(GHQ_MARKER)
            + PATH_CHARS
            + r")"
        ),
        match_group="path",
    ),
)


def tracked_paths(root: Path) -> list[PurePosixPath]:
    """Return paths tracked by git under ``root``."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        PurePosixPath(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw
    ]


def is_excluded(path: PurePosixPath) -> bool:
    """Return whether a tracked path is intentionally outside this check."""
    return path.as_posix() in EXCLUDED_FILES or any(
        part in EXCLUDED_DIRS for part in path.parts
    )


def read_text_file(path: Path) -> str | None:
    """Return text file contents, or ``None`` for binary/non-files."""
    if not path.is_file():
        return None
    content = path.read_bytes()
    if b"\0" in content:
        return None
    return content.decode("utf-8", errors="replace")


def find_violations_in_text(rel_path: str, text: str) -> list[PathViolation]:
    """Return host-specific absolute path occurrences in one file."""
    violations: list[PathViolation] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for spec in PATTERNS:
            for match in spec.pattern.finditer(line):
                violations.append(
                    PathViolation(
                        path=rel_path,
                        line=line_number,
                        kind=spec.kind,
                        match=match.group(spec.match_group),
                    )
                )
    return violations


def check_committed_paths(root: Path) -> int:
    """Check tracked text files for host-specific absolute paths."""
    violations: list[PathViolation] = []
    for rel_path in tracked_paths(root):
        if is_excluded(rel_path):
            continue
        text = read_text_file(root / rel_path)
        if text is None:
            continue
        violations.extend(find_violations_in_text(rel_path.as_posix(), text))

    if not violations:
        return 0

    print(
        "Host-specific absolute paths are not allowed in committed files:",
        file=sys.stderr,
    )
    for violation in violations:
        print(f"  {violation.format()}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the committed-path checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to scan",
    )
    args = parser.parse_args(argv)
    return check_committed_paths(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
