"""Report advisory PR risk without granting merge or publication approval."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum, auto
from pathlib import PurePosixPath
from typing import Final

SHA_RE: Final = re.compile(r"[0-9a-fA-F]{40,64}\Z")
SENSITIVE_PREFIXES: Final = (
    ".agents/",
    ".claude/",
    ".github/",
    "lib/",
    "models/",
    "tools/",
    "vendor/",
)
SENSITIVE_FILES: Final = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "codecov.yml",
        "mkdocs.yml",
        "docs/architecture.md",
        "docs/conventions.md",
        "docs/data-sources.md",
        "docs/implementation-checklist.md",
        "docs/roadmap.md",
        "docs/training-reproduction.md",
    }
)


class Lane(StrEnum):
    """Advisory classification, never an authorization to merge."""

    auto_candidate = auto()
    review = auto()
    block = auto()


@dataclass(frozen=True)
class RiskReport:
    """Explain a preliminary lane tied to the exact compared Git revisions."""

    base_sha: str
    head_sha: str
    lane: Lane
    paths: tuple[str, ...]
    reasons: tuple[str, ...]
    merge_authorized: bool = False


def read_changed_paths(data: bytes) -> tuple[str, ...]:
    """Read Git's NUL-delimited name-status format, including both rename paths."""
    if not data:
        return ()

    fields = data.split(b"\0")
    if fields[-1] != b"":
        raise ValueError("git diff output is not NUL terminated")

    fields.pop()
    paths: set[str] = set()
    position = 0
    while position < len(fields):
        try:
            status = fields[position].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid git status encoding") from exc

        position += 1
        if not status or status[0] not in "ACDMRTUXB":
            raise ValueError(f"unsupported git change status: {status!r}")

        count = 2 if status[0] in "RC" else 1
        if position + count > len(fields):
            raise ValueError("incomplete git change record")

        for field in fields[position : position + count]:
            path = field.decode("utf-8", errors="surrogateescape")
            if not valid_path(path):
                raise ValueError("git diff contains an unsafe or ambiguous path")

            paths.add(path)

        position += count

    return tuple(sorted(paths))


def valid_path(path: str) -> bool:
    """Reject paths that cannot be analyzed as unambiguous repository paths."""
    if not path or path.startswith("/") or "\\" in path:
        return False

    if any(part in ("", ".", "..") for part in path.split("/")):
        return False

    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        return False

    return not PurePosixPath(path).is_absolute()


def changed_paths(base_sha: str, head_sha: str) -> tuple[str, ...]:
    """Compare the merge base with the exact head without executing changed code."""
    if not SHA_RE.fullmatch(base_sha) or not SHA_RE.fullmatch(head_sha):
        raise ValueError("--base and --head must be full Git commit SHAs")

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "-z",
            f"{base_sha}...{head_sha}",
            "--",
        ],
        capture_output=True,
        check=True,
    )
    return read_changed_paths(result.stdout)


def classify(base_sha: str, head_sha: str, paths: tuple[str, ...]) -> RiskReport:
    """Report a candidate for further verification, never a merge decision."""
    if not paths:
        return RiskReport(
            base_sha, head_sha, Lane.block, paths, ("No changes to assess.",)
        )

    if any(not valid_path(path) for path in paths):
        return RiskReport(
            base_sha, head_sha, Lane.block, paths, ("Ambiguous changed path.",)
        )

    reasons: list[str] = []
    if any(
        path in SENSITIVE_FILES or path.startswith(SENSITIVE_PREFIXES) for path in paths
    ):
        reasons.append(
            "Policy, runtime, model, training, or publication scope requires review."
        )

    if any(path.startswith("scripts/") or path.startswith("tests/") for path in paths):
        reasons.append("Checker or test changes cannot authorize their own validation.")

    if any(path.startswith("docs/templates/") for path in paths):
        reasons.append("Template changes affect future generated artifacts.")

    if any(not path.endswith(".md") for path in paths):
        reasons.append("Non-Markdown changes need independent behavioral verification.")

    if any(not path.startswith("docs/") for path in paths):
        reasons.append("Changes outside ordinary docs need explicit review.")

    if reasons:
        return RiskReport(base_sha, head_sha, Lane.review, paths, tuple(reasons))

    return RiskReport(
        base_sha,
        head_sha,
        Lane.auto_candidate,
        paths,
        (
            "Docs-only candidate; still requires trusted evidence, current-head CI, "
            "independent review, and publication safeguards before any AUTO decision.",
        ),
    )


def render_summary(report: RiskReport) -> str:
    """Create a Markdown step summary without interpolating untrusted filenames."""
    reasons = "\n".join(f"- {reason}" for reason in report.reasons)
    return (
        "### PR risk report (advisory only)\n\n"
        f"Lane: **{report.lane.value}**; merge authorized: **no**.\n\n"
        f"Base: {report.base_sha}; HEAD: {report.head_sha}; "
        f"changed paths: {len(report.paths)}.\n\n{reasons}\n\n"
        "This report is not a required gate and must never be used as merge or Hub "
        "publication approval. See issue #543 for the proposed trusted gate.\n"
    )


def main() -> int:
    """Print a JSON report and optionally append an advisory Actions summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--summary", help="GitHub Actions step-summary file")
    args = parser.parse_args()

    try:
        paths = changed_paths(args.base, args.head)
        report = classify(args.base, args.head, paths)
    except (ValueError, subprocess.CalledProcessError) as exc:
        report = RiskReport(
            args.base,
            args.head,
            Lane.block,
            (),
            (f"Cannot verify diff: {type(exc).__name__}.",),
        )

    print(json.dumps(asdict(report), indent=2, ensure_ascii=True))
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as stream:
            stream.write(render_summary(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
