"""Ensure pull requests reference a real implementation issue."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_EXCLUDED_ISSUES = {2, 60}
ISSUE_REF_RE = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|refs?|references?)\s+"
    r"(?:https://github\.com/creative-graphic-design/design-generators/issues/)?#?(\d+)"
)


def issue_references(body: str) -> set[int]:
    """Return implementation issue numbers referenced by PR body keywords."""
    return {int(match) for match in ISSUE_REF_RE.findall(body)}


def valid_issue_references(
    body: str, excluded_issues: set[int] | None = None
) -> set[int]:
    """Return PR issue references that are not standing policy/checklist issues."""
    excluded = DEFAULT_EXCLUDED_ISSUES if excluded_issues is None else excluded_issues
    return issue_references(body) - excluded


def read_body_from_event(event_path: Path) -> str:
    """Read the pull request body from a GitHub Actions event payload."""
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    return str(payload.get("pull_request", {}).get("body") or "")


def main(argv: list[str] | None = None) -> int:
    """Validate that a PR body links an implementation issue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-file", type=Path)
    args = parser.parse_args(argv)

    if args.body_file is not None:
        body = args.body_file.read_text(encoding="utf-8")
    else:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            print("GITHUB_EVENT_PATH is required outside --body-file.", file=sys.stderr)
            return 2
        body = read_body_from_event(Path(event_path))

    references = valid_issue_references(body)
    if references:
        print(
            "Found implementation issue reference(s): "
            + ", ".join(f"#{number}" for number in sorted(references))
        )
        return 0

    print(
        "PR body must include `Closes #N` or `Refs #N` for the implementation "
        "issue. Standing issues #2 and #60 do not satisfy this check.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
