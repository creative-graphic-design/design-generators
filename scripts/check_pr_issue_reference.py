"""Ensure pull requests reference a real issue and fill the PR checklist."""

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
CHECKLIST_HEADING_RE = re.compile(r"(?im)^## Checklist\s*$")
SECTION_HEADING_RE = re.compile(r"(?m)^##\s+")
CHECKBOX_RE = re.compile(r"(?m)^- \[(?P<state>[ xX])\]\s+(?P<text>.+?)\s*$")


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


def checklist_section(body: str) -> str | None:
    """Return the PR checklist section body, excluding the heading."""
    heading = CHECKLIST_HEADING_RE.search(body)
    if heading is None:
        return None
    next_section = SECTION_HEADING_RE.search(body, heading.end())
    end = len(body) if next_section is None else next_section.start()
    return body[heading.end() : end]


def required_checklist_items(template_path: Path) -> list[str]:
    """Return required checklist item text from the PR template."""
    template = template_path.read_text(encoding="utf-8")
    section = checklist_section(template)
    if section is None:
        raise ValueError(f"{template_path} is missing a ## Checklist section")
    return [match.group("text") for match in CHECKBOX_RE.finditer(section)]


def checklist_errors(body: str, required_items: list[str]) -> list[str]:
    """Return checklist validation errors for a PR body."""
    section = checklist_section(body)
    if section is None:
        return ["PR body must include a `## Checklist` section."]

    checked: set[str] = set()
    unchecked: set[str] = set()
    for match in CHECKBOX_RE.finditer(section):
        item = match.group("text")
        if match.group("state").lower() == "x":
            checked.add(item)
        else:
            unchecked.add(item)

    errors: list[str] = []
    missing = [item for item in required_items if item not in checked]
    if missing:
        errors.append(
            "PR body checklist must include these checked template item(s): "
            + "; ".join(missing)
        )
    remaining_unchecked = [item for item in required_items if item in unchecked]
    if remaining_unchecked:
        errors.append(
            "PR body checklist has unchecked required item(s): "
            + "; ".join(remaining_unchecked)
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate that a PR body links an issue and fills the checklist."""
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

    errors: list[str] = []
    references = valid_issue_references(body)
    if not references:
        errors.append(
            "PR body must include `Closes #N` or `Refs #N` for the implementation "
            "issue. Standing issues #2 and #60 do not satisfy this check."
        )

    template_path = (
        Path(__file__).resolve().parents[1] / ".github" / ("PULL_REQUEST_TEMPLATE.md")
    )
    errors.extend(checklist_errors(body, required_checklist_items(template_path)))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    if references:
        print(
            "Found implementation issue reference(s): "
            + ", ".join(f"#{number}" for number in sorted(references))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
