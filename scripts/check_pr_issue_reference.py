"""Ensure pull requests reference a real issue and fill the PR checklist."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

DEFAULT_EXCLUDED_ISSUES = {2, 60}
COMPLETION_GATE_EFFECTIVE_AT = datetime(2026, 7, 25, tzinfo=UTC)
ISSUE_REF_RE = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|refs?|references?)\s+"
    r"(?:https://github\.com/creative-graphic-design/design-generators/issues/)?#?(\d+)"
)
CHECKLIST_HEADING_RE = re.compile(r"(?im)^## Checklist\s*$")
COMPLETION_GATE_HEADING_RE = re.compile(r"(?im)^## Completion Gate\s*$")
SECTION_HEADING_RE = re.compile(r"(?m)^##\s+")
CHECKBOX_RE = re.compile(r"(?m)^- \[(?P<state>[ xX])\]\s+(?P<text>.+?)\s*$")
PLACEHOLDER_RE = re.compile(r"<[^>]+>")


class PullRequestMetadata(NamedTuple):
    """Pull request metadata read from a GitHub Actions event."""

    body: str
    draft: bool
    created_at: datetime | None


class CompletionRequirement(NamedTuple):
    """A ready-for-review completion gate requirement."""

    name: str
    prefix: str
    allowed_justification: str | None
    justification_label: str | None


COMPLETION_REQUIREMENTS = (
    CompletionRequirement(
        name="vendor parity",
        prefix="Vendor parity verified, or gated-pending:",
        allowed_justification="gated-pending:",
        justification_label="a named blocker",
    ),
    CompletionRequirement(
        name="training S5",
        prefix="Training S5 reproduction complete, or N/A:",
        allowed_justification="N/A:",
        justification_label="a reason",
    ),
    CompletionRequirement(
        name="pre-PR adversarial review",
        prefix=(
            "Pre-PR adversarial review completed "
            "(reviewer spawned before opening the PR; findings resolved)"
        ),
        allowed_justification=None,
        justification_label=None,
    ),
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
    return read_pull_request_metadata_from_event(event_path).body


def _parse_github_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_pull_request_metadata_from_event(event_path: Path) -> PullRequestMetadata:
    """Read pull request metadata from a GitHub Actions event payload."""
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request", {})
    if not isinstance(pull_request, dict):
        pull_request = {}
    return PullRequestMetadata(
        body=str(pull_request.get("body") or ""),
        draft=bool(pull_request.get("draft", False)),
        created_at=_parse_github_datetime(pull_request.get("created_at")),
    )


def checklist_section(body: str) -> str | None:
    """Return the PR checklist section body, excluding the heading."""
    return section_body(body, CHECKLIST_HEADING_RE)


def completion_gate_section(body: str) -> str | None:
    """Return the PR completion gate section body, excluding the heading."""
    return section_body(body, COMPLETION_GATE_HEADING_RE)


def section_body(body: str, heading_re: re.Pattern[str]) -> str | None:
    """Return a markdown section body, excluding the heading."""
    heading = heading_re.search(body)
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


def _has_actionable_justification(text: str, marker: str) -> bool:
    _, _, reason = text.partition(marker)
    cleaned = PLACEHOLDER_RE.sub("", reason).strip(" .:-")
    return bool(cleaned)


def _find_completion_item(
    section: str, requirement: CompletionRequirement
) -> tuple[bool, str] | None:
    for match in CHECKBOX_RE.finditer(section):
        text = match.group("text")
        if text.startswith(requirement.prefix):
            return match.group("state").lower() == "x", text
    return None


def _legacy_body_without_completion_gate(
    created_at: datetime | None,
) -> bool:
    return created_at is not None and created_at < COMPLETION_GATE_EFFECTIVE_AT


def completion_gate_errors(
    body: str,
    *,
    draft: bool,
    created_at: datetime | None = None,
) -> list[str]:
    """Return ready-for-review completion gate validation errors."""
    if draft:
        return []

    section = completion_gate_section(body)
    if section is None:
        if _legacy_body_without_completion_gate(created_at):
            return []
        return [
            "Ready-for-review PRs must include a `## Completion Gate` section. "
            "Complete the items or convert the PR back to draft."
        ]

    errors: list[str] = []
    for requirement in COMPLETION_REQUIREMENTS:
        item = _find_completion_item(section, requirement)
        if item is None:
            errors.append(
                f"Ready-for-review PRs must include the completion item for "
                f"{requirement.name}: `{requirement.prefix}`. Complete it or "
                "convert the PR back to draft."
            )
            continue

        checked, text = item
        marker = requirement.allowed_justification
        if checked:
            continue
        if marker is not None and _has_actionable_justification(text, marker):
            continue
        if requirement.justification_label is None:
            errors.append(
                f"Ready-for-review PRs must check `{requirement.prefix}` or "
                "convert the PR back to draft."
            )
        else:
            errors.append(
                f"Ready-for-review PRs must complete `{requirement.prefix}` with "
                f"{requirement.justification_label} or convert the PR back to draft."
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate that a PR body links an issue and fills the checklist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-file", type=Path)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args(argv)

    if args.body_file is not None:
        metadata = PullRequestMetadata(
            body=args.body_file.read_text(encoding="utf-8"),
            draft=args.draft,
            created_at=None,
        )
    else:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            print("GITHUB_EVENT_PATH is required outside --body-file.", file=sys.stderr)
            return 2
        metadata = read_pull_request_metadata_from_event(Path(event_path))

    errors: list[str] = []
    references = valid_issue_references(metadata.body)
    if not references:
        errors.append(
            "PR body must include `Closes #N` or `Refs #N` for the implementation "
            "issue. Standing issues #2 and #60 do not satisfy this check."
        )

    template_path = (
        Path(__file__).resolve().parents[1] / ".github" / ("PULL_REQUEST_TEMPLATE.md")
    )
    errors.extend(
        checklist_errors(metadata.body, required_checklist_items(template_path))
    )
    errors.extend(
        completion_gate_errors(
            metadata.body,
            draft=metadata.draft,
            created_at=metadata.created_at,
        )
    )

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
