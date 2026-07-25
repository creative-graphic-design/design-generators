from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_pr_issue_reference() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_pr_issue_reference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_pr_issue_reference", module_path
    )
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_pr_issue_reference = load_check_pr_issue_reference()
REQUIRED_CHECKLIST_ITEMS = [
    "Confirmed the applicable issue #60 checklist items.",
    "Referenced the implementation issue with `Closes #N` or `Refs #N` in the Summary; standing issues #2 and #60 alone do not satisfy this.",
    "Confirmed the implementation issue has a milestone and native Priority field set.",
    "Applied the same lane/topic labels as the implementation issue to this PR; status labels such as `plan-agreed`, `in-progress`, and `parity-verified` stay on the issue.",
    "Read the model plan and amendment comments, if this is a model PR.",
    "Left `vendor/` read-only and did not commit generated fixtures, weights, images, or downloaded artifacts.",
    "Did not push Hub repositories or model artifacts unless explicitly requested.",
    "Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.",
    "README reproducibility steps are copy-pasteable commands, if README docs changed.",
    "Documented any deviations from the plan, checklist, or repository conventions below.",
]
COMPLETION_GATE_ITEMS = [
    "Vendor parity verified, or gated-pending: <blocker name and short reason>.",
    "Training S5 reproduction complete, or N/A: <reason>.",
    "Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)",
]


def filled_body(reference: str = "Refs #127") -> str:
    checklist = "\n".join(f"- [x] {item}" for item in REQUIRED_CHECKLIST_ITEMS)
    completion_gate = "\n".join(f"- [x] {item}" for item in COMPLETION_GATE_ITEMS)
    return (
        f"## Summary\n\n{reference}\n\n"
        f"## Checklist\n\n{checklist}\n\n"
        f"## Completion Gate\n\n{completion_gate}\n"
    )


def completion_body(*, draft: bool = False, completion_gate: str) -> str:
    checklist = "\n".join(f"- [x] {item}" for item in REQUIRED_CHECKLIST_ITEMS)
    draft_note = "\n\nDraft only.\n" if draft else "\n"
    return (
        "## Summary\n\nRefs #127\n\n"
        f"## Checklist\n\n{checklist}\n"
        f"{draft_note}"
        "## Completion Gate\n\n"
        f"{completion_gate}\n"
    )


def test_valid_issue_references_accepts_refs_and_closes() -> None:
    body = "## Summary\n\nRefs #123 and Closes #124.\n"

    assert check_pr_issue_reference.valid_issue_references(body) == {123, 124}


def test_valid_issue_references_ignores_standing_policy_issues() -> None:
    body = "Full checklist: see issue #60.\n\nRefs #60 and Refs #2.\n"

    assert check_pr_issue_reference.valid_issue_references(body) == set()


def test_valid_issue_references_accepts_repository_issue_urls() -> None:
    body = (
        "Closes https://github.com/creative-graphic-design/design-generators/issues/125"
    )

    assert check_pr_issue_reference.valid_issue_references(body) == {125}


def test_read_body_from_event_reads_pull_request_body(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": "Refs #126"}}),
        encoding="utf-8",
    )

    assert check_pr_issue_reference.read_body_from_event(event_path) == "Refs #126"


def test_read_pull_request_metadata_from_event_reads_draft_and_created_at(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "Refs #126",
                    "draft": True,
                    "created_at": "2026-07-24T12:34:56Z",
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = check_pr_issue_reference.read_pull_request_metadata_from_event(
        event_path
    )

    assert metadata.body == "Refs #126"
    assert metadata.draft is True
    assert metadata.created_at.isoformat() == "2026-07-24T12:34:56+00:00"


def test_required_checklist_items_are_loaded_from_template() -> None:
    template_path = (
        Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md"
    )

    assert check_pr_issue_reference.required_checklist_items(template_path) == (
        REQUIRED_CHECKLIST_ITEMS
    )


def test_checklist_errors_require_checklist_section() -> None:
    assert check_pr_issue_reference.checklist_errors(
        "Refs #127", REQUIRED_CHECKLIST_ITEMS
    ) == ["PR body must include a `## Checklist` section."]


def test_checklist_errors_require_template_item_text() -> None:
    body = filled_body().replace(
        REQUIRED_CHECKLIST_ITEMS[0],
        "Confirmed a different checklist item.",
    )

    errors = check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS)

    assert "Confirmed the applicable issue #60 checklist items." in errors[0]


def test_checklist_errors_reject_unchecked_template_items() -> None:
    body = filled_body().replace(
        f"- [x] {REQUIRED_CHECKLIST_ITEMS[0]}",
        f"- [ ] {REQUIRED_CHECKLIST_ITEMS[0]}",
    )

    errors = check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS)

    assert any("unchecked required item" in error for error in errors)


def test_checklist_errors_accept_filled_template() -> None:
    assert (
        check_pr_issue_reference.checklist_errors(
            filled_body(), REQUIRED_CHECKLIST_ITEMS
        )
        == []
    )


def test_completion_gate_allows_draft_incomplete() -> None:
    body = completion_body(
        draft=True,
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: <blocker name and short reason>.\n"
            "- [ ] Training S5 reproduction complete, or N/A: <reason>.\n"
            "- [ ] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=True) == []


def test_completion_gate_rejects_ready_incomplete() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: <blocker name and short reason>.\n"
            "- [ ] Training S5 reproduction complete, or N/A: <reason>.\n"
            "- [ ] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert len(errors) == 3
    assert all("convert the PR back to draft" in error for error in errors)


def test_completion_gate_accepts_ready_complete() -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: <blocker name and short reason>.\n"
            "- [x] Training S5 reproduction complete, or N/A: <reason>.\n"
            "- [x] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=False) == []


def test_completion_gate_accepts_na_and_blocker_reasons() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: no redistributable checkpoint is available.\n"
            "- [ ] Training S5 reproduction complete, or N/A: documentation-only PR.\n"
            "- [x] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=False) == []


def test_completion_gate_rejects_unchecked_adversarial_review() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: no redistributable checkpoint is available.\n"
            "- [ ] Training S5 reproduction complete, or N/A: documentation-only PR.\n"
            "- [ ] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert errors == [
        "Ready-for-review PRs must check `Pre-PR adversarial review completed "
        "(reviewer spawned before opening the PR; findings resolved)` or convert "
        "the PR back to draft."
    ]


def test_completion_gate_allows_legacy_ready_body_without_section() -> None:
    body = "## Summary\n\nRefs #127\n\n## Checklist\n\n- [x] Existing item\n"

    assert (
        check_pr_issue_reference.completion_gate_errors(
            body,
            draft=False,
            created_at=check_pr_issue_reference.COMPLETION_GATE_EFFECTIVE_AT.replace(
                day=24
            ),
        )
        == []
    )


def test_main_fails_without_implementation_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text(filled_body("Refs #60"), encoding="utf-8")

    assert check_pr_issue_reference.main(["--body-file", str(body_path)]) == 1
    assert "implementation issue" in capsys.readouterr().err


def test_main_fails_without_checked_template_checklist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text("Refs #127\n", encoding="utf-8")

    assert check_pr_issue_reference.main(["--body-file", str(body_path)]) == 1
    assert "Checklist" in capsys.readouterr().err


def test_main_passes_with_implementation_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text(filled_body(), encoding="utf-8")

    assert check_pr_issue_reference.main(["--body-file", str(body_path)]) == 0
    assert "#127" in capsys.readouterr().out
