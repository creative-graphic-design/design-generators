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


def filled_body(reference: str = "Refs #127") -> str:
    checklist = "\n".join(f"- [x] {item}" for item in REQUIRED_CHECKLIST_ITEMS)
    return f"## Summary\n\n{reference}\n\n## Checklist\n\n{checklist}\n"


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
