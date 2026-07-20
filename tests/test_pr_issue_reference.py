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


def test_main_fails_without_implementation_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text("Refs #60\n", encoding="utf-8")

    assert check_pr_issue_reference.main(["--body-file", str(body_path)]) == 1
    assert "implementation issue" in capsys.readouterr().err


def test_main_passes_with_implementation_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text("Refs #127\n", encoding="utf-8")

    assert check_pr_issue_reference.main(["--body-file", str(body_path)]) == 0
    assert "#127" in capsys.readouterr().out
