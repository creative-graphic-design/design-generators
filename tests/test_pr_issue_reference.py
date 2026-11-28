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
    "Confirmed the applicable implementation checklist items.",
    "Verify the issue reference, issue metadata, PR labels, and issue status follow [repository tracking rules](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#tracking).",
    "Verify model plans and amendments using the [repository source-of-truth rule](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#sources-of-truth), if this is a model PR.",
    "Follow the [model-conversion parity procedure](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-model-conversion/SKILL.md#establish-inference-parity) for vendor sources and generated artifacts.",
    "Did not push Hub repositories or model artifacts unless explicitly requested.",
    "Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.",
    "Follow the [documentation skill](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#model-readme-and-model-card) for README reproduction guidance, if README docs changed.",
    "Documented any deviations from the plan, checklist, or repository conventions below.",
]
COMPLETION_GATE_ITEMS = [
    "Vendor parity verified, or gated-pending: parity checks passed.",
    "Training S5 reproduction complete, or N/A: documentation-only change.",
    "Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)",
]


def filled_body(reference: str = "Refs #127") -> str:
    checklist = "\n".join(f"- [x] {item}" for item in REQUIRED_CHECKLIST_ITEMS)
    completion_gate = "\n".join(f"- [x] {item}" for item in COMPLETION_GATE_ITEMS)
    return (
        f"## Summary\n\n{reference}\n\n"
        f"## Checklist\n\n{checklist}\n\n"
        f"## Completion Gate\n\n{completion_gate}\n"
    )


def filled_body_with_shared_library_changes(reason: str) -> str:
    return filled_body() + f"\n## Shared Library Changes\n\n- {reason}\n"


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


def test_valid_issue_references_require_supported_summary_syntax() -> None:
    body = "## Summary\n\nFixes #123.\n\n## Changes\n\nRefs #124.\n"

    assert check_pr_issue_reference.valid_issue_references(body) == set()


def test_valid_issue_references_ignores_standing_policy_issues() -> None:
    body = (
        "## Summary\n\nFull checklist: see docs/implementation-checklist.md.\n\n"
        "Refs #60 and Refs #2.\n"
    )

    assert check_pr_issue_reference.valid_issue_references(body) == set()


def test_valid_issue_references_require_summary_section() -> None:
    body = "Refs #125\n\n## Changes\n\nMore details.\n"

    assert check_pr_issue_reference.valid_issue_references(body) == set()


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

    assert "Confirmed the applicable implementation checklist items." in errors[0]


def test_checklist_errors_accept_unchecked_template_items() -> None:
    body = filled_body().replace(
        f"- [x] {REQUIRED_CHECKLIST_ITEMS[0]}",
        f"- [ ] {REQUIRED_CHECKLIST_ITEMS[0]}",
    )

    assert (
        check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS) == []
    )


def test_checklist_errors_accept_filled_template() -> None:
    assert (
        check_pr_issue_reference.checklist_errors(
            filled_body(), REQUIRED_CHECKLIST_ITEMS
        )
        == []
    )


def test_checklist_errors_accept_deprecated_template_wording() -> None:
    body = filled_body()
    body = body.replace(
        REQUIRED_CHECKLIST_ITEMS[0],
        "Confirmed the applicable issue #60 checklist items.",
    )
    body = body.replace(
        REQUIRED_CHECKLIST_ITEMS[1],
        "Referenced the implementation issue with `Closes #N` or `Refs #N` in the Summary; standing issues #2 and #60 alone do not satisfy this.",
    )

    assert (
        check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS) == []
    )


@pytest.mark.parametrize(
    ("item_index", "legacy_item"),
    [
        (2, "Read the model plan and amendment comments, if this is a model PR."),
        (
            3,
            "Left `vendor/` read-only and did not commit generated fixtures, weights, "
            "images, or downloaded artifacts.",
        ),
        (
            6,
            "README reproducibility steps are copy-pasteable commands, "
            "if README docs changed.",
        ),
    ],
)
def test_checklist_errors_accept_previous_main_template_wording(
    item_index: int, legacy_item: str
) -> None:
    body = filled_body().replace(REQUIRED_CHECKLIST_ITEMS[item_index], legacy_item)

    assert (
        check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS) == []
    )


def test_checklist_errors_reject_missing_current_and_deprecated_wording() -> None:
    body = filled_body()
    body = body.replace(REQUIRED_CHECKLIST_ITEMS[0], "A different checklist item.")
    body = body.replace(
        REQUIRED_CHECKLIST_ITEMS[1], "Another different checklist item."
    )

    errors = check_pr_issue_reference.checklist_errors(body, REQUIRED_CHECKLIST_ITEMS)

    assert REQUIRED_CHECKLIST_ITEMS[0] in errors[0]
    assert REQUIRED_CHECKLIST_ITEMS[1] in errors[0]


def test_completion_gate_allows_draft_incomplete() -> None:
    body = completion_body(
        draft=True,
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: <blocker name and short reason>.\n"
            "- [ ] Training S5 reproduction complete, or N/A: <reason>.\n"
            "- [ ] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=True) == []


def test_completion_gate_rejects_ready_incomplete() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: <blocker name and short reason>.\n"
            "- [ ] Training S5 reproduction complete, or N/A: <reason>.\n"
            "- [ ] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert len(errors) == 3
    assert all("convert the PR back to draft" in error for error in errors)


def test_completion_gate_accepts_ready_complete() -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: parity checks passed.\n"
            "- [x] Training S5 reproduction complete, or N/A: documentation-only change.\n"
            "- [x] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=False) == []


def test_completion_gate_accepts_comparisons_and_autolinks() -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: max diff < 1e-5; "
            "cosine > 0.9999; x<a and b>y; run <https://example.org/parity>.\n"
            "- [x] Training S5 reproduction complete, or N/A: documentation-only change.\n"
            "- [x] Independent adversarial review completed before ready for review "
            "(reviewer with no prior involvement reviewed the head being marked "
            "ready; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=False) == []


def test_completion_gate_checks_verbatim_template_placeholders() -> None:
    template_path = (
        Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md"
    )
    template = template_path.read_text(encoding="utf-8")

    errors = check_pr_issue_reference.completion_gate_errors(template, draft=False)

    assert any("Training S5 reproduction complete" in error for error in errors)


@pytest.mark.parametrize("checkbox", ["[ ]", "[x]"])
def test_completion_gate_rejects_untouched_s5_template_placeholders(
    checkbox: str,
) -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: parity checks passed.\n"
            f"- {checkbox} Training S5 reproduction complete, or N/A: "
            "<claimed datasets and training/evaluation seed scope, with S0-S4 "
            "evidence cited before S5; or N/A with reason>. Infrastructure-only "
            "work must state that full-run reproduction is not claimed.\n"
            "- [x] Independent adversarial review completed before ready for review "
            "(reviewer with no prior involvement reviewed the head being marked "
            "ready; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert len(errors) == 1
    assert "Training S5 reproduction complete" in errors[0]


@pytest.mark.parametrize(
    "placeholder",
    ["<...>", "<independent rerun, ...>", "<datasets, ...>", "<path>"],
)
def test_completion_gate_rejects_checked_placeholder_forms(
    placeholder: str,
) -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: "
            f"parity checks passed with {placeholder}.\n"
            "- [x] Training S5 reproduction complete, or N/A: documentation-only change.\n"
            "- [x] Independent adversarial review completed before ready for review "
            "(reviewer with no prior involvement reviewed the head being marked "
            "ready; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert len(errors) == 1
    assert "Vendor parity verified" in errors[0]


def test_completion_gate_accepts_na_and_blocker_reasons() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: no redistributable checkpoint is available.\n"
            "- [ ] Training S5 reproduction complete, or N/A: documentation-only PR.\n"
            "- [x] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)"
        ),
    )

    assert check_pr_issue_reference.completion_gate_errors(body, draft=False) == []


def test_completion_gate_rejects_unchecked_adversarial_review() -> None:
    body = completion_body(
        completion_gate=(
            "- [ ] Vendor parity verified, or gated-pending: no redistributable checkpoint is available.\n"
            "- [ ] Training S5 reproduction complete, or N/A: documentation-only PR.\n"
            "- [ ] Independent adversarial review completed before ready for review (reviewer with no prior involvement reviewed the head being marked ready; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert errors == [
        "Ready-for-review PRs must check `Independent adversarial review completed "
        "before ready for review (reviewer with no prior involvement reviewed the "
        "head being marked ready; findings resolved)` or convert "
        "the PR back to draft."
    ]


def test_completion_gate_rejects_checked_legacy_adversarial_review() -> None:
    body = completion_body(
        completion_gate=(
            "- [x] Vendor parity verified, or gated-pending: not applicable.\n"
            "- [x] Training S5 reproduction complete, or N/A: documentation-only PR.\n"
            "- [x] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)"
        ),
    )

    errors = check_pr_issue_reference.completion_gate_errors(body, draft=False)

    assert errors == [
        "Ready-for-review PRs must include the completion item for "
        "independent adversarial review: `Independent adversarial review completed "
        "before ready for review (reviewer with no prior involvement reviewed the "
        "head being marked ready; findings resolved)`. Complete it or convert the "
        "PR back to draft."
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


def test_guarded_cross_cutting_paths_include_lib_src_docs_and_agent_skills() -> None:
    changed_files = [
        "lib/laygen/src/laygen/common/vendor.py",
        "docs/getting-started.md",
        ".agents/skills/design-generators-model-conversion/SKILL.md",
        "models/ltnet/README.md",
    ]

    assert check_pr_issue_reference.guarded_cross_cutting_paths(changed_files) == [
        "lib/laygen/src/laygen/common/vendor.py",
        "docs/getting-started.md",
        ".agents/skills/design-generators-model-conversion/SKILL.md",
    ]


def test_shared_library_change_errors_require_reason_for_docs_changes() -> None:
    errors = check_pr_issue_reference.shared_library_change_errors(
        filled_body(),
        ["docs/getting-started.md"],
    )

    assert errors == [
        "PRs changing cross-cutting paths must explain the rationale in "
        "`## Shared Library Changes`: docs/getting-started.md"
    ]


def test_shared_library_change_errors_require_reason_for_agent_skill_changes() -> None:
    errors = check_pr_issue_reference.shared_library_change_errors(
        filled_body() + "\n## Shared Library Changes\n\n- N/A\n",
        [
            ".agents/skills/design-generators-model-conversion/references/model-readme-template.md"
        ],
    )

    assert errors == [
        "PRs changing cross-cutting paths must explain the rationale in "
        "`## Shared Library Changes`: "
        ".agents/skills/design-generators-model-conversion/references/model-readme-template.md"
    ]


@pytest.mark.parametrize("placeholder", ["TODO", "N/A"])
def test_shared_library_change_errors_ignore_template_comment(
    placeholder: str,
) -> None:
    template_path = (
        Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md"
    )
    template = template_path.read_text(encoding="utf-8")
    section = check_pr_issue_reference.shared_library_section(template)
    assert section is not None
    section = section.replace("- TODO", f"- {placeholder}", 1)
    body = f"## Shared Library Changes\n{section}"

    errors = check_pr_issue_reference.shared_library_change_errors(
        body, ["docs/architecture.md"]
    )

    assert len(errors) == 1
    assert "Shared Library Changes" in errors[0]


def test_shared_library_change_errors_accept_actionable_reason() -> None:
    assert (
        check_pr_issue_reference.shared_library_change_errors(
            filled_body_with_shared_library_changes(
                "Shared template wording changed to keep model READMEs consistent."
            ),
            [
                ".agents/skills/design-generators-model-conversion/references/model-readme-template.md"
            ],
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


def test_main_requires_issue_reference_in_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = filled_body().replace("Refs #127", "Documentation changes")
    body += "\n## Changes\n\nRefs #128\n"
    body_path = tmp_path / "body.md"
    body_path.write_text(body, encoding="utf-8")

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


def test_main_fails_for_guarded_docs_change_without_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body_path = tmp_path / "body.md"
    body_path.write_text(filled_body(), encoding="utf-8")

    assert (
        check_pr_issue_reference.main(
            [
                "--body-file",
                str(body_path),
                "--changed-file",
                "docs/training-reproduction.md",
            ]
        )
        == 1
    )
    assert "Shared Library Changes" in capsys.readouterr().err
