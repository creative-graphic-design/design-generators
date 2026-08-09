from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_draft_prs() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_draft_prs.py"
    spec = importlib.util.spec_from_file_location("check_draft_prs", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_draft_prs"] = module
    spec.loader.exec_module(module)
    return module


check_draft_prs = load_check_draft_prs()


class FakeClient:
    def __init__(self) -> None:
        self.pull_requests = [
            check_draft_prs.PullRequest(
                number=166,
                title="Complete draft",
                url="https://github.com/creative-graphic-design/design-generators/pull/166",
                body="## Summary\n\nDone.\n",
                head_oid="abc123",
                is_draft=True,
                state="OPEN",
                mergeable="MERGEABLE",
            )
        ]
        self.status_contexts = {
            "abc123": [
                check_draft_prs.CheckContext(
                    kind="CheckRun",
                    name="CI",
                    status="COMPLETED",
                    conclusion="SUCCESS",
                    state=None,
                    started_at="2026-08-04T12:00:00Z",
                )
            ]
        }
        self.review_threads = {166: []}
        self.comments = {166: []}
        self.created_comments: list[tuple[int, str]] = []
        self.updated_comments: list[tuple[int, str]] = []

    def list_open_pull_requests(self) -> list[object]:
        return self.pull_requests

    def get_pull_request(self, number: int) -> object | None:
        for pull_request in self.pull_requests:
            if pull_request.number == number:
                return pull_request
        return None

    def list_review_threads(self, number: int) -> list[bool]:
        return self.review_threads[number]

    def list_status_contexts(self, head_oid: str) -> list[object]:
        return self.status_contexts[head_oid]

    def list_issue_comments(self, number: int) -> list[object]:
        return self.comments[number]

    def create_issue_comment(self, number: int, body: str) -> None:
        self.created_comments.append((number, body))

    def update_issue_comment(self, comment_id: int, body: str) -> None:
        self.updated_comments.append((comment_id, body))


def test_has_draft_reason_section_ignores_commented_template_stub() -> None:
    body = (
        "## Summary\n\nDone.\n\n"
        "<!-- Optional hold section:\n"
        "## Draft Reason\n"
        "State the blocker here.\n"
        "-->\n"
    )

    assert check_draft_prs.has_draft_reason_section(body) is False


def test_has_draft_reason_section_accepts_real_heading() -> None:
    body = "## Summary\n\nDone.\n\n## Draft Reason\n\nWaiting for maintainer review.\n"

    assert check_draft_prs.has_draft_reason_section(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "## Summary\n\nDone.\n\n```md\n## Draft Reason\nNot real.\n```\n",
        "## Summary\n\nDone.\n\n~~~md\n## Draft Reason\nNot real.\n~~~\n",
        "## Summary\n\nDone.\n\n## Draft Reason\n\n## Deviations\n\n- None\n",
        "## Summary\n\nDone.\n\n## Draft Reason\n\n### Nested\n",
        "## Summary\n\nDone.\n\n### Draft Reason\n\nWaiting.\n",
    ],
)
def test_has_draft_reason_section_rejects_non_actionable_or_wrong_headings(
    body: str,
) -> None:
    assert check_draft_prs.has_draft_reason_section(body) is False


def test_head_checks_all_successful_requires_at_least_one_check() -> None:
    assert check_draft_prs.head_checks_all_successful([]) is False


def test_head_checks_all_successful_accepts_success_status_and_check_runs() -> None:
    contexts = [
        check_draft_prs.CheckContext(
            kind="StatusContext",
            name="legacy",
            status=None,
            conclusion=None,
            state="SUCCESS",
            started_at="2026-08-04T12:00:00Z",
        ),
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="CI",
            status="COMPLETED",
            conclusion="SUCCESS",
            state=None,
            started_at="2026-08-04T12:01:00Z",
        ),
    ]

    assert check_draft_prs.head_checks_all_successful(contexts) is True


def test_head_checks_all_successful_rejects_pending_or_failing_contexts() -> None:
    pending = check_draft_prs.CheckContext(
        kind="CheckRun",
        name="CI",
        status="IN_PROGRESS",
        conclusion=None,
        state=None,
        started_at="2026-08-04T12:00:00Z",
    )
    failing = check_draft_prs.CheckContext(
        kind="StatusContext",
        name="legacy",
        status=None,
        conclusion=None,
        state="FAILURE",
        started_at="2026-08-04T12:00:00Z",
    )

    assert check_draft_prs.head_checks_all_successful([pending]) is False
    assert check_draft_prs.head_checks_all_successful([failing]) is False


def test_head_checks_all_successful_rejects_all_skipped() -> None:
    contexts = [
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="skipped",
            status="COMPLETED",
            conclusion="SKIPPED",
            state=None,
            started_at="2026-08-04T12:00:00Z",
        )
    ]

    assert check_draft_prs.head_checks_all_successful(contexts) is False


def test_head_checks_all_successful_rejects_expected_status() -> None:
    contexts = [
        check_draft_prs.CheckContext(
            kind="StatusContext",
            name="external",
            status=None,
            conclusion=None,
            state="EXPECTED",
            started_at=None,
        )
    ]

    assert check_draft_prs.head_checks_all_successful(contexts) is False


def test_audit_flags_complete_draft_without_draft_reason() -> None:
    client = FakeClient()

    violations = check_draft_prs.audit_draft_prs(client)

    assert violations == [
        check_draft_prs.DraftPrViolation(
            number=166,
            title="Complete draft",
            url="https://github.com/creative-graphic-design/design-generators/pull/166",
            reason=(
                "head checks all successful, unresolved review threads: 0, "
                "missing `## Draft Reason`"
            ),
        )
    ]


@pytest.mark.parametrize(
    ("body", "contexts", "threads"),
    [
        (
            "## Summary\n\nDone.\n\n## Draft Reason\n\nWaiting on S5.\n",
            [
                check_draft_prs.CheckContext(
                    kind="CheckRun",
                    name="CI",
                    status="COMPLETED",
                    conclusion="SUCCESS",
                    state=None,
                    started_at="2026-08-04T12:00:00Z",
                )
            ],
            [],
        ),
        (
            "## Summary\n\nDone.\n",
            [
                check_draft_prs.CheckContext(
                    kind="CheckRun",
                    name="CI",
                    status="COMPLETED",
                    conclusion="FAILURE",
                    state=None,
                    started_at="2026-08-04T12:00:00Z",
                )
            ],
            [],
        ),
        (
            "## Summary\n\nDone.\n",
            [
                check_draft_prs.CheckContext(
                    kind="CheckRun",
                    name="CI",
                    status="COMPLETED",
                    conclusion="SUCCESS",
                    state=None,
                    started_at="2026-08-04T12:00:00Z",
                )
            ],
            [False],
        ),
    ],
)
def test_audit_ignores_drafts_that_are_not_violations(
    body: str, contexts: list[object], threads: list[bool]
) -> None:
    client = FakeClient()
    client.pull_requests[0] = check_draft_prs.PullRequest(
        number=166,
        title="Not a violation",
        url="https://github.com/creative-graphic-design/design-generators/pull/166",
        body=body,
        head_oid="abc123",
        is_draft=True,
        state="OPEN",
        mergeable="MERGEABLE",
    )
    client.status_contexts["abc123"] = contexts
    client.review_threads[166] = threads

    assert check_draft_prs.audit_draft_prs(client) == []


def test_head_checks_all_successful_uses_latest_context_by_name() -> None:
    contexts = [
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="CI",
            status="COMPLETED",
            conclusion="SUCCESS",
            state=None,
            started_at="2026-08-04T12:00:00Z",
        ),
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="CI",
            status="IN_PROGRESS",
            conclusion=None,
            state=None,
            started_at="2026-08-04T12:05:00Z",
        ),
    ]

    assert check_draft_prs.head_checks_all_successful(contexts) is False


def test_head_checks_all_successful_treats_queued_without_started_at_as_latest() -> (
    None
):
    contexts = [
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="CI",
            status="COMPLETED",
            conclusion="SUCCESS",
            state=None,
            started_at="2026-08-04T12:00:00Z",
        ),
        check_draft_prs.CheckContext(
            kind="CheckRun",
            name="CI",
            status="QUEUED",
            conclusion=None,
            state=None,
            started_at=None,
        ),
    ]

    assert check_draft_prs.head_checks_all_successful(contexts) is False


def test_audit_ignores_conflicting_draft_prs() -> None:
    client = FakeClient()
    client.pull_requests[0] = check_draft_prs.PullRequest(
        number=166,
        title="Conflicting draft",
        url="https://github.com/creative-graphic-design/design-generators/pull/166",
        body="## Summary\n\nDone.\n",
        head_oid="abc123",
        is_draft=True,
        state="OPEN",
        mergeable="CONFLICTING",
    )

    assert check_draft_prs.audit_draft_prs(client) == []
    assert check_draft_prs.explain_draft_prs(client) == [
        "#166 Conflicting draft: ok; PR has merge conflicts"
    ]


def test_post_or_update_audit_comment_creates_when_missing() -> None:
    client = FakeClient()
    violation = check_draft_prs.audit_draft_prs(client)[0]

    check_draft_prs.post_or_update_audit_comment(client, violation)

    assert client.created_comments == [(166, check_draft_prs.AUDIT_COMMENT_BODY)]
    assert client.updated_comments == []


def test_post_or_update_audit_comment_updates_existing_marker_comment() -> None:
    client = FakeClient()
    client.comments[166] = [
        check_draft_prs.IssueComment(
            database_id=123,
            body=f"{check_draft_prs.AUDIT_COMMENT_MARKER}\nOld body",
        )
    ]
    violation = check_draft_prs.audit_draft_prs(client)[0]

    check_draft_prs.post_or_update_audit_comment(client, violation)

    assert client.created_comments == []
    assert client.updated_comments == [(123, check_draft_prs.AUDIT_COMMENT_BODY)]
