"""Audit complete draft pull requests that lack an explicit draft reason."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol, cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

DRAFT_REASON_HEADING_RE = re.compile(r"(?im)^## Draft Reason\s*$")
SECTION_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FENCED_BLOCK_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=fence)[ \t]*$"
)
AUDIT_COMMENT_MARKER = "<!-- design-generators:draft-pr-audit -->"
AUDIT_COMMENT_BODY = (
    f"{AUDIT_COMMENT_MARKER}\n"
    "## Draft PR Audit\n\n"
    "This draft PR currently looks complete by machine checks: the head checks "
    "are all passing, there are no unresolved review threads, and the PR body "
    "has no `## Draft Reason` section.\n\n"
    "Please mark it ready for review or add `## Draft Reason` with the blocking "
    "condition and resolution trigger."
)
PASSING_CHECK_RUN_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED"}


@dataclass(frozen=True)
class PullRequest:
    """Draft pull request metadata required for auditing."""

    number: int
    title: str
    url: str
    body: str
    head_oid: str
    is_draft: bool
    state: str
    mergeable: str | None


@dataclass(frozen=True)
class CheckContext:
    """A status or check-run context from a pull request head commit."""

    kind: str
    name: str
    status: str | None
    conclusion: str | None
    state: str | None
    started_at: str | None


@dataclass(frozen=True)
class IssueComment:
    """Issue comment metadata used for idempotent audit comments."""

    database_id: int
    body: str


@dataclass(frozen=True)
class DraftPrViolation:
    """A complete draft pull request without a draft reason."""

    number: int
    title: str
    url: str
    reason: str


class DraftPrClient(Protocol):
    """GitHub operations required by the draft PR audit."""

    def list_open_pull_requests(self) -> list[PullRequest]:
        """Return open pull requests."""

    def get_pull_request(self, number: int) -> PullRequest | None:
        """Return one pull request by number."""

    def list_review_threads(self, number: int) -> list[bool]:
        """Return review thread resolved states for one pull request."""

    def list_status_contexts(self, head_oid: str) -> list[CheckContext]:
        """Return head commit status and check-run contexts."""

    def list_issue_comments(self, number: int) -> list[IssueComment]:
        """Return issue comments for one pull request."""

    def create_issue_comment(self, number: int, body: str) -> None:
        """Create one issue comment."""

    def update_issue_comment(self, comment_id: int, body: str) -> None:
        """Update one issue comment."""


class GhCliClient:
    """GitHub client backed by `gh api` and the GH_TOKEN environment variable."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.owner, self.name = repo.split("/", 1)

    def _run_gh(self, command: list[str], *, stdin: str | None = None) -> str:
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            env=os.environ,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"`{' '.join(command)}` failed with exit code "
                f"{result.returncode}: {message}"
            )
        return result.stdout

    def _graphql(self, query: str, **variables: JsonValue) -> JsonObject:
        command = ["gh", "api", "graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            if value is None:
                continue
            command.extend(["-F", f"{key}={value}"])
        payload = json.loads(self._run_gh(command))
        if not isinstance(payload, dict):
            raise TypeError("GraphQL response must be a JSON object")
        return payload

    def _rest(
        self,
        method: str,
        path: str,
        *,
        body: str | None = None,
        paginate: bool = False,
    ) -> JsonValue:
        command = ["gh", "api", "-X", method]
        if paginate:
            command.append("--paginate")
        command.append(path)
        stdin = None
        if body is not None:
            command.extend(["--input", "-"])
            stdin = json.dumps({"body": body})
        output = self._run_gh(command, stdin=stdin)
        if not output.strip():
            return None
        return json.loads(output)

    def list_open_pull_requests(self) -> list[PullRequest]:
        query = """
        query DraftPrAuditPullRequests($owner: String!, $name: String!, $after: String) {
          repository(owner: $owner, name: $name) {
            pullRequests(first: 100, states: OPEN, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
              nodes {
                number
                title
                url
                body
                isDraft
                state
                mergeable
                headRefOid
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
        }
        """
        pull_requests: list[PullRequest] = []
        cursor: str | None = None
        while True:
            payload = self._graphql(
                query,
                owner=self.owner,
                name=self.name,
                after=cursor,
            )
            connection = _repository_field(payload, "pullRequests")
            for node in _nodes(connection):
                pull_requests.append(_pull_request_from_node(node))
            page_info = _page_info(connection)
            if not page_info.get("hasNextPage"):
                return pull_requests
            cursor = str(page_info.get("endCursor") or "")

    def get_pull_request(self, number: int) -> PullRequest | None:
        query = """
        query DraftPrAuditPullRequest($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              number
              title
              url
              body
              isDraft
              state
              mergeable
              headRefOid
            }
          }
        }
        """
        payload = self._graphql(
            query,
            owner=self.owner,
            name=self.name,
            number=number,
        )
        repository = _repository(payload)
        node = repository.get("pullRequest")
        if node is None:
            return None
        return _pull_request_from_node(_object(node))

    def list_review_threads(self, number: int) -> list[bool]:
        query = """
        query DraftPrAuditReviewThreads($owner: String!, $name: String!, $number: Int!, $after: String) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              reviewThreads(first: 100, after: $after) {
                nodes {
                  isResolved
                }
                pageInfo {
                  hasNextPage
                  endCursor
                }
              }
            }
          }
        }
        """
        resolved_states: list[bool] = []
        cursor: str | None = None
        while True:
            payload = self._graphql(
                query,
                owner=self.owner,
                name=self.name,
                number=number,
                after=cursor,
            )
            pull_request = _pull_request_field(payload)
            connection = _object(pull_request.get("reviewThreads"))
            for node in _nodes(connection):
                resolved_states.append(bool(node.get("isResolved")))
            page_info = _page_info(connection)
            if not page_info.get("hasNextPage"):
                return resolved_states
            cursor = str(page_info.get("endCursor") or "")

    def list_status_contexts(self, head_oid: str) -> list[CheckContext]:
        query = """
        query DraftPrAuditStatusContexts($owner: String!, $name: String!, $oid: GitObjectID!, $after: String) {
          repository(owner: $owner, name: $name) {
            object(oid: $oid) {
              ... on Commit {
                statusCheckRollup {
                  contexts(first: 100, after: $after) {
                    nodes {
                      __typename
                      ... on CheckRun {
                        name
                        status
                        conclusion
                        startedAt
                      }
                      ... on StatusContext {
                        context
                        state
                        createdAt
                      }
                    }
                    pageInfo {
                      hasNextPage
                      endCursor
                    }
                  }
                }
              }
            }
          }
        }
        """
        contexts: list[CheckContext] = []
        cursor: str | None = None
        while True:
            payload = self._graphql(
                query,
                owner=self.owner,
                name=self.name,
                oid=head_oid,
                after=cursor,
            )
            repository = _repository(payload)
            commit = _object(repository.get("object"))
            rollup = commit.get("statusCheckRollup")
            if rollup is None:
                return contexts
            connection = _object(_object(rollup).get("contexts"))
            for node in _nodes(connection):
                contexts.append(_check_context_from_node(node))
            page_info = _page_info(connection)
            if not page_info.get("hasNextPage"):
                return contexts
            cursor = str(page_info.get("endCursor") or "")

    def list_issue_comments(self, number: int) -> list[IssueComment]:
        payload = self._rest(
            "GET",
            f"repos/{self.repo}/issues/{number}/comments",
            paginate=True,
        )
        if not isinstance(payload, list):
            return []
        comments: list[IssueComment] = []
        for node in payload:
            if not isinstance(node, dict):
                continue
            comments.append(
                IssueComment(
                    database_id=_int_value(node.get("id")),
                    body=str(node.get("body") or ""),
                )
            )
        return comments

    def create_issue_comment(self, number: int, body: str) -> None:
        self._rest("POST", f"repos/{self.repo}/issues/{number}/comments", body=body)

    def update_issue_comment(self, comment_id: int, body: str) -> None:
        self._rest(
            "PATCH", f"repos/{self.repo}/issues/comments/{comment_id}", body=body
        )


def _object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    return cast("JsonObject", value)


def _required_object(value: JsonValue, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"GitHub API response is missing object `{path}`")
    return cast("JsonObject", value)


def _repository(payload: JsonObject) -> JsonObject:
    data = _required_object(payload.get("data"), "data")
    return _required_object(data.get("repository"), "data.repository")


def _int_value(value: JsonValue) -> int:
    if isinstance(value, int | str):
        return int(value)
    return 0


def _repository_field(payload: JsonObject, field: str) -> JsonObject:
    repository = _repository(payload)
    return _required_object(repository.get(field), f"data.repository.{field}")


def _pull_request_field(payload: JsonObject) -> JsonObject:
    repository = _repository(payload)
    return _required_object(
        repository.get("pullRequest"), "data.repository.pullRequest"
    )


def _nodes(connection: JsonObject) -> list[JsonObject]:
    nodes = connection.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [_object(node) for node in nodes]


def _page_info(connection: JsonObject) -> JsonObject:
    return _required_object(connection.get("pageInfo"), "pageInfo")


def _pull_request_from_node(node: JsonObject) -> PullRequest:
    return PullRequest(
        number=_int_value(node.get("number")),
        title=str(node.get("title") or ""),
        url=str(node.get("url") or ""),
        body=str(node.get("body") or ""),
        head_oid=str(node.get("headRefOid") or ""),
        is_draft=bool(node.get("isDraft")),
        state=str(node.get("state") or ""),
        mergeable=(
            str(node.get("mergeable")) if node.get("mergeable") is not None else None
        ),
    )


def _check_context_from_node(node: JsonObject) -> CheckContext:
    kind = str(node.get("__typename") or "")
    return CheckContext(
        kind=kind,
        name=str(node.get("name") or node.get("context") or ""),
        status=str(node.get("status")) if node.get("status") is not None else None,
        conclusion=(
            str(node.get("conclusion")) if node.get("conclusion") is not None else None
        ),
        state=str(node.get("state")) if node.get("state") is not None else None,
        started_at=(
            str(node.get("startedAt") or node.get("createdAt"))
            if node.get("startedAt") is not None or node.get("createdAt") is not None
            else None
        ),
    )


def has_draft_reason_section(body: str) -> bool:
    """Return whether a PR body contains a real Draft Reason heading."""
    cleaned_body = FENCED_BLOCK_RE.sub("", HTML_COMMENT_RE.sub("", body))
    heading = DRAFT_REASON_HEADING_RE.search(cleaned_body)
    if heading is None:
        return False
    next_section = SECTION_HEADING_RE.search(cleaned_body, heading.end())
    section_end = len(cleaned_body) if next_section is None else next_section.start()
    section = cleaned_body[heading.end() : section_end]
    return any(
        line.strip() and not line.lstrip().startswith("#")
        for line in section.splitlines()
    )


def check_context_passed(context: CheckContext) -> bool:
    """Return whether one status or check-run context is successful."""
    if context.kind == "StatusContext":
        return context.state == "SUCCESS"
    if context.kind == "CheckRun":
        return (
            context.status == "COMPLETED"
            and context.conclusion in PASSING_CHECK_RUN_CONCLUSIONS
        )
    return False


def check_context_succeeded(context: CheckContext) -> bool:
    """Return whether one status or check-run is a true success signal."""
    if context.kind == "StatusContext":
        return context.state == "SUCCESS"
    if context.kind == "CheckRun":
        return context.status == "COMPLETED" and context.conclusion == "SUCCESS"
    return False


def check_context_incomplete(context: CheckContext) -> bool:
    """Return whether one status or check-run is still pending or expected."""
    if context.kind == "StatusContext":
        return context.state in {"PENDING", "EXPECTED"}
    if context.kind == "CheckRun":
        return context.status != "COMPLETED"
    return True


def _context_time_key(context: CheckContext) -> str:
    return context.started_at or "9999-12-31T23:59:59Z"


def latest_status_contexts(contexts: list[CheckContext]) -> list[CheckContext]:
    """Return the latest context for each GitHub check/status name."""
    latest_by_name: dict[str, CheckContext] = {}
    for context in contexts:
        existing = latest_by_name.get(context.name)
        if existing is None:
            latest_by_name[context.name] = context
            continue
        existing_incomplete = check_context_incomplete(existing)
        context_incomplete = check_context_incomplete(context)
        if (
            context_incomplete
            and not existing_incomplete
            or context_incomplete == existing_incomplete
            and _context_time_key(context) >= _context_time_key(existing)
        ):
            latest_by_name[context.name] = context
    return list(latest_by_name.values())


def head_checks_all_successful(contexts: list[CheckContext]) -> bool:
    """Return whether the head commit has at least one check and all checks passed."""
    latest_contexts = latest_status_contexts(contexts)
    return (
        bool(latest_contexts)
        and all(check_context_passed(context) for context in latest_contexts)
        and any(check_context_succeeded(context) for context in latest_contexts)
    )


def pull_requests_to_audit(
    client: DraftPrClient, pr_numbers: list[int] | None
) -> list[PullRequest]:
    """Return pull requests selected for this audit run."""
    if pr_numbers is None:
        return [
            pull_request
            for pull_request in client.list_open_pull_requests()
            if pull_request.is_draft
        ]

    pull_requests: list[PullRequest] = []
    for number in pr_numbers:
        pull_request = client.get_pull_request(number)
        if pull_request is None:
            print(f"PR #{number} was not found.", file=sys.stderr)
            continue
        pull_requests.append(pull_request)
    return pull_requests


def audit_draft_prs(
    client: DraftPrClient,
    *,
    pr_numbers: list[int] | None = None,
) -> list[DraftPrViolation]:
    """Return complete draft pull requests that lack a Draft Reason section."""
    violations: list[DraftPrViolation] = []
    for pull_request in pull_requests_to_audit(client, pr_numbers):
        if pull_request.state != "OPEN" or not pull_request.is_draft:
            continue
        if pull_request.mergeable == "CONFLICTING":
            continue

        contexts = client.list_status_contexts(pull_request.head_oid)
        if not head_checks_all_successful(contexts):
            continue

        review_thread_states = client.list_review_threads(pull_request.number)
        if any(not is_resolved for is_resolved in review_thread_states):
            continue

        if has_draft_reason_section(pull_request.body):
            continue

        violations.append(
            DraftPrViolation(
                number=pull_request.number,
                title=pull_request.title,
                url=pull_request.url,
                reason=(
                    "head checks all successful, unresolved review threads: 0, "
                    "missing `## Draft Reason`"
                ),
            )
        )
    return violations


def explain_draft_prs(
    client: DraftPrClient,
    *,
    pr_numbers: list[int] | None = None,
) -> list[str]:
    """Return one-line audit explanations for selected pull requests."""
    explanations: list[str] = []
    for pull_request in pull_requests_to_audit(client, pr_numbers):
        prefix = f"#{pull_request.number} {pull_request.title}"
        if pull_request.state != "OPEN":
            explanations.append(
                f"{prefix}: ignored; PR is {pull_request.state.lower()}"
            )
            continue
        if not pull_request.is_draft:
            explanations.append(f"{prefix}: ignored; PR is ready for review")
            continue
        if pull_request.mergeable == "CONFLICTING":
            explanations.append(f"{prefix}: ok; PR has merge conflicts")
            continue

        contexts = client.list_status_contexts(pull_request.head_oid)
        latest_contexts = latest_status_contexts(contexts)
        passing_checks = head_checks_all_successful(contexts)
        review_thread_states = client.list_review_threads(pull_request.number)
        unresolved_threads = sum(
            1 for is_resolved in review_thread_states if not is_resolved
        )
        has_reason = has_draft_reason_section(pull_request.body)

        if passing_checks and unresolved_threads == 0 and not has_reason:
            explanations.append(
                f"{prefix}: violation; head checks all successful, unresolved "
                "review threads: 0, missing `## Draft Reason`"
            )
            continue

        blockers: list[str] = []
        if not passing_checks:
            blockers.append(
                f"head checks not all successful ({len(latest_contexts)} latest checks)"
            )
        if unresolved_threads:
            blockers.append(f"{unresolved_threads} unresolved review thread(s)")
        if has_reason:
            blockers.append("has `## Draft Reason`")
        explanations.append(f"{prefix}: ok; " + ", ".join(blockers))
    return explanations


def post_or_update_audit_comment(
    client: DraftPrClient, violation: DraftPrViolation
) -> None:
    """Create or update the single draft audit comment for a violation."""
    for comment in client.list_issue_comments(violation.number):
        if AUDIT_COMMENT_MARKER in comment.body:
            client.update_issue_comment(comment.database_id, AUDIT_COMMENT_BODY)
            return
    client.create_issue_comment(violation.number, AUDIT_COMMENT_BODY)


def _default_repo() -> str:
    env_repo = os.environ.get("GITHUB_REPOSITORY")
    if env_repo:
        return env_repo
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = result.stdout.strip()
    match = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\.git)?$", remote)
    if match is None:
        raise ValueError(
            "Could not infer repository from GITHUB_REPOSITORY or remote.origin.url"
        )
    return match.group("repo")


def main(argv: list[str] | None = None) -> int:
    """Run the draft PR audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository in owner/name form. Defaults to GITHUB_REPOSITORY or origin.",
    )
    parser.add_argument(
        "--pr",
        dest="prs",
        type=int,
        action="append",
        help="Audit one PR number. May be passed more than once.",
    )
    parser.add_argument(
        "--comment",
        action="store_true",
        help="Post or update audit comments on violating PRs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one-line explanations for audited PRs.",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("GH_TOKEN"):
        print("GH_TOKEN is required to query GitHub.", file=sys.stderr)
        return 2

    client = GhCliClient(args.repo or _default_repo())
    if args.verbose:
        for explanation in explain_draft_prs(client, pr_numbers=args.prs):
            print(explanation)

    violations = audit_draft_prs(client, pr_numbers=args.prs)
    if not violations:
        print("No complete draft PR audit violations found.")
        return 0

    print("Complete draft PR audit violations:")
    for violation in violations:
        print(f"- #{violation.number} {violation.title}: {violation.reason}")
        if args.comment:
            post_or_update_audit_comment(client, violation)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
