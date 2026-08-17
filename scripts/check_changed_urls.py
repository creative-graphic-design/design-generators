"""Check newly added external URLs in a pull request diff."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import socket
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
IGNORE_PATH = ROOT / ".lycheeignore"
URL_RE = re.compile(r"https?://[^\s<>'\"\\]+")
VCS_GIT_REVISION_RE = re.compile(
    r"^(?P<repo>https?://[^\s<>'\"\\]+?\.git)@(?P<revision>[^/?#]+)(?:[?#].*)?$"
)
TRAILING_PUNCTUATION = ".,;:!?"
TRANSIENT_STATUS_MIN = 500
HARD_FAIL_STATUSES = {404, 410}
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_RETRIES = 2
USER_AGENT = "design-generators-changed-url-check/1.0"
EXCLUDED_DIFF_FILES = {".lycheeignore"}
REPO_MAIN_BLOB_URL = (
    "https://github.com/creative-graphic-design/design-generators/blob/main/"
)


@dataclass(frozen=True)
class ChangedUrl:
    """URL occurrence found on an added diff line."""

    url: str
    source: str


@dataclass(frozen=True)
class UrlCheckResult:
    """Result of checking one URL."""

    url: str
    outcome: str
    status: int | None = None
    error: str | None = None

    def format(self) -> str:
        """Return a stable human-readable summary line."""
        details: list[str] = [self.url, self.outcome]
        if self.status is not None:
            details.append(f"status={self.status}")
        if self.error is not None:
            details.append(f"error={self.error}")
        return " | ".join(details)


UrlChecker = Callable[[str], UrlCheckResult]


def normalize_url(raw_url: str) -> str:
    """Strip punctuation that commonly follows URLs in prose or Markdown."""
    url = raw_url.rstrip(TRAILING_PUNCTUATION)
    while url.endswith((")", "]", "}")) and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def probe_url(url: str) -> str:
    """Return the HTTP URL used for probing an extracted URL."""
    match = VCS_GIT_REVISION_RE.match(url)
    if match is not None:
        return match.group("repo")
    return url


def extract_urls_from_added_lines(diff_text: str) -> list[ChangedUrl]:
    """Return HTTP(S) URLs that appear on added diff lines."""
    urls: list[ChangedUrl] = []
    current_file = "<diff>"
    new_line = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_line = int(match.group(1)) - 1 if match is not None else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            new_line += 1
            if current_file in EXCLUDED_DIFF_FILES:
                continue
            for match in URL_RE.finditer(line[1:]):
                urls.append(
                    ChangedUrl(
                        url=normalize_url(match.group(0)),
                        source=f"{current_file}:{new_line}",
                    )
                )
            continue
        if line.startswith("-") and not line.startswith("---"):
            continue
        if not line.startswith("\\"):
            new_line += 1

    return urls


def unique_urls(urls: Iterable[ChangedUrl]) -> list[ChangedUrl]:
    """Return URL occurrences deduplicated by URL while preserving first source."""
    seen: set[str] = set()
    unique: list[ChangedUrl] = []
    for changed_url in urls:
        if changed_url.url in seen:
            continue
        seen.add(changed_url.url)
        unique.append(changed_url)
    return unique


def load_ignore_patterns(path: Path = IGNORE_PATH) -> list[re.Pattern[str]]:
    """Load lychee-compatible regex ignore patterns."""
    if not path.exists():
        return []
    patterns: list[re.Pattern[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        patterns.append(re.compile(pattern))
    return patterns


def is_ignored(url: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    """Return whether ``url`` matches a configured ignore pattern."""
    return any(pattern.search(url) for pattern in patterns)


def local_main_blob_path(url: str, root: Path = ROOT) -> Path | None:
    """Return the local path for a repo ``blob/main`` URL when it exists."""
    if not url.startswith(REPO_MAIN_BLOB_URL):
        return None
    relative = url.removeprefix(REPO_MAIN_BLOB_URL)
    if not relative or relative.startswith(("../", "/")):
        return None
    path = root / relative
    return path if path.exists() else None


def git_diff(root: Path, base: str, head: str) -> str:
    """Return a merge-base diff for ``base...head``."""
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--unified=0", f"{base}...{head}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def check_url(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> UrlCheckResult:
    """Check one URL, treating only 404/410 as hard failures."""
    last_error: str | None = None
    last_status: int | None = None

    for attempt in range(retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                status = response.status
            if status < 400:
                return UrlCheckResult(url=url, outcome="ok", status=status)
            last_status = status
            if status in HARD_FAIL_STATUSES:
                return UrlCheckResult(url=url, outcome="fail", status=status)
            if status >= TRANSIENT_STATUS_MIN and attempt < retries:
                continue
            return UrlCheckResult(url=url, outcome="warning", status=status)
        except HTTPError as error:
            last_status = error.code
            if error.code in HARD_FAIL_STATUSES:
                return UrlCheckResult(url=url, outcome="fail", status=error.code)
            if error.code >= TRANSIENT_STATUS_MIN and attempt < retries:
                continue
            return UrlCheckResult(url=url, outcome="warning", status=error.code)
        except (TimeoutError, socket.timeout, URLError) as error:
            last_error = str(error)
            if attempt < retries:
                continue

    return UrlCheckResult(
        url=url,
        outcome="warning",
        status=last_status,
        error=last_error,
    )


def check_changed_urls(
    changed_urls: Iterable[ChangedUrl],
    ignore_patterns: Iterable[re.Pattern[str]],
    checker: UrlChecker = check_url,
    root: Path = ROOT,
) -> list[UrlCheckResult]:
    """Check changed URLs and return checked, ignored, and warning results."""
    results: list[UrlCheckResult] = []
    for changed_url in unique_urls(changed_urls):
        if is_ignored(changed_url.url, ignore_patterns):
            results.append(UrlCheckResult(url=changed_url.url, outcome="ignored"))
            continue
        url_to_probe = probe_url(changed_url.url)
        if local_main_blob_path(url_to_probe, root=root) is not None:
            results.append(
                UrlCheckResult(url=changed_url.url, outcome="ok", status=200)
            )
            continue
        result = checker(url_to_probe)
        if result.url != changed_url.url:
            result = UrlCheckResult(
                url=changed_url.url,
                outcome=result.outcome,
                status=result.status,
                error=result.error,
            )
        results.append(result)
    return results


def report_results(results: Iterable[UrlCheckResult]) -> int:
    """Print URL check results and return a process exit status."""
    results = list(results)
    failures = [result for result in results if result.outcome == "fail"]
    warnings = [result for result in results if result.outcome == "warning"]
    checked = [result for result in results if result.outcome == "ok"]
    ignored = [result for result in results if result.outcome == "ignored"]

    print(
        "Changed URL check: "
        f"{len(checked)} ok, {len(ignored)} ignored, "
        f"{len(warnings)} warnings, {len(failures)} failures."
    )
    for label, bucket in (
        ("OK", checked),
        ("IGNORED", ignored),
        ("WARNING", warnings),
        ("FAIL", failures),
    ):
        for result in bucket:
            stream = sys.stderr if label in {"WARNING", "FAIL"} else sys.stdout
            print(f"{label}: {result.format()}", file=stream)

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Run the changed URL checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--diff-file", type=Path)
    parser.add_argument("--ignore-file", type=Path, default=IGNORE_PATH)
    args = parser.parse_args(argv)

    if args.diff_file is not None:
        diff_text = args.diff_file.read_text(encoding="utf-8")
    else:
        if args.base is None or args.head is None:
            parser.error("--base and --head are required unless --diff-file is used")
        diff_text = git_diff(args.root, args.base, args.head)

    changed_urls = extract_urls_from_added_lines(diff_text)
    if not changed_urls:
        print("Changed URL check: no added HTTP(S) URLs.")
        return 0
    return report_results(
        check_changed_urls(changed_urls, load_ignore_patterns(args.ignore_file))
    )


if __name__ == "__main__":
    raise SystemExit(main())
