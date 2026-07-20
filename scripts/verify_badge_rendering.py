"""Fetch README Shields badges and verify rendered SVGs.

This script intentionally performs network I/O and is not part of regular CI.
Use it during README badge review when shields.io rendering behavior matters.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from check_readme_badges import BADGE_DOCS, _iter_badges


USER_AGENT = "design-generators-badge-rendering-check/1.0"
ERROR_WORD_RE = re.compile(
    r"\b(error|invalid|not found|unknown|cannot)\b", re.IGNORECASE
)
SIMPLE_ICONS_SLUG_RE = re.compile(r"\|\s*`[^`]+`\s*\|\s*`([^`]+)`\s*\|")
SIMPLE_ICONS_SLUGS_URL = (
    "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/slugs.md"
)


def _fetch(url: str, timeout: float) -> tuple[int, str, bytes]:
    with tempfile.TemporaryDirectory() as tmpdir:
        headers_path = Path(tmpdir) / "headers.txt"
        body_path = Path(tmpdir) / "body.svg"
        result = subprocess.run(
            [
                "curl",
                "--location",
                "--max-time",
                str(timeout),
                "--user-agent",
                USER_AGENT,
                "--header",
                "Accept: image/svg+xml,text/plain;q=0.8,*/*;q=0.1",
                "--silent",
                "--show-error",
                "--dump-header",
                str(headers_path),
                "--output",
                str(body_path),
                url,
            ],
            check=False,
            capture_output=True,
            text=False,
        )
        body = body_path.read_bytes() if body_path.exists() else b""
        headers = (
            headers_path.read_text(encoding="iso-8859-1")
            if headers_path.exists()
            else ""
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise AssertionError(f"curl exited {result.returncode}: {stderr}")
        status = 0
        content_type = ""
        for block in [block for block in headers.split("\r\n\r\n") if block.strip()]:
            lines = block.splitlines()
            if lines and lines[0].startswith("HTTP/"):
                status = int(lines[0].split()[1])
                content_type = ""
            for line in lines[1:]:
                if line.lower().startswith("content-type:"):
                    content_type = line.split(":", 1)[1].strip()
        return status, content_type, body


def _fetch_simple_icon_slugs(timeout: float) -> set[str]:
    request = Request(
        SIMPLE_ICONS_SLUGS_URL,
        headers={"Accept": "text/plain", "User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return set(SIMPLE_ICONS_SLUG_RE.findall(text))


def _validate_svg(
    url: str, logo: str | None, status: int, content_type: str, body: bytes
) -> None:
    text = body.decode("utf-8", errors="replace")
    if status != 200:
        raise AssertionError(f"HTTP {status}")
    if "svg" not in content_type.lower():
        raise AssertionError(f"content-type is not SVG: {content_type!r}")
    if "<svg" not in text:
        raise AssertionError("response body does not contain <svg")
    if ERROR_WORD_RE.search(text):
        raise AssertionError("response SVG contains an error marker")
    if logo is not None and "<path" not in text:
        raise AssertionError(
            f"logo={logo!r} was requested but rendered SVG has no icon path"
        )
    if logo is None and "logo=" in url and "<path" not in text:
        raise AssertionError("logo query is present but no icon path rendered")


def _verify_one(url: str, logo: str | None, timeout: float) -> str | None:
    try:
        status, content_type, body = _fetch(url, timeout=timeout)
        _validate_svg(url, logo, status, content_type, body)
    except (AssertionError, HTTPError, URLError, TimeoutError, socket.timeout) as exc:
        return f"{url}\n  {type(exc).__name__}: {exc}"
    return None


def verify_rendering(timeout: float, jobs: int) -> int:
    badges = [badge for path in BADGE_DOCS for badge in _iter_badges(path)]
    unique: dict[str, str | None] = {}
    for badge in badges:
        unique.setdefault(badge.url, badge.logo)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_verify_one, url, logo, timeout): url
            for url, logo in sorted(unique.items())
        }
        for future in as_completed(futures):
            failure = future.result()
            if failure is not None:
                failures.append(failure)

    print(
        f"Checked {len(badges)} badge occurrences across "
        f"{sum(1 for path in BADGE_DOCS if _iter_badges(path))} documents with badges; "
        f"{len(unique)} unique shields.io URLs."
    )
    if failures:
        print("Badge rendering failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Badge rendering checks passed.")
    return 0


def verify_simple_icon_slugs(timeout: float) -> int:
    used = {
        badge.logo
        for path in BADGE_DOCS
        for badge in _iter_badges(path)
        if badge.logo is not None
    }
    try:
        official = _fetch_simple_icon_slugs(timeout)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Could not fetch Simple Icons slug list: {exc}", file=sys.stderr)
        return 1
    missing = sorted(used - official)
    print(
        f"Checked {len(used)} used Simple Icons slugs against {len(official)} official slugs."
    )
    if missing:
        print(f"Unknown Simple Icons slugs: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("Simple Icons slug checks passed.")
    return 0


def summarize() -> None:
    badges = [badge for path in BADGE_DOCS for badge in _iter_badges(path)]
    by_label = Counter(badge.label for badge in badges)
    by_logo = Counter(badge.logo or "(none)" for badge in badges)
    print(f"Badge occurrences: {len(badges)}")
    print(
        f"Documents with badges: {sum(1 for path in BADGE_DOCS if _iter_badges(path))}"
    )
    print("Labels:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")
    print("Logos:")
    for logo, count in sorted(by_logo.items()):
        print(f"  {logo}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--slugs-only",
        action="store_true",
        help="Only verify used logo slugs against the official Simple Icons slug list.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print badge counts before running checks.",
    )
    args = parser.parse_args()

    if args.summary:
        summarize()
    if args.slugs_only:
        return verify_simple_icon_slugs(args.timeout)
    return verify_rendering(args.timeout, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
