"""Validate README badge placement, grammar, and order."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
BADGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|\[!\[[^\]]*\]\(([^)]+)\)\]\([^)]+\)")

ROOT_ORDER = ["CI", "docs", "license", "python", "uv", "models"]
LAYGEN_ORDER = ["package", "license", "python", "core", "extras", "docs"]
POSGEN_ORDER = ["package", "license", "python", "runtime", "status", "docs"]
MODEL_ORDER = [
    ("paper", "arXiv", "OpenReview", "DOI"),
    ("venue",),
    ("license",),
    ("base",),
    ("dataset", "task"),
    ("vendor--parity",),
    ("hub",),
]


def _badge_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for match in BADGE_RE.finditer(path.read_text(encoding="utf-8")):
        url = match.group(1) or match.group(2)
        if "actions/workflows/" in url and url.endswith("/badge.svg"):
            labels.append("CI")
            continue
        parsed = urlparse(url)
        if parsed.netloc != "img.shields.io" or parsed.path != "/static/v1":
            raise AssertionError(f"{path}: non-static shields badge URL: {url}")
        query = parse_qs(parsed.query)
        if query.get("style") != ["flat-square"]:
            raise AssertionError(f"{path}: badge must use style=flat-square: {url}")
        if "label" not in query or "message" not in query or "color" not in query:
            raise AssertionError(f"{path}: badge missing label/message/color: {url}")
        labels.append(query["label"][0])
    return labels


def _assert_prefix(path: Path, expected: list[str]) -> None:
    labels = _badge_labels(path)
    if labels[: len(expected)] != expected:
        raise AssertionError(
            f"{path}: badge order {labels[: len(expected)]} != {expected}"
        )


def _assert_model_order(path: Path) -> None:
    labels = _badge_labels(path)
    if not labels:
        raise AssertionError(f"{path}: no badges found")
    cursor = 0
    for aliases in MODEL_ORDER:
        positions = [
            i
            for i, label in enumerate(labels[cursor:], start=cursor)
            if label in aliases
        ]
        if not positions:
            raise AssertionError(f"{path}: missing badge for {aliases}")
        cursor = positions[-1] + 1 if aliases[0] == "dataset" else positions[0] + 1

    text = path.read_text(encoding="utf-8")
    h1 = re.search(r"^# .+$", text, re.MULTILINE)
    if h1 is None:
        raise AssertionError(f"{path}: missing H1")
    after_h1 = text[h1.end() :].lstrip()
    if not after_h1.startswith("![") and not after_h1.startswith("[!["):
        raise AssertionError(f"{path}: badges must be directly below the H1")


def check() -> None:
    _assert_prefix(REPO_ROOT / "README.md", ROOT_ORDER)
    _assert_prefix(REPO_ROOT / "lib/laygen/README.md", LAYGEN_ORDER)
    _assert_prefix(REPO_ROOT / "lib/posgen/README.md", POSGEN_ORDER)
    for path in sorted((REPO_ROOT / "models").glob("*/README.md")):
        _assert_model_order(path)


def main() -> int:
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("README badge checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
