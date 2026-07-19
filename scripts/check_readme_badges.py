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

VERIFIED_SIMPLE_ICON_SLUGS = {
    "apache",
    "arxiv",
    "creativecommons",
    "doi",
    "githubactions",
    "huggingface",
    "opensourceinitiative",
    "pydantic",
    "python",
    "readthedocs",
    "uv",
}


def _allowed_logos(label: str, message: str | None) -> set[str | None]:
    if label == "CI":
        return {"githubactions"}
    if label == "docs":
        return {"readthedocs"}
    if label == "license":
        if message == "Apache--2.0":
            return {"apache"}
        if message and message.startswith("CC--"):
            return {"creativecommons"}
        if message == "review-needed":
            return {None}
        return {"opensourceinitiative"}
    if label == "python":
        return {"python"}
    if label == "uv":
        return {"uv"}
    if label in {"models", "package", "core", "extras", "runtime", "status"}:
        return {None}
    if label == "arXiv":
        return {"arxiv"}
    if label == "DOI":
        return {"doi"}
    if label in {"paper", "OpenReview", "venue"}:
        return {None}
    if label == "base":
        return {"pydantic"} if message == "pydantic-ai" else {"huggingface"}
    if label in {"dataset", "hub"}:
        return {"huggingface"}
    if label == "vendor--parity":
        return {None}
    raise AssertionError(f"no badge logo rule for label={label!r} message={message!r}")


def _badge_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for match in BADGE_RE.finditer(path.read_text(encoding="utf-8")):
        url = match.group(1) or match.group(2)
        parsed = urlparse(url)
        if parsed.netloc != "img.shields.io":
            raise AssertionError(f"{path}: non-static shields badge URL: {url}")
        query = parse_qs(parsed.query)
        if query.get("style") != ["flat-square"]:
            raise AssertionError(f"{path}: badge must use style=flat-square: {url}")
        if parsed.path == "/static/v1":
            if "label" not in query or "message" not in query or "color" not in query:
                raise AssertionError(
                    f"{path}: badge missing label/message/color: {url}"
                )
        elif not parsed.path.startswith("/github/actions/workflow/status/"):
            raise AssertionError(f"{path}: unsupported shields badge path: {url}")
        if "label" not in query:
            raise AssertionError(f"{path}: badge missing label: {url}")
        label = query["label"][0]
        message = query.get("message", [None])[0]
        logo = query.get("logo", [None])[0]
        allowed_logos = _allowed_logos(label, message)
        if logo not in allowed_logos:
            raise AssertionError(
                f"{path}: badge {label!r} logo {logo!r} not in {allowed_logos!r}: {url}"
            )
        if logo is None:
            if "logoColor" in query:
                raise AssertionError(f"{path}: logoColor requires logo: {url}")
        elif logo not in VERIFIED_SIMPLE_ICON_SLUGS:
            raise AssertionError(f"{path}: unverified Simple Icons slug {logo!r}")
        elif query.get("logoColor") != ["white"]:
            raise AssertionError(f"{path}: badge logoColor must be white: {url}")
        labels.append(label)
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
            if aliases == ("venue",):
                continue
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
