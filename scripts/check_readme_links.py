"""Enforce repository README link conventions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PAGES_BASE_URL = "https://creative-graphic-design.github.io/design-generators/"
REPO_URL = "https://github.com/creative-graphic-design/design-generators"
REPO_BLOB_URL = f"{REPO_URL}/blob/main/"
REPO_TREE_URL = f"{REPO_URL}/tree/main/"
MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]+\]\((?P<link>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)


@dataclass(frozen=True)
class LinkViolation:
    """README link convention violation."""

    path: Path
    line: int
    link: str
    reason: str

    def format(self, root: Path) -> str:
        """Return a stable human-readable violation line."""
        rel_path = self.path.relative_to(root).as_posix()
        return f"{rel_path}:{self.line}: {self.reason}: {self.link}"


def readme_paths(root: Path = ROOT) -> list[Path]:
    """Return first-party README, guide, and skill files checked by this script."""
    paths = [root / "README.md"]
    paths.extend(sorted((root / "lib").glob("*/README.md")))
    paths.extend(sorted((root / "models").glob("*/README.md")))
    paths.extend(sorted((root / "models").glob("*/TRAINING.md")))
    paths.extend(sorted((root / "models").glob("*/REPRODUCING.md")))
    paths.extend(sorted((root / ".agents" / "skills").rglob("SKILL.md")))
    return [path for path in paths if path.is_file()]


def strip_link_suffix(link: str) -> str:
    """Drop fragment and query suffixes from a Markdown link target."""
    return re.split(r"[#?]", link, maxsplit=1)[0]


def is_external_or_anchor(link: str) -> bool:
    """Return whether a link target is external or page-local."""
    return link.startswith(("#", "mailto:", "http://", "https://"))


def _is_package_document(path: Path, root: Path) -> bool:
    """Return whether a path is a workspace-member document."""
    return path.relative_to(root).parts[0] in {"lib", "models"}


def violation_for_link(
    path: Path,
    line: int,
    link: str,
    root: Path = ROOT,
) -> LinkViolation | None:
    """Return a README link convention violation, if any."""
    if link.startswith(PAGES_BASE_URL) and strip_link_suffix(link).endswith(".md"):
        return LinkViolation(
            path,
            line,
            link,
            "repository README links must not point at unpublished Pages markdown URLs",
        )

    if is_external_or_anchor(link):
        return None

    if _is_package_document(path, root):
        return LinkViolation(
            path,
            line,
            link,
            f"package document links must use absolute {REPO_BLOB_URL} or {REPO_TREE_URL} URLs",
        )

    target = strip_link_suffix(link).removeprefix("./")
    if target.startswith("../"):
        return LinkViolation(
            path,
            line,
            link,
            "Markdown links must use repo-root-relative paths",
        )

    resolved = (root / target).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return LinkViolation(
            path,
            line,
            link,
            "repo-root-relative Markdown links must stay within the repository",
        )

    if not resolved.exists():
        return LinkViolation(
            path,
            line,
            link,
            "repo-root-relative Markdown links must resolve to an existing path",
        )

    return None


def violations_for_readme(path: Path, root: Path = ROOT) -> list[LinkViolation]:
    """Return link convention violations for one README."""
    text = path.read_text(encoding="utf-8")
    violations: list[LinkViolation] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        link = match.group("link")
        line = text.count("\n", 0, match.start()) + 1
        if violation := violation_for_link(path, line, link, root):
            violations.append(violation)
    return violations


def current_violations(root: Path = ROOT) -> list[LinkViolation]:
    """Return README link convention violations under the repository root."""
    violations: list[LinkViolation] = []
    for path in readme_paths(root):
        violations.extend(violations_for_readme(path, root))
    return violations


def check_readme_links(root: Path = ROOT) -> int:
    """Check first-party README link conventions."""
    violations = current_violations(root)
    if not violations:
        return 0
    print("README link convention violations:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation.format(root)}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the README link convention checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    return check_readme_links(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
