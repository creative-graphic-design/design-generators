"""README link convention checker tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_README_LINKS = REPO_ROOT / "scripts" / "check_readme_links.py"


def load_check_readme_links() -> ModuleType:
    """Load the README link checker script as a module."""
    spec = importlib.util.spec_from_file_location(
        "check_readme_links", CHECK_README_LINKS
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_readme_links_accepts_repo_root_relative_docs_link(
    tmp_path: Path,
) -> None:
    check_readme_links = load_check_readme_links()
    (tmp_path / "README.md").write_text(
        "See the [training reproduction protocol](docs/training-reproduction.md).\n",
        encoding="utf-8",
    )

    assert check_readme_links.check_readme_links(tmp_path) == 0


def test_check_readme_links_rejects_parent_relative_docs_link(tmp_path: Path) -> None:
    check_readme_links = load_check_readme_links()
    readme = tmp_path / "lib" / "traingen-parity" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        "See the [training reproduction protocol](../../docs/training-reproduction.md).\n",
        encoding="utf-8",
    )

    violations = check_readme_links.current_violations(tmp_path)

    assert check_readme_links.check_readme_links(tmp_path) == 1
    assert len(violations) == 1
    assert violations[0].link == "../../docs/training-reproduction.md"
    assert "../ relative escapes" in violations[0].reason


def test_check_readme_links_rejects_pages_markdown_url(tmp_path: Path) -> None:
    check_readme_links = load_check_readme_links()
    link = check_readme_links.PAGES_BASE_URL + "docs/training-reproduction.md"
    (tmp_path / "README.md").write_text(
        f"See [training]({link}).\n",
        encoding="utf-8",
    )

    violations = check_readme_links.current_violations(tmp_path)

    assert check_readme_links.check_readme_links(tmp_path) == 1
    assert len(violations) == 1
    assert "unpublished Pages markdown URLs" in violations[0].reason


def test_check_readme_links_accepts_current_first_party_readmes() -> None:
    check_readme_links = load_check_readme_links()

    assert check_readme_links.check_readme_links(REPO_ROOT) == 0
