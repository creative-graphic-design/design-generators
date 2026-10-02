"""Run every model README policy check in scan order."""

from __future__ import annotations

import sys

from . import card, citation, install, parity, reproducing, root_readme
from .constants import (
    LIB_MEMBER_DIRS,
    LIB_READMES,
    MODEL_MEMBER_DIRS,
    MODEL_READMES,
    MODEL_REPRODUCING,
    README_LINK_CONTRACTS,
    README_POLICY_DOCS,
    REPO_ROOT,
)


def check() -> None:
    """Run every model README policy check in scan order."""
    root_readme.assert_model_doc_sets()

    root_readme.assert_root_model_badge_count(
        REPO_ROOT / "README.md", len(MODEL_MEMBER_DIRS)
    )
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "index.md"):
        root_slugs = root_readme.root_model_slugs(path)
        root_readme.assert_root_models_table_matches_members(root_slugs, path)

    root_readme.assert_root_libraries_table_matches_members(REPO_ROOT / "README.md")

    for path in MODEL_READMES:
        text = path.read_text(encoding="utf-8")
        card.assert_frontmatter(path, text)
        card.assert_expected_frontmatter(path, text)
        card.assert_runtime_contract(path, text)
        card.parse_model_type(path, text)
        card.assert_heading_order(path, text)
        install.assert_model_pip_install_snippet(path, text)
        card.assert_model_summary_subject(path, text)

        card.assert_expected_repository_links(path, text)
        card.assert_prompt_only_readme(path, text)
        card.assert_unpublished_hub_get_started_note(path, text)
        card.assert_code_fences_tagged(path, text)
        parity.assert_parity_table(path, text)
        citation.assert_citation_bibtex(path, text)
        parity.assert_vendor_parity_badge(path, text)
        reproducing.assert_readme_reproducibility_link(path, text)
        card.assert_banned_patterns(path, text)

    for path in MODEL_REPRODUCING:
        text = path.read_text(encoding="utf-8")
        card.assert_code_fences_tagged(path, text)
        reproducing.assert_reproducing_commands(path, text)
        card.assert_banned_patterns(path, text)

    for path in README_LINK_CONTRACTS:
        root_readme.assert_linked_first_reference_policy(path)
        root_readme.assert_library_name_style(path)

    for member_dir in LIB_MEMBER_DIRS:
        path = member_dir / "README.md"
        install.assert_library_pip_install_snippet(
            path,
            path.read_text(encoding="utf-8"),
        )

    for path in README_POLICY_DOCS:
        text = path.read_text(encoding="utf-8")
        card.assert_code_fences_tagged(path, text)
        card.assert_banned_patterns(path, text)

    for path in LIB_READMES:
        install.assert_lib_readme_install_contract(
            path,
            path.read_text(encoding="utf-8"),
        )


def main() -> int:
    """Print one diagnostic and return the checker status."""
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Model README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
