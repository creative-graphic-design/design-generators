"""Run model README policy in its legacy order."""

from __future__ import annotations

import sys

from .citation import (
    _assert_banned_patterns,
    _assert_citation_bibtex,
    _assert_readme_reproducibility_link,
    _assert_reproducing_commands,
    _assert_vendor_parity_badge,
)
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
from .metadata import (
    _assert_code_fences_tagged,
    _assert_expected_frontmatter,
    _assert_expected_repository_links,
    _assert_frontmatter,
    _assert_heading_order,
    _assert_lib_readme_install_contract,
    _assert_model_summary_subject,
    _assert_parity_table,
    _assert_prompt_only_readme,
    _assert_runtime_contract,
    _assert_unpublished_hub_get_started_note,
)
from .repository import (
    _assert_library_pip_install_snippet,
    _assert_model_pip_install_snippet,
)
from .root_readme import (
    _assert_library_name_style,
    _assert_linked_first_reference_policy,
    _assert_model_doc_sets,
    _assert_root_libraries_table_matches_members,
    _assert_root_model_badge_count,
    _assert_root_models_table_matches_members,
    _root_model_slugs,
)


def check() -> None:
    """Run every model README policy check in legacy order."""
    _assert_model_doc_sets()

    root_slugs = _root_model_slugs(REPO_ROOT / "README.md")
    _assert_root_model_badge_count(REPO_ROOT / "README.md", len(MODEL_MEMBER_DIRS))
    _assert_root_models_table_matches_members(root_slugs)
    _assert_root_libraries_table_matches_members(REPO_ROOT / "README.md")

    for path in MODEL_READMES:
        text = path.read_text(encoding="utf-8")
        _assert_frontmatter(path, text)
        _assert_expected_frontmatter(path, text)
        _assert_runtime_contract(path, text)
        _assert_heading_order(path, text)
        _assert_model_pip_install_snippet(path, text)
        _assert_model_summary_subject(path, text)

        _assert_expected_repository_links(path, text)
        _assert_prompt_only_readme(path, text)
        _assert_unpublished_hub_get_started_note(path, text)
        _assert_code_fences_tagged(path, text)
        _assert_parity_table(path, text)
        _assert_citation_bibtex(path, text)
        _assert_vendor_parity_badge(path, text)
        _assert_readme_reproducibility_link(path, text)
        _assert_banned_patterns(path, text)

    for path in MODEL_REPRODUCING:
        text = path.read_text(encoding="utf-8")
        _assert_code_fences_tagged(path, text)
        _assert_reproducing_commands(path, text)
        _assert_banned_patterns(path, text)

    for path in README_LINK_CONTRACTS:
        _assert_linked_first_reference_policy(path)
        _assert_library_name_style(path)

    for member_dir in LIB_MEMBER_DIRS:
        path = member_dir / "README.md"
        _assert_library_pip_install_snippet(
            path,
            path.read_text(encoding="utf-8"),
        )

    for path in README_POLICY_DOCS:
        text = path.read_text(encoding="utf-8")
        _assert_code_fences_tagged(path, text)
        _assert_banned_patterns(path, text)

    for path in LIB_READMES:
        _assert_lib_readme_install_contract(path, path.read_text(encoding="utf-8"))


def main() -> int:
    """Print one diagnostic and return the legacy checker status."""
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Model README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
