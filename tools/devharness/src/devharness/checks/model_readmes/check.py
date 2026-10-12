"""Run every model README policy check in scan order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devharness.baselines import (
    diff_entry_baseline,
    read_entry_baseline,
    write_entry_baseline,
)

from . import card, citation, install, parity, reproducing, root_readme
from .constants import (
    DOCS_MODEL_TABLE_HEADER,
    LIB_MEMBER_DIRS,
    LIB_READMES,
    MODEL_MEMBER_DIRS,
    MODEL_README_BASELINE_PATH,
    MODEL_READMES,
    MODEL_REPRODUCING,
    README_LINK_CONTRACTS,
    README_POLICY_DOCS,
    REPO_ROOT,
    ROOT_MODEL_TABLE_HEADER,
)


def current_model_readme_usage_violations() -> dict[str, str]:
    """Return repository-relative paths and diagnostics for missing output."""
    violations: dict[str, str] = {}
    for path in MODEL_READMES:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        violation = card.usage_output_violation(
            Path(relative_path), path.read_text(encoding="utf-8")
        )
        if violation:
            violations[relative_path] = violation

    return violations


def assert_model_readme_usage_baseline() -> None:
    """Require the model README usage-output baseline to match current violations."""
    current = current_model_readme_usage_violations()
    baseline = read_entry_baseline(MODEL_README_BASELINE_PATH)
    unexpected, stale = diff_entry_baseline(set(current), baseline)
    messages: list[str] = []
    if unexpected:
        messages.append("New model README usage-output violations:")
        messages.extend(f"  + {current[entry]}" for entry in unexpected)

    if stale:
        messages.append("Stale model README usage-output baseline entries:")
        messages.extend(f"  - {entry}" for entry in stale)

    if messages:
        raise AssertionError("\n".join(messages))


def check() -> None:
    """Run every model README policy check in scan order."""
    root_readme.assert_model_doc_sets()

    root_readme.assert_root_model_badge_count(
        REPO_ROOT / "README.md", len(MODEL_MEMBER_DIRS)
    )
    for path, header, link_pattern in (
        (
            REPO_ROOT / "README.md",
            ROOT_MODEL_TABLE_HEADER,
            "models/<slug>/README.md",
        ),
        (
            REPO_ROOT / "docs" / "index.md",
            DOCS_MODEL_TABLE_HEADER,
            "api/models/<slug>/",
        ),
    ):
        root_slugs = root_readme.root_model_slugs(path, header, link_pattern, REPO_ROOT)
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

    assert_model_readme_usage_baseline()

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


def main(argv: list[str] | None = None) -> int:
    """Print one diagnostic and return the checker status."""
    parser = argparse.ArgumentParser(prog="devharness check model-readmes")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the shrink-only baseline from current violations",
    )
    args = parser.parse_args([] if argv is None else argv)
    if args.write_baseline:
        write_entry_baseline(
            MODEL_README_BASELINE_PATH,
            current_model_readme_usage_violations(),
        )
        return 0

    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Model README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
