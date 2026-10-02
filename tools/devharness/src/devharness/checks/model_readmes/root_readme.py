"""Root README model and library table contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .constants import (
    EXPECTED_MODEL_NAMES,
    LINK_REQUIRED_DATASET_IDS,
    REPO_ROOT,
    ROOT_LIBRARY_BADGE_COLORS,
    ROOT_REPO_BLOB_URL,
)
from .card import (
    badge_messages,
    inline_code_span_at,
    in_any_span,
    library_member_slugs,
    markdown_link_spans,
    model_member_slugs,
    parse_model_type,
    semantic_badge_label,
    without_frontmatter_and_code,
    without_badges,
)


def normalize_root_repo_link(link: str) -> str:
    """Normalize a root README repository link to a repository-relative path."""
    return link.removeprefix(ROOT_REPO_BLOB_URL)


def _assert_root_model_type_cells(
    path: Path,
    repository_root: Path,
    slug: str,
    task_cell: str,
    content_cell: str,
) -> None:
    model_readme = repository_root / "models" / slug / "README.md"
    expected_content, expected_task, _ = parse_model_type(
        model_readme, model_readme.read_text(encoding="utf-8")
    )

    task_badges = _static_badge_messages(task_cell, "task")
    if len(task_badges) != 1:
        raise AssertionError(
            f"{path}: Models table package {slug} Task cell must contain exactly one task badge"
        )

    if task_badges[0] != expected_task:
        raise AssertionError(
            f"{path}: Models table package {slug} Task cell {task_badges[0]!r} "
            f"!= Model type task {expected_task!r}"
        )

    content_badges = _static_badge_messages(content_cell, "content")
    if len(content_badges) != 1:
        raise AssertionError(
            f"{path}: Models table package {slug} Content cell must contain exactly one content badge"
        )

    if content_badges[0] != expected_content:
        raise AssertionError(
            f"{path}: Models table package {slug} Content cell {content_badges[0]!r} "
            f"!= Model type content {expected_content!r}"
        )


def _root_packages_table_lines(text: str) -> list[str]:
    marker = "## Models\n\n"
    start = text.index(marker) + len(marker)
    lines = text[start:].splitlines()
    table_start = next(
        (index for index, line in enumerate(lines) if line.startswith("| ")),
        None,
    )
    if table_start is None:
        raise AssertionError("root README missing Models table")

    table_lines: list[str] = []
    for line in lines[table_start:]:
        if not line.startswith("|"):
            break

        table_lines.append(line)

    return table_lines


def _root_libraries_table_lines(text: str) -> list[str]:
    marker = "## Libraries\n\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n\n", start)
    return text[start:end].splitlines()


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise AssertionError(f"malformed markdown table row: {line}")

    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_markdown_table_separator(line: str, width: int) -> bool:
    cells = _split_markdown_table_row(line)
    if len(cells) != width:
        return False

    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _badge_message(alt: str, badge_url: str, label: str) -> str | None:
    query = parse_qs(urlparse(badge_url).query)
    query_label = query.get("label", [None])[0]
    if (
        query_label is None
        or semantic_badge_label(alt, query_label) != label
        or "message" not in query
    ):
        return None

    return unquote(query["message"][0])


def _linked_static_badge(cell: str, label: str) -> tuple[str, str] | None:
    badges = _linked_static_badges(cell, label)
    match = _LINKED_BADGE_RE.search(cell)
    if len(badges) != 1 or match is None or cell != match.group(0):
        return None

    return badges[0]


_LINKED_BADGE_RE = re.compile(
    r"\[!\[([^\]]*)\]\((https://img\.shields\.io/static/v1\?[^)]*)\)\]\(([^)]+)\)"
)


def _linked_static_badges(cell: str, label: str) -> list[tuple[str, str]]:
    badges: list[tuple[str, str]] = []
    for match in _LINKED_BADGE_RE.finditer(cell):
        message = _badge_message(match.group(1), match.group(2), label)
        if message is not None:
            badges.append((message, match.group(3)))

    return badges


def _static_badge_messages(cell: str, label: str) -> list[str]:
    messages: list[str] = []
    for match in re.finditer(
        r"!\[([^\]]*)\]\((https://img\.shields\.io/static/v1\?[^)]*)\)", cell
    ):
        message = _badge_message(match.group(1), match.group(2), label)
        if message is not None:
            messages.append(message)

    return messages


def _static_badge_colors(cell: str, label: str) -> list[str]:
    colors: list[str] = []
    for match in re.finditer(
        r"!\[([^\]]*)\]\((https://img\.shields\.io/static/v1\?[^)]*)\)", cell
    ):
        if _badge_message(match.group(1), match.group(2), label) is None:
            continue

        query = parse_qs(urlparse(match.group(2)).query)
        if "color" not in query:
            raise AssertionError(f"badge missing color: {match.group(0)}")

        colors.append(query["color"][0])

    return colors


def _model_training_slugs() -> set[str]:
    return {
        path.parent.name
        for path in sorted((REPO_ROOT / "models").glob("*/TRAINING.md"))
    }


def _assert_root_reproduction_cells(
    path: Path, slug: str, checkpoint_cell: str, training_cell: str
) -> None:
    expected_checkpoint_link = f"models/{slug}/REPRODUCING.md"
    checkpoint_badges = [
        (message, normalize_root_repo_link(link))
        for message, link in _linked_static_badges(checkpoint_cell, "checkpoint")
    ]
    if checkpoint_badges != [("ckpt", expected_checkpoint_link)]:
        raise AssertionError(
            f"{path}: package {slug} must use one linked checkpoint reproduction badge"
        )

    training_badges = [
        (message, normalize_root_repo_link(link))
        for message, link in _linked_static_badges(training_cell, "training")
    ]
    training_messages = _static_badge_messages(training_cell, "training")
    expected_link = f"models/{slug}/TRAINING.md"
    if slug in _model_training_slugs():
        if training_badges != [("train", expected_link)]:
            raise AssertionError(
                f"{path}: package {slug} must use one linked training reproduction badge"
            )

        return

    if training_badges:
        raise AssertionError(
            f"{path}: package {slug} without TRAINING.md must not link training badge"
        )

    if training_messages != ["n/a"]:
        raise AssertionError(
            f"{path}: package {slug} without TRAINING.md must use training n/a badge"
        )


def assert_root_model_badge_count(path: Path, expected_count: int) -> None:
    """Require the root README model-count badge to match workspace members."""
    text = path.read_text(encoding="utf-8")
    messages = badge_messages(text, "models")
    if messages != [str(expected_count)]:
        raise AssertionError(
            f"{path}: models badge {messages} != workspace model member count {expected_count}"
        )


def root_model_slugs(
    path: Path,
    expected_header: Sequence[str],
    model_link_pattern: str,
    repository_root: Path,
) -> set[str]:
    """Validate the root model table and return its model slugs."""
    text = path.read_text(encoding="utf-8")
    table_lines = _root_packages_table_lines(text)
    if _split_markdown_table_row(
        table_lines[0]
    ) != expected_header or not _is_markdown_table_separator(
        table_lines[1], len(expected_header)
    ):
        raise AssertionError(
            f"{path}: Models table must use {', '.join(expected_header)}"
        )

    slugs: set[str] = set()
    for line in table_lines[2:]:
        cells = _split_markdown_table_row(line)
        if len(cells) != len(expected_header):
            raise AssertionError(f"{path}: malformed Models table row: {line}")

        method_cell, task_cell, content_cell = cells[:3]
        _, _, _, venue_cell, checkpoint_cell, training_cell = cells[:6]

        model_link = re.fullmatch(r"\[`([^`\]]+)`\]\(([^)]+)\)", method_cell)
        if model_link is None:
            raise AssertionError(
                f"{path}: Model cell must be a `Model` markdown link: {line}"
            )

        model_name = model_link.group(1)
        normalized_model_link = normalize_root_repo_link(model_link.group(2))
        link_pattern = re.escape(model_link_pattern).replace(
            re.escape("<slug>"), r"([^/)]+)"
        )
        slug_match = re.fullmatch(link_pattern, normalized_model_link)
        if slug_match is None:
            raise AssertionError(
                f"{path}: Model cell must link {model_link_pattern}: {line}"
            )

        slug = slug_match.group(1)
        expected_name = EXPECTED_MODEL_NAMES.get(slug)
        if expected_name is not None and model_name != expected_name:
            raise AssertionError(
                f"{path}: model link text {model_name!r} != {expected_name!r}"
            )

        _assert_root_model_type_cells(
            path,
            repository_root,
            slug,
            task_cell,
            content_cell,
        )

        if len(_static_badge_messages(venue_cell, "venue")) != 1:
            raise AssertionError(
                f"{path}: Models table Venue cell must contain exactly one venue badge: {line}"
            )

        if (
            "documented" in checkpoint_cell.lower()
            or "documented" in training_cell.lower()
        ):
            raise AssertionError(
                f"{path}: Models table reproduction cells must not use status wording"
            )

        _assert_root_reproduction_cells(path, slug, checkpoint_cell, training_cell)
        slugs.add(slug)

    return slugs


def assert_root_libraries_table_matches_members(path: Path) -> None:
    """Require the root library table to match workspace library members."""
    text = path.read_text(encoding="utf-8")
    table_lines = _root_libraries_table_lines(text)
    if _split_markdown_table_row(table_lines[0]) != [
        "Library",
        "Description",
    ] or not _is_markdown_table_separator(table_lines[1], 2):
        raise AssertionError(
            f"{path}: Libraries table must use Library and Description"
        )

    root_slugs: set[str] = set()
    for line in table_lines[2:]:
        cells = _split_markdown_table_row(line)
        if len(cells) != 2:
            raise AssertionError(f"{path}: malformed Libraries table row: {line}")

        library_cell, description_cell = cells
        library_badge = _linked_static_badge(library_cell, "library")
        if library_badge is None:
            raise AssertionError(
                f"{path}: Library cell must be a linked library badge: {line}"
            )

        label, library_link = library_badge
        normalized_library_link = normalize_root_repo_link(library_link)
        slug_match = re.fullmatch(r"lib/([^/)]+)/README\.md", normalized_library_link)
        if slug_match is None:
            raise AssertionError(
                f"{path}: Library cell must link lib/<slug>/README.md: {line}"
            )

        slug = slug_match.group(1)
        if label != slug:
            raise AssertionError(
                f"{path}: Library badge message {label!r} must match {slug!r}"
            )

        expected_color = ROOT_LIBRARY_BADGE_COLORS.get(slug)
        if expected_color is None:
            raise AssertionError(f"{path}: no library badge color for {slug!r}")

        library_colors = _static_badge_colors(library_cell, "library")
        if library_colors != [expected_color]:
            raise AssertionError(
                f"{path}: Library {slug} badge color {library_colors} != {expected_color!r}"
            )

        if not description_cell:
            raise AssertionError(f"{path}: Library {slug} must have a description")

        root_slugs.add(slug)

    member_slugs = library_member_slugs()
    missing = sorted(member_slugs - root_slugs)
    extra = sorted(root_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"root README Libraries table mismatch: missing={missing}, extra={extra}"
        )


def assert_model_doc_sets() -> None:
    """Require every model workspace member to have its README documents."""
    member_slugs = model_member_slugs()
    readme_slugs = {
        path.parent.name for path in sorted((REPO_ROOT / "models").glob("*/README.md"))
    }
    reproducing_slugs = {
        path.parent.name
        for path in sorted((REPO_ROOT / "models").glob("*/REPRODUCING.md"))
    }
    for label, actual in (
        ("README.md", readme_slugs),
        ("REPRODUCING.md", reproducing_slugs),
    ):
        missing = sorted(member_slugs - actual)
        extra = sorted(actual - member_slugs)
        if missing or extra:
            raise AssertionError(
                f"model {label} set mismatch: missing={missing}, extra={extra}"
            )


def assert_root_models_table_matches_members(root_slugs: set[str], path: Path) -> None:
    """Require root README model rows to match workspace model members."""
    member_slugs = model_member_slugs()
    missing = sorted(member_slugs - root_slugs)
    extra = sorted(root_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"{path}: Models table mismatch: missing={missing}, extra={extra}"
        )


def assert_linked_first_reference_policy(path: Path) -> None:
    """Require first dataset and arXiv references to be Markdown links."""
    text = without_frontmatter_and_code(path.read_text(encoding="utf-8"))
    spans = markdown_link_spans(text)
    for dataset_id in LINK_REQUIRED_DATASET_IDS:
        for match in re.finditer(re.escape(dataset_id), text):
            if not in_any_span(match.start(), spans):
                raise AssertionError(f"{path}: dataset id must be linked: {dataset_id}")

    for match in re.finditer(r"\barXiv\s+\d{4}\.\d{4,5}\b", text):
        if not in_any_span(match.start(), spans):
            raise AssertionError(f"{path}: arXiv id must be linked: {match.group(0)}")

    for match in re.finditer(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}", text):
        if not in_any_span(match.start(), spans):
            raise AssertionError(f"{path}: arXiv URL must be in a markdown link")


def assert_library_name_style(path: Path) -> None:
    """Require reader-facing library names to use their annotated code forms."""
    text = without_badges(
        without_frontmatter_and_code(path.read_text(encoding="utf-8"))
    )
    banned_names = {
        "Transformers": "`🤗transformers`",
        "Diffusers": "`🧨diffusers`",
        "Pydantic AI": "`🤖pydantic-ai`",
    }
    for name, replacement in banned_names.items():
        match = re.search(rf"(?<![`/\w]){re.escape(name)}(?![`/\w])", text)
        if match:
            raise AssertionError(
                f"{path}: use {replacement} instead of prose library name {name!r}"
            )

    huggingface_mentions = re.findall(r"`🤗transformers`", text)
    if text.count("🤗") != len(huggingface_mentions):
        raise AssertionError(f"{path}: 🤗 must annotate a transformers library mention")

    diffusers_mentions = re.findall(r"`🧨diffusers`", text)
    if text.count("🧨") != len(diffusers_mentions):
        raise AssertionError(f"{path}: 🧨 must annotate a diffusers library mention")

    pydantic_ai_mentions = re.findall(r"`🤖pydantic-ai`", text)
    if text.count("🤖") != len(pydantic_ai_mentions):
        raise AssertionError(f"{path}: 🤖 must annotate a pydantic-ai library mention")

    if re.search(r"🤗\s+(?:\[`pydantic-ai`\]\([^)]*\)|`pydantic-ai`)", text):
        raise AssertionError(f"{path}: pydantic-ai mentions must not use 🤗")

    if re.search(r"🤗\s+pydantic-ai", text):
        raise AssertionError(f"{path}: pydantic-ai mentions must not use 🤗")

    required_code_spans = {
        "transformers": "`🤗transformers`",
        "diffusers": "`🧨diffusers`",
        "pydantic-ai": "`🤖pydantic-ai`",
    }
    for library in ("transformers", "diffusers", "pydantic-ai"):
        for match in re.finditer(rf"(?<![`/\w=-]){re.escape(library)}(?![`/\w])", text):
            if inline_code_span_at(text, match.start()) == required_code_spans[library]:
                continue

            raise AssertionError(
                f"{path}: use code-form {required_code_spans[library]} for library names"
            )
