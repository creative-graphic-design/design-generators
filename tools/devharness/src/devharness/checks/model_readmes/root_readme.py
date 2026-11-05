"""Repository model and library table contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlparse, urlunparse

from .constants import (
    DOCS_LIBRARY_TABLE_HEADER,
    EXPECTED_MODEL_NAMES,
    IMAGE_GENERATION_CATEGORY,
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
    ModelType,
    semantic_badge_label,
    without_frontmatter_and_code,
    without_badges,
)


def normalize_root_repo_link(link: str) -> str:
    """Normalize a root README repository link to a repository-relative path."""
    return link.removeprefix(ROOT_REPO_BLOB_URL)


def _model_catalog_badge_values(model_type: ModelType) -> tuple[str, str]:
    """Return the Task and Content badge messages for a model classification."""
    category, task, conditioning = model_type
    content = (
        " + ".join(conditioning) if category == IMAGE_GENERATION_CATEGORY else category
    )

    return task, content


def _assert_root_model_type_cells(
    path: Path,
    repository_root: Path,
    slug: str,
    task_cell: str,
    content_cell: str,
) -> None:
    model_readme = repository_root / "models" / slug / "README.md"
    if not model_readme.is_file():
        raise AssertionError(
            f"{path}: Models table package {slug} has no models/{slug}/README.md"
        )

    model_type = parse_model_type(
        model_readme, model_readme.read_text(encoding="utf-8")
    )
    category, _, _ = model_type
    expected_task, expected_content = _model_catalog_badge_values(model_type)

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
        content_label = (
            "conditioning" if category == IMAGE_GENERATION_CATEGORY else "content"
        )
        raise AssertionError(
            f"{path}: Models table package {slug} Content cell {content_badges[0]!r} "
            f"!= Model type {content_label} {expected_content!r}"
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


def _docs_libraries_table_lines(text: str) -> list[str]:
    marker = "## Libraries\n\n"
    start = text.index(marker) + len(marker)
    lines = text[start:].splitlines()
    table_lines: list[str] = []
    for line in lines:
        if not line.startswith("|"):
            break

        table_lines.append(line)

    return table_lines


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
_STATIC_BADGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<url>https://img\.shields\.io/static/v1\?[^)]+)\)"
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


def update_root_model_badge_count(path: Path, expected_count: int) -> bool:
    """Rewrite the root README model-count badge and report whether it changed."""
    text = path.read_text(encoding="utf-8")
    matching_badges = 0

    def replace_badge(match: re.Match[str]) -> str:
        nonlocal matching_badges

        alt = match.group("alt")
        url = match.group("url")
        parsed_url = urlparse(url)
        query = parse_qsl(parsed_url.query, keep_blank_values=True)
        labels = [value for key, value in query if key == "label"]
        if len(labels) != 1 or labels[0] != "models":
            return match.group(0)

        if semantic_badge_label(alt, labels[0]) != "models":
            return match.group(0)

        matching_badges += 1
        message_indexes = [
            index for index, (key, _) in enumerate(query) if key == "message"
        ]
        if len(message_indexes) != 1:
            raise AssertionError(f"{path}: models badge must have one message value")

        message_index = message_indexes[0]
        if query[message_index][1] == str(expected_count):
            return match.group(0)

        query[message_index] = ("message", str(expected_count))
        updated_url = urlunparse(parsed_url._replace(query=urlencode(query)))
        return f"![{alt}]({updated_url})"

    updated_text = _STATIC_BADGE_RE.sub(replace_badge, text)
    if matching_badges != 1:
        raise AssertionError(
            f"{path}: expected one models badge, found {matching_badges}"
        )

    if updated_text == text:
        return False

    path.write_text(updated_text, encoding="utf-8")
    return True


def root_model_slugs(
    path: Path,
    expected_header: Sequence[str],
    model_link_pattern: str,
    repository_root: Path,
) -> list[str]:
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

    slugs: list[str] = []
    for line in table_lines[2:]:
        cells = _split_markdown_table_row(line)
        if len(cells) != len(expected_header):
            raise AssertionError(f"{path}: malformed Models table row: {line}")

        (
            method_cell,
            task_cell,
            content_cell,
            venue_cell,
            checkpoint_cell,
            training_cell,
        ) = cells[:6]

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
        slugs.append(slug)

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


def docs_library_slugs(path: Path) -> set[str]:
    """Validate the docs library table and return its workspace slugs."""
    text = path.read_text(encoding="utf-8")
    table_lines = _docs_libraries_table_lines(text)
    if _split_markdown_table_row(table_lines[0]) != DOCS_LIBRARY_TABLE_HEADER or not (
        _is_markdown_table_separator(table_lines[1], len(DOCS_LIBRARY_TABLE_HEADER))
    ):
        raise AssertionError(
            f"{path}: Libraries table must use {', '.join(DOCS_LIBRARY_TABLE_HEADER)}"
        )

    slugs: set[str] = set()
    for line in table_lines[2:]:
        cells = _split_markdown_table_row(line)
        if len(cells) != len(DOCS_LIBRARY_TABLE_HEADER):
            raise AssertionError(f"{path}: malformed Libraries table row: {line}")

        library_badge = _linked_static_badge(cells[0], "library")
        if library_badge is None:
            raise AssertionError(
                f"{path}: Library cell must be a linked library badge: {line}"
            )

        label, library_link = library_badge
        slug_match = re.fullmatch(r"api/libraries/([^/)]+)/", library_link)
        if slug_match is None:
            raise AssertionError(
                f"{path}: Library cell must link api/libraries/<slug>/: {line}"
            )

        slug = slug_match.group(1)
        if label != slug:
            raise AssertionError(
                f"{path}: Library badge message {label!r} must match {slug!r}"
            )

        slugs.add(slug)

    return slugs


def assert_docs_libraries_table_matches_members(
    docs_slugs: set[str], path: Path
) -> None:
    """Require the docs library table to match workspace library members."""
    member_slugs = library_member_slugs()
    missing = sorted(member_slugs - docs_slugs)
    extra = sorted(docs_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"{path}: Libraries table mismatch: missing={missing}, extra={extra}"
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


def assert_root_models_table_matches_members(
    root_slugs: Sequence[str], path: Path
) -> None:
    """Require root README model rows to match workspace model members."""
    seen_slugs: set[str] = set()
    duplicate_slugs: set[str] = set()
    for slug in root_slugs:
        if slug in seen_slugs:
            duplicate_slugs.add(slug)

        seen_slugs.add(slug)

    if duplicate_slugs:
        raise AssertionError(
            f"{path}: Models table contains duplicate model slugs: "
            f"{sorted(duplicate_slugs)}"
        )

    root_slug_set = set(root_slugs)
    member_slugs = model_member_slugs()
    missing = sorted(member_slugs - root_slug_set)
    extra = sorted(root_slug_set - member_slugs)
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
    """Require a linked emoji form first and a plain code form afterward."""
    text = without_badges(
        without_frontmatter_and_code(path.read_text(encoding="utf-8"))
    )
    banned_names = ("Transformers", "Diffusers", "Pydantic AI")
    for name in banned_names:
        match = re.search(rf"(?<![`/\w]){re.escape(name)}(?![`/\w])", text)
        if match:
            raise AssertionError(
                f"{path}: use a code-form library name instead of prose name {name!r}"
            )

    library_styles = {
        "transformers": ("🤗", "https://huggingface.co/docs/transformers/index"),
        "diffusers": ("🧨", "https://huggingface.co/docs/diffusers/index"),
        "pydantic-ai": ("🤖", "https://ai.pydantic.dev/"),
    }
    for library, (emoji, official_docs_url) in library_styles.items():
        emoji_code = f"`{emoji}{library}`"
        plain_code = f"`{library}`"
        if text.count(emoji) != text.count(emoji_code):
            raise AssertionError(
                f"{path}: {emoji} must annotate a {library} library mention"
            )

        mentions: list[tuple[re.Match[str], str | None]] = []
        for match in re.finditer(rf"(?<![/\w=-]){re.escape(library)}(?![/\w])", text):
            code_span = inline_code_span_at(text, match.start())
            if code_span is not None and code_span not in (emoji_code, plain_code):
                if code_span.startswith(f"`{library}."):
                    continue

                continue

            mentions.append((match, code_span))

        if not mentions:
            continue

        link_pattern = re.compile(r"\[(`[^`\n]+`)\]\(([^)\n]+)\)")
        for index, (match, code_span) in enumerate(mentions):
            if index == 0:
                link = next(
                    (
                        candidate
                        for candidate in link_pattern.finditer(text)
                        if candidate.start() <= match.start() < candidate.end()
                    ),
                    None,
                )
                if (
                    code_span != emoji_code
                    or link is None
                    or link.group(1) != emoji_code
                    or link.group(2) != official_docs_url
                ):
                    raise AssertionError(
                        f"{path}: first {library} mention must link {emoji_code} "
                        f"to its official documentation at {official_docs_url}"
                    )

                continue

            if code_span != plain_code:
                raise AssertionError(
                    f"{path}: later {library} mention must use {plain_code}"
                )
