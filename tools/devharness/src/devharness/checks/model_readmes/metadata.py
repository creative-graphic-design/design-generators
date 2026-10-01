"""Model-card metadata, structure, and package README contracts."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .constants import (
    EXPECTED_FRONTMATTER,
    EXPECTED_MODEL_NAMES,
    EXPECTED_REPOSITORY_LINKS,
    MODEL_MEMBER_DIRS,
    PROMPT_ONLY_SLUGS,
    PROMPT_ONLY_STALE_PHRASES,
    REQUIRED_HEADINGS,
    REPO_ROOT,
)
from .repository import _project_metadata, _section


def _frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return ""

    end = text.find("\n---\n", 4)
    if end == -1:
        return ""

    return text[:end]


def _without_frontmatter_and_code(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + len("\n---\n") :]

    lines: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue

        if not in_code:
            lines.append(line)

    return "\n".join(lines)


def _without_badges(text: str) -> str:
    return re.sub(
        r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)",
        " ",
        text,
    )


def _markdown_link_spans(text: str) -> list[range]:
    return [
        range(match.start(), match.end())
        for match in re.finditer(
            r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!?\[[^\]]*\]\([^)]*\)", text
        )
    ]


def _in_any_span(position: int, spans: list[range]) -> bool:
    return any(position in span for span in spans)


def _inline_code_span_at(text: str, position: int) -> str | None:
    for match in re.finditer(r"`[^`\n]+`", text):
        if position in range(match.start(), match.end()):
            return match.group(0)

    return None


def _frontmatter_scalar(frontmatter: str, key: str) -> str | None:
    match = re.search(
        rf"^{re.escape(key)}:\s*[\"']?([^\"'\n]+)[\"']?\s*$", frontmatter, re.MULTILINE
    )
    return match.group(1) if match else None


def _frontmatter_list(frontmatter: str, key: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(key)}:\s*\n((?:  - .+\n?)*)", frontmatter, re.MULTILINE
    )
    if match is None:
        return []

    return [
        line.split("-", 1)[1].strip().strip('"').strip("'")
        for line in match.group(1).splitlines()
        if line.strip().startswith("- ")
    ]


def _dataset_display_name(value: str) -> str:
    normalized = value.removeprefix("https://huggingface.co/datasets/")
    return {
        "creative-graphic-design/Rico": "RICO25",
        "creative-graphic-design/PubLayNet": "PubLayNet",
        "creative-graphic-design/magazine": "Magazine",
        "creative-graphic-design/CGL-Dataset": "CGL",
        "creative-graphic-design/PKU-PosterLayout": "PKU",
        "cyberagent/crello": "Crello",
    }.get(normalized, normalized)


def _semantic_badge_label(alt: str, query_label: str) -> str:
    alt_prefix, separator, _ = alt.partition(":")
    semantic_alt_prefixes = {
        "checkpoint",
        "dataset",
        "framework",
        "library",
        "model",
        "training",
        "venue",
    }
    if separator and alt_prefix in semantic_alt_prefixes:
        return alt_prefix

    return query_label


def _badge_messages(text: str, label: str) -> list[str]:
    messages: list[str] = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        parsed = urlparse(match.group(2))
        if parsed.netloc != "img.shields.io" or parsed.path != "/static/v1":
            continue

        query = parse_qs(parsed.query)
        query_label = query.get("label", [None])[0]
        if (
            query_label is not None
            and _semantic_badge_label(match.group(1), query_label) == label
            and "message" in query
        ):
            messages.append(unquote(query["message"][0]).replace("--", "-"))

    return messages


def _model_member_slugs() -> set[str]:
    return {member_dir.name for member_dir in MODEL_MEMBER_DIRS}


def _library_member_slugs() -> set[str]:
    return {
        path.parent.name
        for path in sorted((REPO_ROOT / "lib").glob("*/pyproject.toml"))
    }


def _assert_frontmatter_list_unique(path: Path, frontmatter: str) -> None:
    for key in ("language", "tags", "datasets"):
        values = _frontmatter_list(frontmatter, key)
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise AssertionError(
                f"{path}: frontmatter {key} has duplicates {duplicates}"
            )


def _assert_pipeline_tag(path: Path, frontmatter: str) -> None:
    pipeline_tag = _frontmatter_scalar(frontmatter, "pipeline_tag")
    if pipeline_tag != "other":
        raise AssertionError(
            f"{path}: pipeline_tag must be 'other' for layout generation, got {pipeline_tag!r}"
        )

    bad_task = re.search(
        r'^\s+type:\s*["\']?text-to-image["\']?\s*$', frontmatter, re.MULTILINE
    )
    if bad_task:
        raise AssertionError(f"{path}: model-index task.type must not be text-to-image")

    task_types = re.findall(
        r'^\s+type:\s*["\']?([^"\'\n]+)["\']?\s*$', frontmatter, re.MULTILINE
    )
    if "other" not in task_types and "model-index:" in frontmatter:
        raise AssertionError(f"{path}: model-index task.type must be 'other'")


def _assert_model_index_policy(path: Path, frontmatter: str) -> None:
    has_model_index = "model-index:" in frontmatter
    if path.parent.name in PROMPT_ONLY_SLUGS:
        if has_model_index:
            raise AssertionError(
                f"{path}: prompt-only README must not include model-index"
            )

        return

    if not has_model_index:
        raise AssertionError(f"{path}: weight-backed README must include model-index")


def _assert_heading_order(path: Path, text: str) -> None:
    cursor = -1
    for heading in REQUIRED_HEADINGS:
        pattern = (
            rf"^{re.escape(heading)}.*$"
            if heading.endswith(" ")
            else rf"^{re.escape(heading)}\s*$"
        )
        matches = [
            match
            for match in re.finditer(pattern, text, re.MULTILINE)
            if match.start() > cursor
        ]
        if not matches:
            raise AssertionError(f"{path}: missing required heading {heading!r}")

        cursor = matches[0].start()


def _assert_frontmatter(path: Path, text: str) -> None:
    if not text.startswith("---\n"):
        raise AssertionError(f"{path}: missing YAML frontmatter")

    end = text.find("\n---\n", 4)
    if end == -1:
        raise AssertionError(f"{path}: unterminated YAML frontmatter")

    frontmatter = _frontmatter(text)
    for key in ("language:", "license:", "library_name:", "pipeline_tag:", "tags:"):
        if key not in frontmatter:
            raise AssertionError(f"{path}: frontmatter missing {key}")


def _assert_expected_frontmatter(path: Path, text: str) -> None:
    slug = path.parent.name
    expected = EXPECTED_FRONTMATTER.get(slug)
    if expected is None:
        raise AssertionError(f"{path}: no expected frontmatter contract for {slug}")

    frontmatter = _frontmatter(text)
    _assert_frontmatter_list_unique(path, frontmatter)
    _assert_pipeline_tag(path, frontmatter)
    _assert_model_index_policy(path, frontmatter)
    actual_license = _frontmatter_scalar(frontmatter, "license")
    expected_license = expected["license"]
    if actual_license != expected_license:
        raise AssertionError(
            f"{path}: frontmatter license {actual_license!r} != {expected_license!r}"
        )

    actual_datasets = set(_frontmatter_list(frontmatter, "datasets"))
    missing = sorted(set(expected["datasets"]) - actual_datasets)
    if missing:
        raise AssertionError(
            f"{path}: frontmatter datasets missing supported checkpoint datasets {missing}"
        )

    expected_badges = {_dataset_display_name(dataset) for dataset in actual_datasets}
    actual_badges = set(_badge_messages(text, "dataset"))
    if actual_badges != expected_badges:
        raise AssertionError(
            f"{path}: dataset badges {sorted(actual_badges)} != frontmatter datasets {sorted(expected_badges)}"
        )


def _assert_runtime_contract(path: Path, text: str) -> None:
    slug = path.parent.name
    frontmatter_library = _frontmatter_scalar(_frontmatter(text), "library_name")
    metadata = _project_metadata(REPO_ROOT / "models" / slug)
    tool = metadata.get("tool")
    if not isinstance(tool, dict):
        raise AssertionError(f"{path}: missing [tool]")

    design_generators = tool.get("design-generators")
    if not isinstance(design_generators, dict):
        raise AssertionError(f"{path}: missing [tool.design-generators]")

    pyproject_library = design_generators.get("framework")
    if not isinstance(pyproject_library, str):
        raise AssertionError(
            f"{path}: tool.design-generators.framework must be a string"
        )

    base_badges = _badge_messages(text, "base")
    if len(base_badges) != 1:
        raise AssertionError(
            f"{path}: expected exactly one base badge, found {base_badges}"
        )

    base_library = base_badges[0]
    values = {
        "frontmatter library_name": frontmatter_library,
        "base badge": base_library,
        "pyproject framework": pyproject_library,
    }
    if len(set(values.values())) != 1:
        raise AssertionError(
            f"{path}: runtime mismatch across package metadata surfaces {values}"
        )


def _assert_model_summary_subject(path: Path, text: str) -> None:
    model_name = EXPECTED_MODEL_NAMES[path.parent.name]
    body = _without_frontmatter_and_code(text)
    h1 = re.search(r"^# Model Card for .+$", body, re.MULTILINE)
    if h1 is None:
        raise AssertionError(f"{path}: missing model-card H1")

    for line in body[h1.end() :].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("![") or stripped.startswith("[!["):
            continue

        if not stripped.startswith("This package "):
            raise AssertionError(
                f"{path}: first prose line must use package subject, got {stripped!r}"
            )

        if re.search(
            rf"^{re.escape(model_name)}\s+(ports|wraps|implements)\b", stripped
        ):
            raise AssertionError(
                f"{path}: method name must not be the subject of {model_name} summary"
            )

        break


def _assert_expected_repository_links(path: Path, text: str) -> None:
    expected = EXPECTED_REPOSITORY_LINKS.get(path.parent.name)
    if expected is None:
        return

    sources = _section(text, "### Model Sources")
    if expected not in sources:
        raise AssertionError(
            f"{path}: Model Sources must link expected repository {expected}"
        )


def _assert_prompt_only_readme(path: Path, text: str) -> None:
    if path.parent.name not in PROMPT_ONLY_SLUGS:
        return

    for phrase in PROMPT_ONLY_STALE_PHRASES:
        if phrase in text:
            raise AssertionError(
                f"{path}: prompt-only README contains stale model-package phrase {phrase!r}"
            )

    if "convert checkpoints" in text.lower():
        raise AssertionError(
            f"{path}: prompt-only README must not mention converting checkpoints"
        )


def _assert_unpublished_hub_get_started_note(path: Path, text: str) -> None:
    supported = _section(text, "## Supported Checkpoints")
    section = _section(text, "## How to Get Started with the Model")
    if "<<'PY'" in section or '<<"PY"' in section:
        raise AssertionError(f"{path}: Get Started must not use heredoc examples")

    if path.parent.name in PROMPT_ONLY_SLUGS:
        required = [
            "git clone https://github.com/creative-graphic-design/design-generators.git",
            f"uv sync --package {path.parent.name}",
            "no learned checkpoints",
        ]
        missing = [snippet for snippet in required if snippet not in section]
        if missing:
            raise AssertionError(
                f"{path}: prompt-only Get Started is missing runnable setup parts {missing}"
            )

        return

    if "creative-graphic-design/" not in supported or "not-published" not in supported:
        return

    required = [
        "git clone https://github.com/creative-graphic-design/design-generators.git",
        f"uv sync --package {path.parent.name}",
        f"`.cache/{path.parent.name}/converted",
        "REPRODUCING.md](",
        "# After Hub publication: from_pretrained(",
    ]
    missing = [snippet for snippet in required if snippet not in section]
    if missing:
        raise AssertionError(
            f"{path}: unpublished Hub Get Started snippet is missing runnable local-loading parts {missing}"
        )


def _assert_code_fences_tagged(path: Path, text: str) -> None:
    if re.search(r"<<['\"]?(PY|EOF)['\"]?", text):
        raise AssertionError(f"{path}: heredoc examples are not allowed")

    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("```"):
            continue

        if in_fence:
            in_fence = False
            continue

        info = line[3:].strip()
        if info == "":
            raise AssertionError(f"{path}: untagged code fence at line {lineno}")

        in_fence = True

    if in_fence:
        raise AssertionError(f"{path}: unterminated code fence")


def _assert_lib_readme_install_contract(path: Path, text: str) -> None:
    package = path.parent.name
    install = _section(text, "## Install")
    direct_reference = (
        f'pip install "{package} @ '
        "git+https://github.com/creative-graphic-design/design-generators.git"
        f'#subdirectory=lib/{package}"'
    )
    if direct_reference not in install:
        raise AssertionError(
            f"{path}: Install must include pip direct-reference subdirectory form"
        )

    if f"uv sync --package {package}" not in install:
        raise AssertionError(f"{path}: Install must include workspace uv sync form")


def _assert_parity_table(path: Path, text: str) -> None:
    section = _section(text, "### Parity Results")
    if "| ---" not in section:
        raise AssertionError(f"{path}: Parity Results must contain a markdown table")

    rows = [
        line
        for line in section.splitlines()
        if line.startswith("|") and not line.startswith("| ---")
    ]
    data_rows = rows[1:]
    if not data_rows:
        raise AssertionError(f"{path}: Parity Results table has no data rows")

    if not any(re.search(r"\d", row) for row in data_rows):
        raise AssertionError(
            f"{path}: Parity Results table must contain numeric evidence"
        )
