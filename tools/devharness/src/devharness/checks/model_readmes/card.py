"""Model-card metadata, structure, and package README contracts."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias
from urllib.parse import parse_qs, unquote, urlparse

from .constants import (
    BANNED_PATTERNS,
    EXPECTED_FRONTMATTER,
    EXPECTED_MODEL_NAMES,
    EXPECTED_REPOSITORY_LINKS,
    MODEL_CONTENT_VALUES,
    MODEL_CONDITIONING_ORDER,
    MODEL_TASK_VALUES,
    MODEL_MEMBER_DIRS,
    PROMPT_ONLY_SLUGS,
    PROMPT_ONLY_STALE_PHRASES,
    REQUIRED_HEADINGS,
    REPO_ROOT,
)
from .install import project_metadata


def section(text: str, heading: str) -> str:
    """Return the body of one level-two Markdown section."""
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        return ""

    rest = text[match.end() :]
    next_heading = re.search(r"\n## ", rest)
    return rest[: next_heading.start()] if next_heading else rest


def _frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return ""

    end = text.find("\n---\n", 4)
    if end == -1:
        return ""

    return text[:end]


def without_frontmatter_and_code(text: str) -> str:
    """Remove frontmatter and fenced code from reader-facing Markdown text."""
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


def without_badges(text: str) -> str:
    """Remove Markdown image badges from reader-facing Markdown text."""
    return re.sub(
        r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)",
        " ",
        text,
    )


def markdown_link_spans(text: str) -> list[range]:
    """Return spans occupied by Markdown links and badges."""
    return [
        range(match.start(), match.end())
        for match in re.finditer(
            r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!?\[[^\]]*\]\([^)]*\)", text
        )
    ]


def in_any_span(position: int, spans: list[range]) -> bool:
    """Return whether a character position belongs to one of the spans."""
    return any(position in span for span in spans)


def inline_code_span_at(text: str, position: int) -> str | None:
    """Return the inline-code span containing a character position, if any."""
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


_DATASET_PRESENTATIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "rico13": (("creative-graphic-design/rico", "rico13"), "RICO13"),
    "rico25": (("creative-graphic-design/rico", "rico25"), "RICO25"),
    "publaynet": (("creative-graphic-design/publaynet", "publaynet"), "PubLayNet"),
    "magazine": (("creative-graphic-design/magazine", "magazine"), "Magazine"),
    "pku_posterlayout": (
        ("creative-graphic-design/pku-posterlayout", "pku_posterlayout", "pku"),
        "PKU",
    ),
    "cgl": (("creative-graphic-design/cgl-dataset", "cgl"), "CGL"),
    "posterlayout": (("posterlayout",), "PosterLayout"),
    "ad_banner": (("ad banner", "ad_banner"), "Ad Banner"),
    "crello": (("cyberagent/crello", "crello"), "Crello"),
    "coco-grounded": (("coco-grounded",), "COCO-grounded"),
    "vg-msdn": (("vg-msdn",), "VG-MSDN"),
    "smarttext-demo": (("smarttext demo", "smarttext-demo"), "SmartText demo"),
    "grit": (("grit",), "GRIT"),
    "nsr-1k": (("nsr-1k",), "NSR-1K"),
    "web": (("web",), "Web"),
    "info ppt": (("infoppt", "info ppt"), "InfoPPT"),
    "coco": (("coco",), "COCO"),
    "housegan-floorplan-vectorized": (
        ("housegan-floorplan-vectorized",),
        "housegan-floorplan-vectorized",
    ),
}


def _dataset_key(value: str) -> str:
    return (
        value.removeprefix("https://huggingface.co/datasets/")
        .casefold()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


@lru_cache(maxsize=None)
def _declared_dataset_displays(slug: str) -> dict[str, str]:
    metadata = project_metadata(REPO_ROOT / "models" / slug)
    tool = metadata.get("tool")
    design_generators = tool.get("design-generators") if isinstance(tool, dict) else {}
    declared = (
        design_generators.get("datasets") if isinstance(design_generators, dict) else []
    )
    if not isinstance(declared, list):
        return {}

    displays: dict[str, str] = {}
    for dataset in declared:
        if not isinstance(dataset, str):
            continue

        presentation = _DATASET_PRESENTATIONS.get(dataset.casefold())
        if presentation is None:
            continue

        sources, display = presentation
        displays.update({_dataset_key(source): display for source in sources})

    return displays


def _dataset_display_name(value: str, slug: str | None = None) -> str:
    if slug is None:
        return value.removeprefix("https://huggingface.co/datasets/").rsplit(
            "/", maxsplit=1
        )[-1]

    return _declared_dataset_displays(slug).get(
        _dataset_key(value),
        value.removeprefix("https://huggingface.co/datasets/").rsplit("/", maxsplit=1)[
            -1
        ],
    )


def semantic_badge_label(alt: str, query_label: str) -> str:
    """Resolve a badge's policy label from its alt text and URL query."""
    alt_prefix, separator, _ = alt.partition(":")
    semantic_alt_prefixes = {
        "checkpoint",
        "content",
        "dataset",
        "framework",
        "library",
        "model",
        "task",
        "training",
        "venue",
    }
    if separator and alt_prefix in semantic_alt_prefixes:
        return alt_prefix

    return query_label


def badge_messages(text: str, label: str) -> list[str]:
    """Return Shields badge messages matching a semantic label."""
    messages: list[str] = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        parsed = urlparse(match.group(2))
        if parsed.netloc != "img.shields.io" or parsed.path != "/static/v1":
            continue

        query = parse_qs(parsed.query)
        query_label = query.get("label", [None])[0]
        if (
            query_label is not None
            and semantic_badge_label(match.group(1), query_label) == label
            and "message" in query
        ):
            messages.append(unquote(query["message"][0]).replace("--", "-"))

    return messages


def model_member_slugs() -> set[str]:
    """Return model workspace member directory names."""
    return {member_dir.name for member_dir in MODEL_MEMBER_DIRS}


ModelType: TypeAlias = tuple[str, str, tuple[str, ...]]

_MODEL_TYPE_LINE_RE = re.compile(
    r"^- \*\*Model type:\*\* (?P<content>[^;\n]+); task: (?P<task>[^;\n]+); conditioning: "
    r"(?P<conditioning>[^.\n]+)\.$"
)


def parse_model_type(path: Path, text: str) -> ModelType:
    """Parse and validate the one structured model classification line."""
    lines = [line for line in text.splitlines() if "**Model type:**" in line]
    if len(lines) != 1:
        raise AssertionError(
            f"{path}: Model type line must appear exactly once, found {len(lines)}"
        )

    match = _MODEL_TYPE_LINE_RE.fullmatch(lines[0])
    if match is None:
        raise AssertionError(
            f"{path}: Model type line must match "
            "'- **Model type:** <content>; task: <task>; conditioning: <values>.'"
        )

    content = match.group("content")
    if content not in MODEL_CONTENT_VALUES:
        raise AssertionError(
            f"{path}: Model type content value {content!r} is invalid; "
            f"expected one of {MODEL_CONTENT_VALUES}"
        )

    task = match.group("task")
    if task not in MODEL_TASK_VALUES:
        raise AssertionError(
            f"{path}: Model type task value {task!r} is invalid; "
            f"expected one of {MODEL_TASK_VALUES}"
        )

    conditioning = tuple(
        value.strip() for value in match.group("conditioning").split(",")
    )
    if any(not value for value in conditioning):
        raise AssertionError(f"{path}: Model type conditioning values cannot be empty")

    for value in conditioning:
        if value not in MODEL_CONDITIONING_ORDER:
            raise AssertionError(
                f"{path}: Model type conditioning value {value!r} is invalid; "
                f"expected one of {MODEL_CONDITIONING_ORDER}"
            )

    if len(set(conditioning)) != len(conditioning):
        raise AssertionError(f"{path}: Model type conditioning values must be unique")

    special_values = {"evaluation", "saliency", "none"}
    if len(conditioning) > 1 and special_values.intersection(conditioning):
        raise AssertionError(
            f"{path}: evaluation, saliency, and none must appear alone"
        )

    if task in {"evaluation", "saliency"}:
        if conditioning != (task,):
            raise AssertionError(
                f"{path}: {task} task requires matching non-generation conditioning"
            )
    elif conditioning in (("evaluation",), ("saliency",)):
        raise AssertionError(
            f"{path}: evaluation and saliency conditioning require matching task"
        )
    elif len(conditioning) == 1 and task != "single-task":
        raise AssertionError(
            f"{path}: single-task is required for one conditioning value"
        )
    elif len(conditioning) >= 2 and task not in {"task-agnostic", "task-aware"}:
        raise AssertionError(
            f"{path}: task-agnostic or task-aware is required for multiple conditioning values"
        )

    positions = [MODEL_CONDITIONING_ORDER.index(value) for value in conditioning]
    if positions != sorted(positions):
        raise AssertionError(
            f"{path}: Model type conditioning values must use the canonical order"
        )

    return content, task, conditioning


def library_member_slugs() -> set[str]:
    """Return library workspace member directory names."""
    return {
        path.parent.name
        for path in sorted((REPO_ROOT / "lib").glob("*/pyproject.toml"))
    }


def assert_frontmatter_list_unique(path: Path, frontmatter: str) -> None:
    """Require unique list values in model-card frontmatter."""
    for key in ("language", "tags", "datasets"):
        values = _frontmatter_list(frontmatter, key)
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise AssertionError(
                f"{path}: frontmatter {key} has duplicates {duplicates}"
            )


def assert_pipeline_tag(path: Path, frontmatter: str) -> None:
    """Require the model-card task metadata for layout generation."""
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


def assert_model_index_policy(path: Path, frontmatter: str) -> None:
    """Require model-index metadata only for weight-backed model cards."""
    has_model_index = "model-index:" in frontmatter
    if path.parent.name in PROMPT_ONLY_SLUGS:
        if has_model_index:
            raise AssertionError(
                f"{path}: prompt-only README must not include model-index"
            )

        return

    if not has_model_index:
        raise AssertionError(f"{path}: weight-backed README must include model-index")


def assert_heading_order(path: Path, text: str) -> None:
    """Require model-card headings in the documented order."""
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


def assert_frontmatter(path: Path, text: str) -> None:
    """Require the model-card frontmatter envelope and required keys."""
    if not text.startswith("---\n"):
        raise AssertionError(f"{path}: missing YAML frontmatter")

    end = text.find("\n---\n", 4)
    if end == -1:
        raise AssertionError(f"{path}: unterminated YAML frontmatter")

    frontmatter = _frontmatter(text)
    for key in ("language:", "license:", "library_name:", "pipeline_tag:", "tags:"):
        if key not in frontmatter:
            raise AssertionError(f"{path}: frontmatter missing {key}")


def assert_expected_frontmatter(path: Path, text: str) -> None:
    """Require frontmatter values and dataset badges for the model slug."""
    slug = path.parent.name
    expected = EXPECTED_FRONTMATTER.get(slug)
    if expected is None:
        raise AssertionError(f"{path}: no expected frontmatter contract for {slug}")

    frontmatter = _frontmatter(text)
    assert_frontmatter_list_unique(path, frontmatter)
    assert_pipeline_tag(path, frontmatter)
    assert_model_index_policy(path, frontmatter)
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

    expected_badges = {
        _dataset_display_name(dataset, slug) for dataset in actual_datasets
    }
    actual_badges = set(badge_messages(text, "dataset"))
    if actual_badges != expected_badges:
        raise AssertionError(
            f"{path}: dataset badges {sorted(actual_badges)} != frontmatter datasets {sorted(expected_badges)}"
        )


def assert_runtime_contract(path: Path, text: str) -> None:
    """Require matching runtime metadata across README and pyproject surfaces."""
    slug = path.parent.name
    frontmatter_library = _frontmatter_scalar(_frontmatter(text), "library_name")
    metadata = project_metadata(REPO_ROOT / "models" / slug)
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

    base_badges = badge_messages(text, "base")
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


def assert_model_summary_subject(path: Path, text: str) -> None:
    """Require the first model-card prose sentence to use the package subject."""
    model_name = EXPECTED_MODEL_NAMES[path.parent.name]
    body = without_frontmatter_and_code(text)
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


def assert_expected_repository_links(path: Path, text: str) -> None:
    """Require the known upstream repository link for a model package."""
    expected = EXPECTED_REPOSITORY_LINKS.get(path.parent.name)
    if expected is None:
        return

    sources = section(text, "### Model Sources")
    if expected not in sources:
        raise AssertionError(
            f"{path}: Model Sources must link expected repository {expected}"
        )


def assert_prompt_only_readme(path: Path, text: str) -> None:
    """Require prompt-only model cards to omit weight-package language."""
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


def assert_unpublished_hub_get_started_note(path: Path, text: str) -> None:
    """Require runnable local setup for unpublished model checkpoints."""
    supported = section(text, "## Supported Checkpoints")
    get_started = section(text, "## How to Get Started with the Model")
    if "<<'PY'" in get_started or '<<"PY"' in get_started:
        raise AssertionError(f"{path}: Get Started must not use heredoc examples")

    if path.parent.name in PROMPT_ONLY_SLUGS:
        required = [
            "git clone https://github.com/creative-graphic-design/design-generators.git",
            f"uv sync --package {path.parent.name}",
            "no learned checkpoints",
        ]
        missing = [snippet for snippet in required if snippet not in get_started]
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
    missing = [snippet for snippet in required if snippet not in get_started]
    if missing:
        raise AssertionError(
            f"{path}: unpublished Hub Get Started snippet is missing runnable local-loading parts {missing}"
        )


def assert_code_fences_tagged(path: Path, text: str) -> None:
    """Require tagged, terminated Markdown code fences without heredocs."""
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


def usage_output_violation(path: Path, text: str) -> str | None:
    """Return a violation when the first Python usage fence lacks text output."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == "```python"),
        None,
    )
    if start is None:
        return f"{path}: usage example must contain a python fence"

    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("```")
        ),
        None,
    )
    if end is None:
        return f"{path}: first python fence is unterminated"

    next_line = end + 1
    while next_line < len(lines) and not lines[next_line].strip():
        next_line += 1

    if next_line == len(lines) or lines[next_line] != "```text":
        return f"{path}: first python fence must be followed by a text fence"

    return None


def assert_banned_patterns(path: Path, text: str) -> None:
    """Reject credentials and repository-process language in reader-facing text."""
    for pattern in BANNED_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raise AssertionError(
                f"{path}: banned README content matched {pattern!r}: {match.group(0)!r}"
            )
