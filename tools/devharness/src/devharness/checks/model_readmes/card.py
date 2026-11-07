"""Model-card metadata, structure, and package README contracts."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import parse_qs, unquote, urlparse

from .constants import (
    BANNED_PATTERNS,
    EXPECTED_FRONTMATTER,
    EXPECTED_MODEL_INDEX_TASKS,
    EXPECTED_MODEL_NAMES,
    EXPECTED_PIPELINE_TAGS,
    EXPECTED_REPOSITORY_LINKS,
    IMAGE_GENERATION_CATEGORY,
    IMAGE_GENERATION_CONDITIONING_ORDER,
    IMAGE_GENERATION_TASK_VALUES,
    LAYOUT_TASK_VALUES,
    MODEL_CONDITIONING_ORDER,
    MODEL_TYPE_VALUES,
    MODEL_MEMBER_DIRS,
    NO_MODEL_WEIGHTS_SLUGS,
    PROMPT_ONLY_SLUGS,
    PROMPT_ONLY_STALE_PHRASES,
    REQUIRED_HEADINGS,
    REPO_ROOT,
)
from .install import _project_name, project_metadata


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


_FENCE_OPENING_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_CLOSING_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")


def without_frontmatter_and_code(text: str) -> str:
    """Remove frontmatter and fenced code from reader-facing Markdown text."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + len("\n---\n") :]

    lines: list[str] = []
    opening_fence: str | None = None
    for line in text.splitlines():
        if opening_fence is None:
            opening = _FENCE_OPENING_RE.match(line)
            if opening is None:
                lines.append(line)
                continue

            opening_fence = opening.group("fence")
            info = opening.group("info")
            if opening_fence.startswith("`") and "`" in info:
                lines.append(line)
                opening_fence = None

            continue

        closing = _FENCE_CLOSING_RE.match(line)
        if closing is None:
            continue

        closing_fence = closing.group("fence")
        if closing_fence[0] == opening_fence[0] and len(closing_fence) >= len(
            opening_fence
        ):
            opening_fence = None

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


_DATASET_PRESENTATIONS: dict[str, tuple[str, str]] = {
    "rico13": ("creative-graphic-design/rico", "RICO13"),
    "rico25": ("creative-graphic-design/rico", "RICO25"),
    "publaynet": ("creative-graphic-design/publaynet", "PubLayNet"),
    "magazine": ("creative-graphic-design/magazine", "Magazine"),
    "pku_posterlayout": ("creative-graphic-design/pku-posterlayout", "PKU"),
    "cgl": ("creative-graphic-design/cgl-dataset", "CGL"),
    "posterlayout": ("posterlayout", "PosterLayout"),
    "ad_banner": ("ad banner", "Ad Banner"),
    "crello": ("cyberagent/crello", "Crello"),
    "coco-grounded": ("coco-grounded", "COCO-grounded"),
    "vg-msdn": ("vg-msdn", "VG-MSDN"),
    "smarttext-demo": ("smarttext demo", "SmartText demo"),
    "grit": ("grit", "GRIT"),
    "nsr-1k": ("nsr-1k", "NSR-1K"),
    "web": ("web", "Web"),
    "info ppt": ("infoppt", "InfoPPT"),
    "coco": ("coco", "COCO"),
    "housegan-floorplan-vectorized": (
        "housegan-floorplan-vectorized",
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

        source, display = presentation
        displays[_dataset_key(dataset)] = display
        displays[_dataset_key(source)] = display

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


def model_member_count() -> int:
    """Return the number of model workspace members."""
    return len(model_member_slugs())


ModelType: TypeAlias = tuple[str, str, tuple[str, ...]]

_MODEL_TYPE_LINE_RE = re.compile(
    r"^- \*\*Model type:\*\* (?P<category>[^;\n]+); task: (?P<task>[^;\n]+); conditioning: "
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
            "'- **Model type:** <category>; task: <task>; conditioning: <values>.'"
        )

    category = match.group("category")
    if category not in MODEL_TYPE_VALUES:
        raise AssertionError(
            f"{path}: Model type category value {category!r} is invalid; "
            f"expected one of {MODEL_TYPE_VALUES}"
        )

    task = match.group("task")
    conditioning = tuple(
        value.strip() for value in match.group("conditioning").split(",")
    )
    if any(not value for value in conditioning):
        raise AssertionError(f"{path}: Model type conditioning values cannot be empty")

    if len(set(conditioning)) != len(conditioning):
        raise AssertionError(f"{path}: Model type conditioning values must be unique")

    if category == IMAGE_GENERATION_CATEGORY:
        if task not in IMAGE_GENERATION_TASK_VALUES:
            raise AssertionError(
                f"{path}: image generation task value {task!r} is invalid; "
                f"expected one of {IMAGE_GENERATION_TASK_VALUES}"
            )

        conditioning_order = IMAGE_GENERATION_CONDITIONING_ORDER
    else:
        if task not in LAYOUT_TASK_VALUES:
            raise AssertionError(
                f"{path}: Model type task value {task!r} is invalid; "
                f"expected one of {LAYOUT_TASK_VALUES}"
            )

        conditioning_order = MODEL_CONDITIONING_ORDER

    for value in conditioning:
        if value not in conditioning_order:
            raise AssertionError(
                f"{path}: Model type conditioning value {value!r} is invalid; "
                f"expected one of {conditioning_order}"
            )

    if category == IMAGE_GENERATION_CATEGORY:
        positions = [conditioning_order.index(value) for value in conditioning]
        if positions != sorted(positions):
            raise AssertionError(
                f"{path}: Model type conditioning values must use the canonical order"
            )

        return category, task, conditioning

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

    positions = [conditioning_order.index(value) for value in conditioning]
    if positions != sorted(positions):
        raise AssertionError(
            f"{path}: Model type conditioning values must use the canonical order"
        )

    return category, task, conditioning


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


@dataclass(frozen=True)
class _ModelIndexScalar:
    value: str
    canonical: str
    kind: Literal["string", "number"]


@dataclass(frozen=True)
class _ModelIndexMapping:
    entries: tuple[tuple[str, _ModelIndexNode], ...]


@dataclass(frozen=True)
class _ModelIndexSequence:
    items: tuple[_ModelIndexNode, ...]


_ModelIndexNode: TypeAlias = (
    _ModelIndexScalar | _ModelIndexMapping | _ModelIndexSequence
)


@dataclass(frozen=True)
class _ModelIndexLine:
    indentation: int
    content: str


_MODEL_INDEX_FIELD_ORDER = {
    ("model-index", "[]"): ("name", "results"),
    ("model-index", "[]", "results", "[]"): ("task", "dataset", "metrics"),
    ("model-index", "[]", "results", "[]", "task"): ("type", "name"),
    ("model-index", "[]", "results", "[]", "dataset"): (
        "type",
        "name",
        "config",
        "split",
        "args",
    ),
    ("model-index", "[]", "results", "[]", "metrics", "[]"): (
        "type",
        "value",
        "name",
    ),
}

_MODEL_INDEX_PLAIN_STRING = re.compile(r"[A-Za-z][A-Za-z0-9./ -]*")
_MODEL_INDEX_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?")
_MODEL_INDEX_YAML_WORDS = {"false", "n", "no", "null", "off", "on", "true", "y", "yes"}


def _unsupported_model_index_format(path: Path) -> AssertionError:
    """Return an explicit error for model-index syntax the checker cannot read."""
    return AssertionError(
        f"{path}: cannot read this model-index format; use block-style YAML "
        "with two-space indentation, canonical field order, and supported scalars"
    )


def _parse_model_index_scalar(path: Path, raw: str) -> _ModelIndexScalar:
    """Parse only scalar forms that can be written back unambiguously."""
    if raw.startswith('"'):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _unsupported_model_index_format(path) from error

        if not isinstance(value, str):
            raise _unsupported_model_index_format(path)

        canonical = json.dumps(value, ensure_ascii=False)
        kind: Literal["string", "number"] = "string"
    elif _MODEL_INDEX_DECIMAL.fullmatch(raw) and raw != "-0":
        value = raw
        canonical = raw
        kind = "number"
    elif (
        _MODEL_INDEX_PLAIN_STRING.fullmatch(raw)
        and raw.casefold() not in _MODEL_INDEX_YAML_WORDS
    ):
        value = raw
        canonical = raw
        kind = "string"
    else:
        raise _unsupported_model_index_format(path)

    if any(not 0x20 <= ord(character) <= 0x7E for character in value):
        raise _unsupported_model_index_format(path)

    return _ModelIndexScalar(value, canonical, kind)


def _strip_yaml_comment(line: str) -> str:
    """Remove a YAML comment while preserving quoted hash characters."""
    quote: str | None = None
    position = 0
    while position < len(line):
        character = line[position]
        if quote == "'":
            if character == "'":
                if line[position + 1 : position + 2] == "'":
                    position += 2

                    continue

                quote = None
        elif quote == '"':
            if character == "\\":
                position += 2

                continue

            if character == '"':
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and (position == 0 or line[position - 1].isspace()):
            return line[:position]

        position += 1

    return line


def _starts_flow_collection(content: str) -> bool:
    """Return whether a YAML line starts a flow-style mapping or sequence."""
    value = content.strip()
    if value.startswith("-"):
        value = value[1:].lstrip()

    if value.startswith(("{", "[")):
        return True

    key_separator = re.match(r"[^:]+:\s*", value)
    if key_separator is None:
        return False

    return value[key_separator.end() :].startswith(("{", "["))


def _model_index_block(frontmatter: str) -> str | None:
    """Extract a model-index block through the next top-level key."""
    lines = frontmatter.splitlines(keepends=True)
    for start, line in enumerate(lines):
        content = _strip_yaml_comment(line.rstrip("\r\n")).strip()
        if not re.match(r"[\"']?model-index[\"']?\s*:", content):
            continue

        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].strip() and not lines[index][0].isspace()
            ),
            len(lines),
        )

        return "".join(lines[start:end])

    return None


def _model_index_lines(path: Path, block: str) -> tuple[_ModelIndexLine, ...]:
    """Tokenize the supported block subset while rejecting flow collections."""
    lines: list[_ModelIndexLine] = []
    for line in block.splitlines():
        uncommented = _strip_yaml_comment(line)
        if not uncommented.strip():
            continue

        leading = re.match(r"[ \t]*", uncommented)
        assert leading is not None
        indentation_text = leading.group()
        if "\t" in indentation_text:
            raise _unsupported_model_index_format(path)

        content = uncommented[len(indentation_text) :].strip()
        if _starts_flow_collection(content):
            raise _unsupported_model_index_format(path)

        lines.append(_ModelIndexLine(len(indentation_text), content))

    return tuple(lines)


def _mapping_entry(content: str) -> tuple[str, str | None] | None:
    match = re.fullmatch(r"(?P<key>[A-Za-z0-9_-]+):(?:\s+(?P<value>.+))?", content)
    if match is None:
        return None

    value = match.group("value")
    return match.group("key"), value.strip() if value is not None else None


def _ordered_model_index_entries(
    entries: tuple[tuple[str, _ModelIndexNode], ...], path: tuple[str, ...]
) -> tuple[tuple[str, _ModelIndexNode], ...]:
    order = _MODEL_INDEX_FIELD_ORDER.get(path)
    if order is None:
        return tuple(sorted(entries, key=lambda entry: entry[0]))

    rank = {key: index for index, key in enumerate(order)}
    return tuple(
        sorted(entries, key=lambda entry: (rank.get(entry[0], len(rank)), entry[0]))
    )


def _parse_model_index_block(
    path: Path,
    lines: tuple[_ModelIndexLine, ...],
    index: int = 0,
    canonical_indentation: int = 0,
    node_path: tuple[str, ...] = (),
) -> tuple[_ModelIndexNode, int]:
    """Parse the small block-YAML subset used by model-card model-index data."""
    if index >= len(lines):
        raise _unsupported_model_index_format(path)

    line = lines[index]
    if line.content == "-" or line.content.startswith("- "):
        sequence_indentation = line.indentation
        items: list[_ModelIndexNode] = []
        sibling_order = _MODEL_INDEX_FIELD_ORDER.get(node_path + ("[]",), ())

        while (
            index < len(lines)
            and lines[index].indentation == sequence_indentation
            and (lines[index].content == "-" or lines[index].content.startswith("- "))
        ):
            rest = lines[index].content[1:].strip()
            index += 1
            first_entry = _mapping_entry(rest)
            if first_entry is None:
                items.append(_parse_model_index_scalar(path, rest))

                continue

            first_key, first_value = first_entry
            item_path = node_path + ("[]",)
            entries: list[tuple[str, _ModelIndexNode]] = []
            seen_keys = {first_key}

            if first_value is None:
                next_entry = (
                    _mapping_entry(lines[index].content) if index < len(lines) else None
                )
                is_sibling = next_entry is not None and next_entry[0] in sibling_order
                if (
                    index < len(lines)
                    and lines[index].indentation > sequence_indentation
                    and not is_sibling
                ):
                    value, index = _parse_model_index_block(
                        path,
                        lines,
                        index,
                        canonical_indentation + 4,
                        item_path + (first_key,),
                    )
                else:
                    raise _unsupported_model_index_format(path)
            else:
                value = _parse_model_index_scalar(path, first_value)

            entries.append((first_key, value))

            while index < len(lines):
                entry = _mapping_entry(lines[index].content)
                if (
                    entry is None
                    or entry[0] not in sibling_order
                    or entry[0] in seen_keys
                ):
                    break

                key, raw_value = entry
                seen_keys.add(key)
                index += 1
                if raw_value is None:
                    if (
                        index < len(lines)
                        and lines[index].indentation > lines[index - 1].indentation
                    ):
                        child, index = _parse_model_index_block(
                            path,
                            lines,
                            index,
                            canonical_indentation + 4,
                            item_path + (key,),
                        )
                    else:
                        raise _unsupported_model_index_format(path)
                else:
                    child = _parse_model_index_scalar(path, raw_value)

                entries.append((key, child))

            items.append(
                _ModelIndexMapping(
                    _ordered_model_index_entries(tuple(entries), item_path)
                )
            )

        return _ModelIndexSequence(tuple(items)), index

    indentation = line.indentation
    entries = []
    seen_keys: set[str] = set()
    while (
        index < len(lines)
        and lines[index].indentation == indentation
        and not lines[index].content.startswith("-")
    ):
        entry = _mapping_entry(lines[index].content)
        if entry is None:
            raise _unsupported_model_index_format(path)

        key, raw_value = entry
        if key in seen_keys:
            raise _unsupported_model_index_format(path)

        seen_keys.add(key)
        index += 1
        if raw_value is None:
            if index < len(lines) and lines[index].indentation > indentation:
                value, index = _parse_model_index_block(
                    path,
                    lines,
                    index,
                    canonical_indentation + 2,
                    node_path + (key,),
                )
            else:
                raise _unsupported_model_index_format(path)
        else:
            value = _parse_model_index_scalar(path, raw_value)

        entries.append((key, value))

    return _ModelIndexMapping(
        _ordered_model_index_entries(tuple(entries), node_path)
    ), index


def _serialize_model_index_node(
    node: _ModelIndexNode, indentation: int, node_path: tuple[str, ...] = ()
) -> list[str]:
    """Write a parsed model-index node in canonical block style."""
    prefix = " " * indentation
    if isinstance(node, _ModelIndexScalar):
        return [prefix + node.canonical]

    if isinstance(node, _ModelIndexMapping):
        output: list[str] = []
        for key, value in _ordered_model_index_entries(node.entries, node_path):
            entry_prefix = prefix + f"{key}:"
            if isinstance(value, _ModelIndexScalar):
                output.append(entry_prefix + f" {value.canonical}")
            else:
                output.append(entry_prefix)
                output.extend(
                    _serialize_model_index_node(
                        value, indentation + 2, node_path + (key,)
                    )
                )

        return output

    output = []
    item_path = node_path + ("[]",)
    for item in node.items:
        if not isinstance(item, _ModelIndexMapping):
            scalar = item.canonical if isinstance(item, _ModelIndexScalar) else ""
            output.append(prefix + (f"- {scalar}" if scalar else "-"))

            continue

        entries = _ordered_model_index_entries(item.entries, item_path)
        if not entries:
            raise ValueError("model-index sequence mappings must not be empty")

        first_key, first_value = entries[0]
        item_prefix = prefix + f"- {first_key}:"
        if isinstance(first_value, _ModelIndexScalar):
            output.append(item_prefix + f" {first_value.canonical}")
        else:
            output.append(item_prefix)
            output.extend(
                _serialize_model_index_node(
                    first_value, indentation + 4, item_path + (first_key,)
                )
            )

        for key, value in entries[1:]:
            entry_prefix = " " * (indentation + 2) + f"{key}:"
            if isinstance(value, _ModelIndexScalar):
                output.append(entry_prefix + f" {value.canonical}")
            else:
                output.append(entry_prefix)
                output.extend(
                    _serialize_model_index_node(
                        value, indentation + 4, item_path + (key,)
                    )
                )

    return output


def _node_mapping_value(node: _ModelIndexNode, key: str) -> _ModelIndexNode | None:
    if not isinstance(node, _ModelIndexMapping):
        return None

    return next((value for entry_key, value in node.entries if entry_key == key), None)


def _model_index_tasks(path: Path, node: _ModelIndexNode) -> tuple[str, ...]:
    model_index = _node_mapping_value(node, "model-index")
    if not isinstance(model_index, _ModelIndexSequence):
        raise _unsupported_model_index_format(path)

    task_types: list[str] = []
    for model in model_index.items:
        name = _node_mapping_value(model, "name")
        if (
            not isinstance(name, _ModelIndexScalar)
            or name.kind != "string"
            or not name.value
        ):
            raise _unsupported_model_index_format(path)

        results = _node_mapping_value(model, "results")
        if not isinstance(results, _ModelIndexSequence):
            raise _unsupported_model_index_format(path)

        for result in results.items:
            task = _node_mapping_value(result, "task")
            task_type = _node_mapping_value(task, "type") if task is not None else None
            if (
                not isinstance(task_type, _ModelIndexScalar)
                or task_type.kind != "string"
                or not task_type.value
            ):
                raise _unsupported_model_index_format(path)

            value = task_type.value
            task_types.append(value)

    return tuple(task_types)


def _model_index_task_types(path: Path, frontmatter: str) -> tuple[str, ...]:
    """Read model-index tasks and reject any noncanonical block serialization."""
    block = _model_index_block(frontmatter)
    if block is None:
        return ()

    lines = _model_index_lines(path, block)
    node, consumed = _parse_model_index_block(path, lines)
    if consumed != len(lines):
        raise _unsupported_model_index_format(path)

    canonical_lines = _serialize_model_index_node(node, 0)
    canonical_block = "\n".join(canonical_lines)
    if block.endswith("\n"):
        canonical_block += "\n"

    if canonical_block != block:
        raise _unsupported_model_index_format(path)

    return _model_index_tasks(path, node)


def assert_pipeline_tag(path: Path, frontmatter: str, model_type: ModelType) -> None:
    """Require the model-card task metadata declared for this package."""
    pipeline_tag = _frontmatter_scalar(frontmatter, "pipeline_tag")
    category, task, _ = model_type
    expected_pipeline_tag = (
        task
        if category == IMAGE_GENERATION_CATEGORY
        else EXPECTED_PIPELINE_TAGS.get(path.parent.name, "other")
    )
    if pipeline_tag != expected_pipeline_tag:
        raise AssertionError(
            f"{path}: pipeline_tag must be {expected_pipeline_tag!r}, got {pipeline_tag!r}"
        )

    task_types = _model_index_task_types(path, frontmatter)
    has_model_index = "model-index:" in frontmatter
    if has_model_index and category == IMAGE_GENERATION_CATEGORY:
        if not task_types or any(task_type != task for task_type in task_types):
            raise AssertionError(
                f"{path}: model-index task.type values must match image generation "
                f"task {task!r}, got {task_types}"
            )

    if has_model_index and category != IMAGE_GENERATION_CATEGORY:
        if any(task_type in IMAGE_GENERATION_TASK_VALUES for task_type in task_types):
            raise AssertionError(
                f"{path}: model-index task.type must not be an image generation task, "
                f"got {task_types}"
            )

    expected_task = (
        task
        if category == IMAGE_GENERATION_CATEGORY
        else EXPECTED_MODEL_INDEX_TASKS.get(path.parent.name, "other")
    )
    if expected_task not in task_types and has_model_index:
        raise AssertionError(f"{path}: model-index task.type must be {expected_task!r}")


def assert_model_index_policy(path: Path, frontmatter: str) -> None:
    """Require model-index metadata only for weight-backed model cards."""
    has_model_index = "model-index:" in frontmatter
    if path.parent.name in NO_MODEL_WEIGHTS_SLUGS:
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
    model_type = parse_model_type(path, text)
    assert_pipeline_tag(path, frontmatter, model_type)
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

    if path.parent.name in NO_MODEL_WEIGHTS_SLUGS:
        required = [
            "git clone https://github.com/creative-graphic-design/design-generators.git",
            f"uv sync --package {_project_name(path.parent)}",
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


def _python_code_fences(text: str) -> list[tuple[int, str]]:
    fences: list[tuple[int, str]] = []
    opening_fence: str | None = None
    language = ""
    fence_start = 0
    code_lines: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if opening_fence is None:
            opening = _FENCE_OPENING_RE.match(line)
            if opening is None:
                continue

            opening_fence = opening.group("fence")
            info = opening.group("info").strip()
            language = info.split(maxsplit=1)[0].casefold() if info else ""
            fence_start = line_number
            code_lines = []
            continue

        closing = _FENCE_CLOSING_RE.match(line)
        if closing is None or closing.group("fence")[0] != opening_fence[0]:
            code_lines.append(line)
            continue

        closing_fence = closing.group("fence")
        if len(closing_fence) < len(opening_fence):
            code_lines.append(line)
            continue

        if language in {"python", "py"}:
            fences.append((fence_start, "\n".join(code_lines)))

        opening_fence = None
        language = ""

    if opening_fence is not None and language in {"python", "py"}:
        fences.append((fence_start, "\n".join(code_lines)))

    return fences


def _qualified_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id

    if isinstance(expression, ast.Attribute):
        prefix = _qualified_name(expression.value)
        if prefix is None:
            return None

        return f"{prefix}.{expression.attr}"

    return None


def _is_cpu_string_literal(expression: ast.AST) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, str)
        and (
            expression.value.casefold() == "cpu"
            or expression.value.casefold().startswith("cpu:")
        )
    )


def _allowed_cpu_literals(tree: ast.Module) -> set[int]:
    allowed_literals: set[int] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue

        function_name = _qualified_name(call.func)
        if function_name == "torch.Generator":
            allowed_keyword = "device"
        elif function_name == "torch.load":
            allowed_keyword = "map_location"
        else:
            continue

        for keyword in call.keywords:
            if (
                keyword.arg == allowed_keyword
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "cpu"
            ):
                allowed_literals.add(id(keyword.value))

    return allowed_literals


def assert_no_cpu_placement(path: Path, text: str) -> None:
    """Reject CPU string literals and .cpu() calls in Python fences."""
    for fence_start, code in _python_code_fences(text):
        normalized_code = re.sub(
            r"<[A-Za-z_][A-Za-z0-9_-]*>", "template_placeholder", code
        )
        try:
            syntax_tree = ast.parse(normalized_code)
        except SyntaxError as exc:
            line_number = fence_start + (exc.lineno or 1)
            raise AssertionError(
                f"{path}: cannot inspect invalid Python code at line {line_number}"
            ) from exc

        allowed_cpu_literals = _allowed_cpu_literals(syntax_tree)
        for node in ast.walk(syntax_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "cpu"
            ):
                line_number = fence_start + node.lineno
                raise AssertionError(
                    f"{path}: CPU placement with .cpu() at line {line_number} "
                    "is prohibited"
                )
            elif (
                isinstance(node, ast.Constant)
                and _is_cpu_string_literal(node)
                and id(node) not in allowed_cpu_literals
            ):
                line_number = fence_start + node.lineno
                raise AssertionError(
                    f"{path}: CPU string literal in Python code at line "
                    f"{line_number} is prohibited here; only direct "
                    'torch.Generator(device="cpu") and '
                    'torch.load(..., map_location="cpu") arguments are allowed'
                )


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
