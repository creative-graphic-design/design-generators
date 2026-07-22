"""Validate model README model-card and reproducibility contracts."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_MEMBER_DIRS = sorted(
    path.parent for path in (REPO_ROOT / "models").glob("*/pyproject.toml")
)
MODEL_READMES = [member_dir / "README.md" for member_dir in MODEL_MEMBER_DIRS]
MODEL_REPRODUCING = [member_dir / "REPRODUCING.md" for member_dir in MODEL_MEMBER_DIRS]
README_LINK_CONTRACTS = [
    REPO_ROOT / "README.md",
    REPO_ROOT
    / ".claude"
    / "skills"
    / "model-conversion"
    / "references"
    / "model-readme-template.md",
    *sorted((REPO_ROOT / "lib").glob("*/README.md")),
    *MODEL_READMES,
    *MODEL_REPRODUCING,
]
README_POLICY_DOCS = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "lib").glob("*/README.md")),
]

REQUIRED_HEADINGS = [
    "# Model Card for ",
    "## Model Details",
    "### Model Description",
    "### Model Sources",
    "## Supported Checkpoints",
    "## Uses",
    "### Direct Use",
    "### Downstream Use",
    "### Out-of-Scope Use",
    "## Bias, Risks, and Limitations",
    "### Recommendations",
    "## How to Get Started with the Model",
    "## Training Details",
    "### Training Data",
    "### Training Procedure",
    "## Evaluation",
    "### Parity Results",
    "## Reproducibility",
]

BANNED_PATTERNS = [
    r"GEN_AI_PROXY_PAT",
    r"genai[-_]?gateway",
    r"example-openai-compatible-endpoint",
    r"sk-[A-Za-z0-9]{16,}",
    r"(?<![A-Za-z0-9_.-])/tmp/",
    r"creative-graphic-design/(rico|rico25|publaynet)\b",
    r"original upstream authors; see Model Sources",
    r"is packaged for the",
    r"for the workspace",
    r"badge is",
    r"tracked in issue",
    r"verification is tracked",
    r"not recorded in the current README",
    r"documentation gap",
    r"preserved from the original README",
    r"The package preserves the upstream",
    r"preserves the upstream architecture",
    r"needed for conversion and inference",
    r"This package provides",
    r"Regular package checks",
    r"current package coverage",
    r"coverage command",
    r"for this PR",
    r"table reports",
    r"table below",
    r"section above",
    r"this README describes",
]

LINK_REQUIRED_DATASET_IDS = [
    "creative-graphic-design/Rico",
    "creative-graphic-design/PubLayNet",
    "creative-graphic-design/magazine",
    "cyberagent/crello",
]

EXPECTED_FRONTMATTER = {
    "coarse-to-fine": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "ds-gan": {
        "license": "other",
        "datasets": ["PosterLayout"],
    },
    "flex-dm": {
        "license": "apache-2.0",
        "datasets": [
            "cyberagent/crello",
            "creative-graphic-design/Rico",
        ],
    },
    "housegan": {
        "license": "gpl-3.0",
        "datasets": ["housegan-floorplan-vectorized"],
    },
    "lace": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "RICO13",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layousyn": {
        "license": "cc-by-nc-4.0",
        "datasets": ["GRIT", "COCO-grounded"],
    },
    "layout-corrector": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
            "cyberagent/crello",
        ],
    },
    "layout-dm": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layout-flow": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layout-action": {
        "license": "other",
        "datasets": [
            "RICO13",
            "creative-graphic-design/PubLayNet",
            "InfoPPT",
        ],
    },
    "layout-gpt": {"license": "mit", "datasets": ["NSR-1K"]},
    "layout-transformer": {"license": "other", "datasets": ["COCO", "VG-MSDN"]},
    "layoutdiffusion": {
        "license": "other",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layoutformerpp": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layoutganpp": {
        "license": "agpl-3.0",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
            "creative-graphic-design/magazine",
        ],
    },
    "layoutprompter": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/PubLayNet",
            "creative-graphic-design/Rico",
            "PosterLayout",
        ],
    },
    "parse-then-place": {
        "license": "mit",
        "datasets": ["creative-graphic-design/Rico", "Web"],
    },
    "ralf": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/CGL-Dataset",
            "creative-graphic-design/PKU-PosterLayout",
        ],
    },
    "smarttext": {"license": "other", "datasets": ["SmartText demo"]},
}


EXPECTED_MODEL_NAMES = {
    "coarse-to-fine": "Coarse-to-Fine",
    "ds-gan": "DS-GAN",
    "flex-dm": "Flex-DM",
    "housegan": "House-GAN",
    "lace": "LACE",
    "layousyn": "LayouSyn",
    "layout-corrector": "Layout-Corrector",
    "layout-dm": "LayoutDM",
    "layout-flow": "LayoutFlow",
    "layout-action": "LayoutAction",
    "layout-gpt": "LayoutGPT",
    "layout-transformer": "LayoutTransformer",
    "layoutdiffusion": "LayoutDiffusion",
    "layoutformerpp": "LayoutFormer++",
    "layoutganpp": "LayoutGAN++",
    "layoutprompter": "LayoutPrompter",
    "parse-then-place": "Parse-Then-Place",
    "ralf": "RALF",
    "smarttext": "SmartText",
}

EXPECTED_REPOSITORY_LINKS = {
    "layousyn": "https://github.com/mlpc-ucsd/Lay-Your-Scene",
    "layout-gpt": "https://github.com/UCSB-AI/LayoutGPT",
    "layoutdiffusion": "https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion",
    "layout-transformer": "https://github.com/davidhalladay/LayoutTransformer",
    "layout-action": "https://github.com/BERYLSHEEP/LayoutActionProject",
    "layoutganpp": "https://github.com/ktrk115/const_layout",
    "ralf": "https://github.com/CyberAgentAILab/RALF",
    "ds-gan": "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023",
    "smarttext": "https://github.com/chenqi008/SmartText",
    "flex-dm": "https://github.com/CyberAgentAILab/flex-dm",
    "housegan": "https://github.com/ennauata/housegan",
}

PROMPT_ONLY_SLUGS = {"layout-gpt", "layoutprompter"}
PROMPT_ONLY_STALE_PHRASES = [
    "CUDA_VISIBLE_DEVICES",
    "converted behavior follows the upstream checkpoints",
    "converted checkpoint",
    "converted checkpoints",
    "converted checkpoint directories",
    "Conversion and parity costs",
    "CUDA is required",
    "heavyweight vendor parity",
]


def _section(text: str, heading: str) -> str:
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


def _markdown_link_spans(text: str) -> list[range]:
    return [
        range(match.start(), match.end())
        for match in re.finditer(
            r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!?\[[^\]]*\]\([^)]*\)", text
        )
    ]


def _in_any_span(position: int, spans: list[range]) -> bool:
    return any(position in span for span in spans)


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


def _badge_messages(text: str, label: str) -> list[str]:
    messages: list[str] = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        parsed = urlparse(match.group(1))
        if parsed.netloc != "img.shields.io" or parsed.path != "/static/v1":
            continue
        query = parse_qs(parsed.query)
        if query.get("label") == [label] and "message" in query:
            messages.append(unquote(query["message"][0]).replace("--", "-"))
    return messages


def _model_member_slugs() -> set[str]:
    return {member_dir.name for member_dir in MODEL_MEMBER_DIRS}


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


def _assert_runtime_contract(
    path: Path, text: str, root_runtime_by_slug: dict[str, str]
) -> None:
    slug = path.parent.name
    frontmatter_library = _frontmatter_scalar(_frontmatter(text), "library_name")
    base_badges = _badge_messages(text, "base")
    if len(base_badges) != 1:
        raise AssertionError(
            f"{path}: expected exactly one base badge, found {base_badges}"
        )
    base_library = base_badges[0]
    root_library = root_runtime_by_slug.get(slug)
    if root_library is None:
        raise AssertionError(f"{path}: root Models table missing {slug}")
    values = {
        "frontmatter library_name": frontmatter_library,
        "base badge": base_library,
        "root Models Runtime": root_library,
    }
    if len(set(values.values())) != 1:
        raise AssertionError(
            f"{path}: runtime mismatch across README surfaces {values}"
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


def _assert_citation_bibtex(path: Path, text: str) -> None:
    section = _section(text, "## Citation")
    # Coordinator approval is required before adding exceptions to this bibtex
    # requirement; README normalization must preserve citation metadata.
    if "```bibtex" not in section:
        raise AssertionError(f"{path}: Citation must contain a bibtex code fence")


def _nonzero_number(text: str) -> bool:
    try:
        return float(text) != 0
    except ValueError:
        return False


def _parity_requires_tolerance(section: str) -> bool:
    for match in re.finditer(r"\b[ra]tol\s*=?\s*`?([0-9.eE+-]+)`?", section):
        if _nonzero_number(match.group(1)):
            return True
    return False


def _assert_vendor_parity_badge(path: Path, text: str) -> None:
    section = _section(text, "### Parity Results")
    badge = re.search(r"!\[vendor--parity\]\([^)]*[?&]message=([^&)]*)", text)
    if badge is None:
        raise AssertionError(f"{path}: missing vendor-parity badge")
    expected = (
        "not--run"
        if "not run" in section
        else "tolerance--verified"
        if _parity_requires_tolerance(section)
        else "bit--exact"
    )
    actual = badge.group(1)
    if actual != expected:
        raise AssertionError(
            f"{path}: vendor-parity badge {actual!r} does not match Parity Results; expected {expected!r}"
        )


def _assert_readme_reproducibility_link(path: Path, text: str) -> None:
    section = _section(text, "## Reproducibility")
    expected = (
        "https://github.com/creative-graphic-design/design-generators/blob/main/"
        f"models/{path.parent.name}/REPRODUCING.md"
    )
    if expected not in section:
        raise AssertionError(
            f"{path}: Reproducibility must link absolute REPRODUCING.md URL {expected}"
        )
    if "uv run --package " in section or "```" in section:
        raise AssertionError(
            f"{path}: README Reproducibility must be a short link, not a walkthrough"
        )


def _assert_reproducing_commands(path: Path, text: str) -> None:
    if "uv run --package " not in text:
        raise AssertionError(f"{path}: REPRODUCING.md must contain uv package commands")
    lower = text.lower()
    bad_command_shapes = ["python scripts/", "cd models/", "../.cache", "/tmp/"]
    for bad in bad_command_shapes:
        if bad in text:
            raise AssertionError(
                f"{path}: stale reproducibility command shape contains {bad!r}"
            )
    required_terms: list[str | tuple[str, ...]] = [
        "Workflow order:",
        "download",
        ("reference", "golden"),
        "pytest",
    ]
    if path.parent.name in PROMPT_ONLY_SLUGS:
        required_terms.extend([("prompt configuration", "save_pretrained"), "smoke"])
        if "convert checkpoints" in lower:
            raise AssertionError(
                f"{path}: prompt-only REPRODUCING.md must not mention converting checkpoints"
            )
    else:
        required_terms.extend(["convert", "from_pretrained"])
    for term in required_terms:
        alternatives = (term,) if isinstance(term, str) else term
        position = max(lower.find(alternative.lower()) for alternative in alternatives)
        if position == -1:
            raise AssertionError(f"{path}: missing reproducibility step {term!r}")
    if path.parent.name in {"coarse-to-fine", "layoutganpp"}:
        expected = (
            "Workflow order: download assets, generate references, convert checkpoints, "
            "run parity checks, then smoke-test local loading."
        )
        if expected not in text:
            raise AssertionError(
                f"{path}: reproducibility workflow must state reference -> conversion -> parity order"
            )


def _assert_banned_patterns(path: Path, text: str) -> None:
    for pattern in BANNED_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raise AssertionError(
                f"{path}: banned README content matched {pattern!r}: {match.group(0)!r}"
            )


def _root_packages_table_lines(text: str) -> list[str]:
    marker = "## Models\n\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n\n", start)
    return text[start:end].splitlines()


def _assert_root_model_badge_count(path: Path, expected_count: int) -> None:
    text = path.read_text(encoding="utf-8")
    messages = _badge_messages(text, "models")
    if messages != [str(expected_count)]:
        raise AssertionError(
            f"{path}: models badge {messages} != workspace model member count {expected_count}"
        )


def _root_packages_runtime_by_slug(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    table_lines = _root_packages_table_lines(text)
    expected_header = "| Model | Method | Runtime | Primary datasets | Weights |"
    if table_lines[:2] != [expected_header, "| --- | --- | --- | --- | --- |"]:
        raise AssertionError(f"{path}: Models table must use a Weights column")
    runtime_by_slug: dict[str, str] = {}
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            raise AssertionError(f"{path}: malformed Packages table row: {line}")
        package_cell, _method_cell, runtime_cell, _datasets_cell, weights_cell = cells
        slug_match = re.search(r"`models/([^`]+)`", package_cell)
        if slug_match is None:
            raise AssertionError(f"{path}: package row missing models/<slug>: {line}")
        slug = slug_match.group(1)
        if runtime_cell not in {"`transformers`", "`diffusers`", "`pydantic-ai`"}:
            raise AssertionError(
                f"{path}: Models table Runtime cell must use code-form library name"
            )
        runtime_by_slug[slug] = runtime_cell.strip("`")
        if "documented" in weights_cell.lower():
            raise AssertionError(
                f"{path}: Packages table Weights column must not use status wording"
            )
        if slug in PROMPT_ONLY_SLUGS:
            if weights_cell != "none (prompt-based)":
                raise AssertionError(
                    f"{path}: prompt-only package {slug} must state no weights"
                )
            continue
        expected_link = f"[REPRODUCING.md](models/{slug}/REPRODUCING.md)"
        if weights_cell != f"convert locally ({expected_link})":
            raise AssertionError(
                f"{path}: weight-backed package {slug} must link conversion steps"
            )
    return runtime_by_slug


def _assert_model_doc_sets() -> None:
    member_slugs = _model_member_slugs()
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


def _assert_root_models_table_matches_members(
    root_runtime_by_slug: dict[str, str],
) -> None:
    member_slugs = _model_member_slugs()
    root_slugs = set(root_runtime_by_slug)
    missing = sorted(member_slugs - root_slugs)
    extra = sorted(root_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"root README Models table mismatch: missing={missing}, extra={extra}"
        )


def _assert_generated_docs_targets_match_members() -> None:
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "gen_ref_pages.py"
    spec = importlib.util.spec_from_file_location("gen_ref_pages", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    docs_model_slugs = {
        package.member_dir.name
        for package in module.discover_api_packages()
        if package.group == "Models"
    }
    member_slugs = _model_member_slugs()
    missing = sorted(member_slugs - docs_model_slugs)
    extra = sorted(docs_model_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"generated docs model targets mismatch: missing={missing}, extra={extra}"
        )


def _assert_linked_first_reference_policy(path: Path) -> None:
    text = _without_frontmatter_and_code(path.read_text(encoding="utf-8"))
    spans = _markdown_link_spans(text)
    for dataset_id in LINK_REQUIRED_DATASET_IDS:
        for match in re.finditer(re.escape(dataset_id), text):
            if not _in_any_span(match.start(), spans):
                raise AssertionError(f"{path}: dataset id must be linked: {dataset_id}")

    for match in re.finditer(r"\barXiv\s+\d{4}\.\d{4,5}\b", text):
        if not _in_any_span(match.start(), spans):
            raise AssertionError(f"{path}: arXiv id must be linked: {match.group(0)}")

    for match in re.finditer(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}", text):
        if not _in_any_span(match.start(), spans):
            raise AssertionError(f"{path}: arXiv URL must be in a markdown link")


def _assert_library_name_style(path: Path) -> None:
    text = _without_frontmatter_and_code(path.read_text(encoding="utf-8"))
    banned_names = {
        "Transformers": "🤗 `transformers`",
        "Diffusers": "🤗 `diffusers`",
        "Pydantic AI": "`pydantic-ai`",
    }
    for name, replacement in banned_names.items():
        match = re.search(rf"(?<![`/\w]){re.escape(name)}(?![`/\w])", text)
        if match:
            raise AssertionError(
                f"{path}: use {replacement} instead of prose library name {name!r}"
            )
    emoji_mentions = re.findall(
        r"🤗\s+(?:\[`(?:transformers|diffusers)`\]\([^)]*\)|`(?:transformers|diffusers)`)",
        text,
    )
    if text.count("🤗") != len(emoji_mentions):
        raise AssertionError(
            f"{path}: 🤗 must annotate a transformers/diffusers library mention"
        )
    if len(emoji_mentions) > 1:
        raise AssertionError(
            f"{path}: 🤗 may appear only on the first Hugging Face library mention"
        )
    if re.search(r"🤗\s+(?:\[`pydantic-ai`\]\([^)]*\)|`pydantic-ai`)", text):
        raise AssertionError(f"{path}: pydantic-ai mentions must not use 🤗")
    spans = _markdown_link_spans(text)
    for library in ("transformers", "diffusers", "pydantic-ai"):
        for match in re.finditer(
            rf"(?<![`/\w=-]){re.escape(library)}(?![`/\w-])", text
        ):
            if not _in_any_span(match.start(), spans):
                raise AssertionError(
                    f"{path}: use code-form `{library}` for library names"
                )


def check() -> None:
    _assert_model_doc_sets()
    root_runtime_by_slug = _root_packages_runtime_by_slug(REPO_ROOT / "README.md")
    _assert_root_model_badge_count(REPO_ROOT / "README.md", len(MODEL_MEMBER_DIRS))
    _assert_root_models_table_matches_members(root_runtime_by_slug)
    _assert_generated_docs_targets_match_members()
    for path in MODEL_READMES:
        text = path.read_text(encoding="utf-8")
        _assert_frontmatter(path, text)
        _assert_expected_frontmatter(path, text)
        _assert_runtime_contract(path, text, root_runtime_by_slug)
        _assert_heading_order(path, text)
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
    for path in README_POLICY_DOCS:
        text = path.read_text(encoding="utf-8")
        _assert_code_fences_tagged(path, text)
        _assert_banned_patterns(path, text)


def main() -> int:
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Model README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
