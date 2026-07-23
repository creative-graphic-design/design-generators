"""Validate model README model-card and reproducibility contracts."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_REPO_URL = "https://github.com/creative-graphic-design/design-generators.git"
ROOT_REPO_BLOB_URL = (
    "https://github.com/creative-graphic-design/design-generators/blob/main/"
)
ROOT_LIBRARY_BADGE_COLORS = {
    "laygen": "2f80ed",
    "posgen": "00a88f",
    "traingen": "27ae60",
    "traingen-parity": "9b51e0",
}
ROOT_MODEL_TABLE_HEADER = [
    "Model",
    "Venue",
    "Ckpt",
    "Train",
]
MODEL_MEMBER_DIRS = sorted(
    path.parent for path in (REPO_ROOT / "models").glob("*/pyproject.toml")
)
MODEL_READMES = [member_dir / "README.md" for member_dir in MODEL_MEMBER_DIRS]
MODEL_REPRODUCING = [member_dir / "REPRODUCING.md" for member_dir in MODEL_MEMBER_DIRS]
LIB_MEMBER_DIRS = sorted(
    path.parent for path in (REPO_ROOT / "lib").glob("*/pyproject.toml")
)
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
LIB_READMES = sorted((REPO_ROOT / "lib").glob("*/README.md"))

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
    "cgb-dm": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/PKU-PosterLayout",
            "creative-graphic-design/CGL-Dataset",
        ],
    },
    "ds-gan": {
        "license": "other",
        "datasets": ["creative-graphic-design/PKU-PosterLayout"],
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
    "layout-detr": {
        "license": "apache-2.0",
        "datasets": ["Ad Banner vendor distribution"],
    },
    "layout-dm": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layout-fid": {
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
    "layoutvae": {
        "license": "mit",
        "datasets": ["creative-graphic-design/PubLayNet"],
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
    "postero": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/PKU-PosterLayout",
            "creative-graphic-design/CGL-Dataset",
        ],
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
    "cgb-dm": "CGB-DM",
    "ds-gan": "DS-GAN",
    "flex-dm": "Flex-DM",
    "housegan": "House-GAN",
    "lace": "LACE",
    "layousyn": "LayouSyn",
    "layout-corrector": "Layout-Corrector",
    "layout-detr": "LayoutDETR",
    "layout-dm": "LayoutDM",
    "layout-fid": "Layout FID",
    "layout-flow": "LayoutFlow",
    "layout-action": "LayoutAction",
    "layoutvae": "LayoutVAE",
    "layout-gpt": "LayoutGPT",
    "layout-transformer": "LayoutTransformer",
    "layoutdiffusion": "LayoutDiffusion",
    "layoutformerpp": "LayoutFormer++",
    "layoutganpp": "LayoutGAN++",
    "layoutprompter": "LayoutPrompter",
    "parse-then-place": "Parse-Then-Place",
    "postero": "PosterO",
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
    "layoutvae": "https://github.com/Layout-Generation/layout-generation",
    "layout-detr": "https://github.com/salesforce/LayoutDETR",
    "ralf": "https://github.com/CyberAgentAILab/RALF",
    "postero": "https://github.com/theKinsley/PosterO-CVPR2025",
    "ds-gan": "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023",
    "cgb-dm": "https://github.com/yuli0103/LayoutDiT",
    "smarttext": "https://github.com/intchous/SmartText",
    "flex-dm": "https://github.com/CyberAgentAILab/flex-dm",
    "housegan": "https://github.com/ennauata/housegan",
}

PROMPT_ONLY_SLUGS = {"layout-gpt", "layoutprompter", "postero"}
SHARED_PACKAGE_SUBDIRS = {
    "laygen": "lib/laygen",
    "posgen": "lib/posgen",
}
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


def _bash_fences(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)


def _project_metadata(member_dir: Path) -> dict[str, object]:
    return tomllib.loads((member_dir / "pyproject.toml").read_text(encoding="utf-8"))


def _project_name(member_dir: Path) -> str:
    project = _project_metadata(member_dir)["project"]
    if not isinstance(project, dict):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: missing [project]")
    name = project.get("name")
    if not isinstance(name, str):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: project.name missing")
    return name


def _normalize_root_repo_link(link: str) -> str:
    return link.removeprefix(ROOT_REPO_BLOB_URL)


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()


def _dependency_direct_name(requirement: str) -> str:
    return re.split(r"[<>=!~;]", requirement, maxsplit=1)[0].strip()


def _project_dependencies(member_dir: Path) -> list[str]:
    project = _project_metadata(member_dir)["project"]
    if not isinstance(project, dict):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: missing [project]")
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise AssertionError(
            f"{member_dir / 'pyproject.toml'}: project.dependencies must be a list"
        )
    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def _direct_requirement(package_name: str, subdirectory: str) -> str:
    return f"{package_name} @ git+{GIT_REPO_URL}#subdirectory={subdirectory}"


def _model_install_requirements(member_dir: Path) -> list[tuple[str, str]]:
    shared_dependencies: list[tuple[str, str]] = []
    for shared_name, subdirectory in SHARED_PACKAGE_SUBDIRS.items():
        for requirement in _project_dependencies(member_dir):
            if _dependency_name(requirement) == shared_name:
                shared_dependencies.append(
                    (_dependency_direct_name(requirement), subdirectory)
                )
                break
    slug = member_dir.name
    return [*shared_dependencies, (_project_name(member_dir), f"models/{slug}")]


def _pip_install_snippet(requirements: list[tuple[str, str]]) -> str:
    direct_requirements = [
        _direct_requirement(package_name, subdirectory)
        for package_name, subdirectory in requirements
    ]
    if len(direct_requirements) == 1:
        return f'pip install "{direct_requirements[0]}"'
    lines = ["pip install \\"]
    for index, requirement in enumerate(direct_requirements):
        suffix = " \\" if index < len(direct_requirements) - 1 else ""
        lines.append(f'  "{requirement}"{suffix}')
    return "\n".join(lines)


def _assert_pip_install_snippet(
    path: Path,
    section: str,
    requirements: list[tuple[str, str]],
    section_label: str,
) -> None:
    expected_requirements = [
        _direct_requirement(package_name, subdirectory)
        for package_name, subdirectory in requirements
    ]
    for fence in _bash_fences(section):
        if "pip install" in fence and all(
            requirement in fence for requirement in expected_requirements
        ):
            return

    expected_example = f"```bash\n{_pip_install_snippet(requirements)}\n```"
    missing = [
        requirement
        for requirement in expected_requirements
        if requirement not in section
    ]
    raise AssertionError(
        f"{path}: {section_label} must include a pip install snippet with "
        f"the package direct URL and required shared package direct URLs. "
        f"Missing: {missing}. Expected example:\n{expected_example}"
    )


def _assert_model_pip_install_snippet(path: Path, text: str) -> None:
    section = _section(text, "## How to Get Started with the Model")
    _assert_pip_install_snippet(
        path,
        section,
        _model_install_requirements(path.parent),
        "How to Get Started",
    )


def _assert_library_pip_install_snippet(path: Path, text: str) -> None:
    section = _section(text, "## Install")
    member_dir = path.parent
    _assert_pip_install_snippet(
        path,
        section,
        [(_project_name(member_dir), f"lib/{member_dir.name}")],
        "Install",
    )


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
    badge = re.search(r"!\[vendor-parity\]\([^)]*[?&]message=([^&)]*)", text)
    if badge is None:
        raise AssertionError(f"{path}: missing vendor-parity badge")
    expected = (
        "not-run"
        if "not run" in section
        else "tolerance-verified"
        if _parity_requires_tolerance(section)
        else "bit-exact"
    )
    actual = badge.group(1)
    accepted = {
        expected,
        expected.replace("--", "-"),
    }
    if actual not in accepted:
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
        or _semantic_badge_label(alt, query_label) != label
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
        (message, _normalize_root_repo_link(link))
        for message, link in _linked_static_badges(checkpoint_cell, "checkpoint")
    ]
    if checkpoint_badges != [("ckpt", expected_checkpoint_link)]:
        raise AssertionError(
            f"{path}: package {slug} must use one linked checkpoint reproduction badge"
        )

    training_badges = [
        (message, _normalize_root_repo_link(link))
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


def _assert_root_model_badge_count(path: Path, expected_count: int) -> None:
    text = path.read_text(encoding="utf-8")
    messages = _badge_messages(text, "models")
    if messages != [str(expected_count)]:
        raise AssertionError(
            f"{path}: models badge {messages} != workspace model member count {expected_count}"
        )


def _root_model_slugs(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    table_lines = _root_packages_table_lines(text)
    if _split_markdown_table_row(
        table_lines[0]
    ) != ROOT_MODEL_TABLE_HEADER or not _is_markdown_table_separator(
        table_lines[1], len(ROOT_MODEL_TABLE_HEADER)
    ):
        raise AssertionError(
            f"{path}: Models table must use {', '.join(ROOT_MODEL_TABLE_HEADER)}"
        )
    slugs: set[str] = set()
    for line in table_lines[2:]:
        cells = _split_markdown_table_row(line)
        if len(cells) != len(ROOT_MODEL_TABLE_HEADER):
            raise AssertionError(f"{path}: malformed Models table row: {line}")
        (
            method_cell,
            venue_cell,
            checkpoint_cell,
            training_cell,
        ) = cells
        model_link = re.fullmatch(r"\[`([^`\]]+)`\]\(([^)]+)\)", method_cell)
        if model_link is None:
            raise AssertionError(
                f"{path}: Model cell must be a `Model` markdown link: {line}"
            )
        model_name = model_link.group(1)
        normalized_model_link = _normalize_root_repo_link(model_link.group(2))
        slug_match = re.fullmatch(r"models/([^/)]+)/README\.md", normalized_model_link)
        if slug_match is None:
            raise AssertionError(
                f"{path}: Model cell must link models/<slug>/README.md: {line}"
            )
        slug = slug_match.group(1)
        expected_name = EXPECTED_MODEL_NAMES.get(slug)
        if expected_name is not None and model_name != expected_name:
            raise AssertionError(
                f"{path}: model link text {model_name!r} != {expected_name!r}"
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


def _assert_root_libraries_table_matches_members(path: Path) -> None:
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
        normalized_library_link = _normalize_root_repo_link(library_link)
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
    member_slugs = _library_member_slugs()
    missing = sorted(member_slugs - root_slugs)
    extra = sorted(root_slugs - member_slugs)
    if missing or extra:
        raise AssertionError(
            f"root README Libraries table mismatch: missing={missing}, extra={extra}"
        )


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


def _assert_root_models_table_matches_members(root_slugs: set[str]) -> None:
    member_slugs = _model_member_slugs()
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
    text = _without_badges(
        _without_frontmatter_and_code(path.read_text(encoding="utf-8"))
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
            if (
                _inline_code_span_at(text, match.start())
                == required_code_spans[library]
            ):
                continue
            raise AssertionError(
                f"{path}: use code-form {required_code_spans[library]} for library names"
            )


def check() -> None:
    _assert_model_doc_sets()
    root_slugs = _root_model_slugs(REPO_ROOT / "README.md")
    _assert_root_model_badge_count(REPO_ROOT / "README.md", len(MODEL_MEMBER_DIRS))
    _assert_root_models_table_matches_members(root_slugs)
    _assert_root_libraries_table_matches_members(REPO_ROOT / "README.md")
    _assert_generated_docs_targets_match_members()
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
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Model README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
