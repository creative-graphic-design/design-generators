"""Validate model README model-card and reproducibility contracts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_READMES = sorted((REPO_ROOT / "models").glob("*/README.md"))
README_LINK_CONTRACTS = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "lib").glob("*/README.md")),
    *MODEL_READMES,
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
    "## Parity Results",
    "## Reproducibility",
]

BANNED_PATTERNS = [
    r"GEN_AI_PROXY_PAT",
    r"genai[-_]?gateway",
    r"example-openai-compatible-endpoint",
    r"sk-[A-Za-z0-9]{16,}",
    r"(?<![A-Za-z0-9_.-])/tmp/",
    r"creative-graphic-design/(rico|rico25|publaynet)\b",
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
    "lace": {
        "license": "mit",
        "datasets": [
            "creative-graphic-design/Rico",
            "creative-graphic-design/PubLayNet",
        ],
    },
    "layousyn": {"license": "cc-by-nc-4.0", "datasets": []},
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
    "layout-gpt": {"license": "other", "datasets": []},
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
        "license": "other",
        "datasets": [
            "creative-graphic-design/PubLayNet",
            "creative-graphic-design/Rico",
        ],
    },
    "parse-then-place": {
        "license": "mit",
        "datasets": ["creative-graphic-design/Rico"],
    },
}


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
        for match in re.finditer(r"!?\[[^\]]*\]\([^)]*\)", text)
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


def _assert_code_fences_tagged(path: Path, text: str) -> None:
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
    section = _section(text, "## Parity Results")
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


def _assert_reproducibility_commands(path: Path, text: str) -> None:
    section = _section(text, "## Reproducibility")
    if "uv run --package " not in section:
        raise AssertionError(
            f"{path}: Reproducibility must contain uv package commands"
        )
    bad_command_shapes = ["python scripts/", "cd models/", "../.cache", "/tmp/"]
    for bad in bad_command_shapes:
        if bad in section:
            raise AssertionError(
                f"{path}: stale reproducibility command shape contains {bad!r}"
            )


def _assert_banned_patterns(path: Path, text: str) -> None:
    for pattern in BANNED_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raise AssertionError(
                f"{path}: banned README content matched {pattern!r}: {match.group(0)!r}"
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


def check() -> None:
    if len(MODEL_READMES) != 12:
        raise AssertionError(f"expected 12 model READMEs, found {len(MODEL_READMES)}")
    for path in MODEL_READMES:
        text = path.read_text(encoding="utf-8")
        _assert_frontmatter(path, text)
        _assert_expected_frontmatter(path, text)
        _assert_heading_order(path, text)
        _assert_code_fences_tagged(path, text)
        _assert_parity_table(path, text)
        _assert_reproducibility_commands(path, text)
        _assert_banned_patterns(path, text)
    for path in README_LINK_CONTRACTS:
        _assert_linked_first_reference_policy(path)


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
