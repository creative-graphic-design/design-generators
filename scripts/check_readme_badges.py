"""Validate README badge placement, grammar, and order."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_REPO_BLOB_URL = (
    "https://github.com/creative-graphic-design/design-generators/blob/main/"
)
BADGE_RE = re.compile(
    r"(?P<linked>\[)?!\[(?P<alt>[^\]]*)\]\((?P<url>https://(?:img\.shields\.io|codecov\.io)/[^)]+)\)"
    r"(?:\]\((?P<link>[^)]+)\))?"
)
BADGE_DOCS = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "lib").glob("*/README.md")),
    *sorted((REPO_ROOT / "models").glob("*/README.md")),
    *sorted((REPO_ROOT / "models").glob("*/REPRODUCING.md")),
]

ROOT_ORDER = ["CI", "codecov", "docs", "license", "python", "uv", "models"]
LAYGEN_ORDER = ["package", "license", "python", "core", "extras", "docs"]
POSGEN_ORDER = ["package", "license", "python", "runtime", "status", "docs"]
MODEL_ORDER = [
    ("paper", "arXiv", "OpenReview", "DOI"),
    ("venue",),
    ("license",),
    ("base",),
    ("dataset", "task"),
    ("vendor-parity",),
    ("hub",),
]

VERIFIED_SIMPLE_ICON_SLUGS = {
    "adobe",
    "android",
    "apache",
    "arxiv",
    "canva",
    "creativecommons",
    "doi",
    "figma",
    "githubactions",
    "googledocs",
    "googlenews",
    "googlescholar",
    "huggingface",
    "opensourceinitiative",
    "pydantic",
    "python",
    "readthedocs",
    "uv",
}
RUNTIME_BADGE_POLICY = {
    "transformers": ("yellow", "huggingface"),
    "diffusers": ("red", "huggingface"),
    "pydantic-ai": ("violet", "pydantic"),
}
DATASET_BADGE_POLICY = {
    "RICO25": ("blue", "android"),
    "RICO13": ("blue", "android"),
    "PubLayNet": ("informational", "googledocs"),
    "Crello": ("violet", "canva"),
    "Magazine": ("orange", "googlenews"),
    "CGL": ("red", "adobe"),
    "PKU": ("blueviolet", "figma"),
    "PKU-PosterLayout": ("blueviolet", "figma"),
}
ROOT_VENUE_BADGE_COLORS = {
    "AAAI 2022": "2f5f8f",
    "AAAI 2023": "2f5f8f",
    "ACM MM 2021": "0085ca",
    "CVPR 2021": "0076a8",
    "CVPR 2023": "0076a8",
    "CVPR 2024": "0076a8",
    "CVPR 2025": "0076a8",
    "ECCV 2020": "009688",
    "ECCV 2024": "009688",
    "ICCV 2019": "0066cc",
    "ICCV 2023": "0066cc",
    "ICCV 2025": "0066cc",
    "ICLR 2024": "00a88f",
    "NeurIPS 2023": "4b2e83",
    "TMM 2021": "00629b",
}
ROOT_TASK_LEGEND_BADGE_COLORS = {
    "content-agnostic": "2f80ed",
    "content-aware": "eb5757",
    "mixed": "9b51e0",
}
ROOT_LIBRARY_BADGE_COLORS = {
    "laygen": "2f80ed",
    "posgen": "00a88f",
    "traingen": "27ae60",
    "traingen-parity": "9b51e0",
}

DOCS_URL = "https://creative-graphic-design.github.io/design-generators/"
CODECOV_URL = "https://codecov.io/gh/creative-graphic-design/design-generators"
CODECOV_BADGE_URL = f"{CODECOV_URL}/graph/badge.svg?token=482TEUSZJ5"
DATASET_LINKS = {
    "RICO25": "https://huggingface.co/datasets/creative-graphic-design/Rico",
    "RICO13": "https://huggingface.co/datasets/creative-graphic-design/Rico",
    "PubLayNet": "https://huggingface.co/datasets/creative-graphic-design/PubLayNet",
    "Crello": "https://huggingface.co/datasets/cyberagent/crello",
    "Magazine": "https://huggingface.co/datasets/creative-graphic-design/magazine",
    "CGL": "https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset",
    "PKU": "https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout",
    "PKU-PosterLayout": "https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout",
}
PAPER_LINKS = {
    (
        "paper",
        "CVPR 2023",
    ): "https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html",
    (
        "paper",
        "CVPR 2025",
    ): "https://openaccess.thecvf.com/content/CVPR2025/html/Hsu_PosterO_Structuring_Layout_Trees_to_Enable_Language_Models_in_Generalized_CVPR_2025_paper.html",
    ("paper", "AAAI"): "https://ojs.aaai.org/index.php/AAAI/article/view/19994",
    (
        "paper",
        "CVPR 2021",
    ): "https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html",
    (
        "paper",
        "AAAI 2023",
    ): "https://ojs.aaai.org/index.php/AAAI/article/view/26277",
    ("OpenReview", "kJ0qp9Xdsh"): "https://openreview.net/forum?id=kJ0qp9Xdsh",
    ("arXiv", "2208.08037"): "https://arxiv.org/abs/2208.08037",
    ("arXiv", "2212.09877"): "https://arxiv.org/abs/2212.09877",
    ("arXiv", "2303.08137"): "https://arxiv.org/abs/2303.08137",
    ("arXiv", "2303.18248"): "https://arxiv.org/abs/2303.18248",
    ("arXiv", "2303.11589"): "https://arxiv.org/abs/2303.11589",
    ("arXiv", "2305.15393"): "https://arxiv.org/abs/2305.15393",
    ("arXiv", "2308.12700"): "https://arxiv.org/abs/2308.12700",
    ("arXiv", "2311.06495"): "https://arxiv.org/abs/2311.06495",
    ("arXiv", "2311.13602"): "https://arxiv.org/abs/2311.13602",
    ("arXiv", "2403.18187"): "https://arxiv.org/abs/2403.18187",
    ("arXiv", "2409.16689"): "https://arxiv.org/abs/2409.16689",
    ("arXiv", "2505.04718"): "https://arxiv.org/abs/2505.04718",
    ("arXiv", "1907.10719"): "https://arxiv.org/abs/1907.10719",
    ("arXiv", "2003.06988"): "https://arxiv.org/abs/2003.06988",
    ("DOI", "10.1145/3474085.3475497"): "https://doi.org/10.1145/3474085.3475497",
    ("paper", "TMM 2021"): "https://ieeexplore.ieee.org/document/9520053",
}
HUB_LINKS = {
    "coarse-to-fine": "https://huggingface.co/creative-graphic-design/coarse-to-fine-rico25",
    "lace": "https://huggingface.co/creative-graphic-design/lace-publaynet",
    "layousyn": "https://huggingface.co/creative-graphic-design/layousyn-grit",
    "layout-corrector": "https://huggingface.co/creative-graphic-design/layout-corrector-rico25",
    "layout-dm": "https://huggingface.co/creative-graphic-design/layoutdm-rico25",
    "layout-flow": "https://huggingface.co/creative-graphic-design/layout-flow-rico25",
    "layoutdiffusion": "https://huggingface.co/creative-graphic-design/layoutdiffusion-rico25",
    "layoutformerpp": "https://huggingface.co/creative-graphic-design/layoutformerpp-rico25-label",
    "layoutganpp": "https://huggingface.co/creative-graphic-design/layoutganpp-rico",
    "parse-then-place": "https://huggingface.co/creative-graphic-design/parse-then-place-rico-finetune",
}


@dataclass(frozen=True)
class Badge:
    path: Path
    line: int
    alt: str
    url: str
    link: str | None
    label: str
    message: str | None
    color: str | None
    logo: str | None


def _is_root_readme(path: Path) -> bool:
    return path == REPO_ROOT / "README.md"


def _semantic_label(alt: str, query_label: str) -> str:
    alt_prefix, separator, _ = alt.partition(":")
    if query_label == ">" and (not separator or not alt_prefix):
        raise AssertionError(
            f"badge with label=> must use '<semantic label>: ...' alt text: {alt!r}"
        )
    semantic_alt_prefixes = {
        "checkpoint",
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
    return alt_prefix


def _allowed_logos(path: Path, label: str, message: str | None) -> set[str | None]:
    if label == "CI":
        return {"githubactions"}
    if label == "coverage":
        return {None}
    if label == "docs":
        return {"readthedocs"}
    if label == "license":
        if message == "Apache-2.0":
            return {"apache"}
        if message and message.startswith("CC-"):
            return {"creativecommons"}
        if message == "review-needed":
            return {None}
        return {"opensourceinitiative"}
    if label == "python":
        return {"python"}
    if label == "uv":
        return {"uv"}
    if label in {"models", "package", "core", "extras", "runtime", "status"}:
        return {None}
    if label == "arXiv":
        return {"arxiv"}
    if label == "DOI":
        return {"doi"}
    if label in {"paper", "OpenReview"}:
        return {None}
    if label == "venue":
        return {None}
    if label == "model":
        return {None}
    if label == "library":
        return {None}
    if label == "task":
        return {None}
    if label == "framework":
        policy = RUNTIME_BADGE_POLICY.get(message or "")
        return {policy[1]} if policy else set()
    if label == "base":
        if _is_root_readme(path):
            policy = RUNTIME_BADGE_POLICY.get(message or "")
            return {policy[1]} if policy else set()
        return {"pydantic"} if message == "pydantic-ai" else {"huggingface"}
    if label == "dataset":
        if _is_root_readme(path):
            return {None}
        return {None, "huggingface"}
    if label == "training":
        return {None}
    if label == "checkpoint":
        return {None}
    if label == "hub":
        if message == "n/a":
            return {None}
        return {"huggingface"}
    if label == "vendor-parity":
        return {None}
    raise AssertionError(f"no badge logo rule for label={label!r} message={message!r}")


def _expected_color(path: Path, label: str, message: str | None) -> str | None:
    if label == "CI":
        return None
    if label == "coverage":
        return None
    if label == "docs":
        return None
    if label == "license":
        if message in {"Apache-2.0", "MIT"}:
            return "green"
        if message in {"AGPL-3.0", "CC-BY-NC-4.0", "GPL-3.0"}:
            return "orange"
        if message == "review-needed":
            return "yellow"
    if label in {"python", "package", "paper", "OpenReview", "DOI"}:
        return "blue"
    if label == "model":
        if _is_root_readme(path):
            return None
        return "blue"
    if label == "library":
        if _is_root_readme(path) and message in ROOT_LIBRARY_BADGE_COLORS:
            return ROOT_LIBRARY_BADGE_COLORS[message]
        return "blue"
    if label == "task":
        if _is_root_readme(path) and message in ROOT_TASK_LEGEND_BADGE_COLORS:
            return ROOT_TASK_LEGEND_BADGE_COLORS[message]
        return "purple"
    if label == "framework" and message:
        policy = RUNTIME_BADGE_POLICY.get(message)
        if policy is not None:
            return policy[0]
    if label == "base" and message and _is_root_readme(path):
        policy = RUNTIME_BADGE_POLICY.get(message)
        if policy is not None:
            return policy[0]
    if label == "base":
        return "blue"
    if label == "arXiv":
        return "b31b1b"
    if label in {"uv", "extras", "runtime"}:
        return "informational"
    if label == "dataset":
        if _is_root_readme(path):
            return None
        return "informational"
    if label == "venue":
        if _is_root_readme(path) and message in ROOT_VENUE_BADGE_COLORS:
            return ROOT_VENUE_BADGE_COLORS[message]
        return "purple"
    if label == "models":
        return "purple"
    if label == "vendor-parity" and message == "not-run":
        return "lightgrey"
    if label == "core" or label == "vendor-parity":
        return "success"
    if label == "status":
        return "lightgrey"
    if label == "checkpoint" and message == "ckpt":
        return "success"
    if label == "training":
        if message == "train":
            return "success"
        if message == "n/a":
            return "lightgrey"
    if label == "hub":
        return "lightgrey" if message == "n/a" else "orange"
    raise AssertionError(f"no badge color rule for label={label!r} message={message!r}")


def _iter_badges(path: Path) -> list[Badge]:
    badges: list[Badge] = []
    text = path.read_text(encoding="utf-8")
    for match in BADGE_RE.finditer(text):
        url = match.group("url")
        line = text.count("\n", 0, match.start()) + 1
        parsed = urlparse(url)
        if parsed.netloc == "codecov.io":
            if url != CODECOV_BADGE_URL:
                raise AssertionError(
                    f"{path}:{line}: unexpected Codecov badge URL: {url}"
                )
            if match.group("alt") != "codecov":
                raise AssertionError(
                    f"{path}:{line}: Codecov badge alt must be codecov: {url}"
                )
            badges.append(
                Badge(
                    path=path,
                    line=line,
                    alt=match.group("alt"),
                    url=url,
                    link=match.group("link"),
                    label="codecov",
                    message=None,
                    color=None,
                    logo=None,
                )
            )
            continue
        if parsed.netloc != "img.shields.io":
            raise AssertionError(f"{path}: non-static shields badge URL: {url}")
        if " " in url:
            raise AssertionError(f"{path}: badge URL contains a literal space: {url}")
        query = parse_qs(parsed.query)
        if parsed.path == "/static/v1":
            if "label" not in query or "message" not in query or "color" not in query:
                raise AssertionError(
                    f"{path}: badge missing label/message/color: {url}"
                )
        elif not (
            parsed.path.startswith("/github/actions/workflow/status/")
            or parsed.path
            == "/github/deployments/creative-graphic-design/design-generators/github-pages"
        ):
            raise AssertionError(f"{path}: unsupported shields badge path: {url}")
        if "label" not in query:
            raise AssertionError(f"{path}: badge missing label: {url}")
        else:
            label = _semantic_label(match.group("alt"), query["label"][0])
        message = query.get("message", [None])[0]
        if parsed.path == "/static/v1":
            for key, value in (("label", label), ("message", message)):
                if value is not None and "--" in value:
                    raise AssertionError(
                        f"{path}:{line}: static/v1 badge {key} must not contain '--': {url}"
                    )
        color = query.get("color", [None])[0]
        logo = query.get("logo", [None])[0]
        allowed_logos = _allowed_logos(path, label, message)
        if logo not in allowed_logos:
            raise AssertionError(
                f"{path}: badge {label!r} logo {logo!r} not in {allowed_logos!r}: {url}"
            )
        if logo is None:
            if "logoColor" in query:
                raise AssertionError(f"{path}: logoColor requires logo: {url}")
        elif logo not in VERIFIED_SIMPLE_ICON_SLUGS:
            raise AssertionError(f"{path}: unverified Simple Icons slug {logo!r}")
        elif query.get("logoColor") != ["white"]:
            raise AssertionError(f"{path}: badge logoColor must be white: {url}")
        expected_color = _expected_color(path, label, message)
        if expected_color is not None and color != expected_color:
            raise AssertionError(
                f"{path}: badge {label!r} color {color!r} != {expected_color!r}: {url}"
            )
        badges.append(
            Badge(
                path=path,
                line=line,
                alt=match.group("alt"),
                url=url,
                link=match.group("link"),
                label=label,
                message=message,
                color=color,
                logo=logo,
            )
        )
    return badges


def _badge_labels(path: Path) -> list[str]:
    return [badge.label for badge in _iter_badges(path)]


def _expected_link(badge: Badge) -> str | None:
    if badge.label == "codecov":
        return CODECOV_URL
    if badge.label == "docs":
        return DOCS_URL
    if badge.label == "dataset":
        if badge.message in DATASET_LINKS:
            return DATASET_LINKS[badge.message]
        if badge.logo == "huggingface":
            raise AssertionError(
                f"{badge.path}:{badge.line}: Hugging Face dataset badge {badge.message!r} must have an explicit DATASET_LINKS entry"
            )
        return None
    if badge.label == "hub":
        if badge.message in {"n/a", "not-published"}:
            return None
        return HUB_LINKS.get(badge.path.parent.name)
    if badge.label == "library" and badge.message:
        if _is_root_readme(badge.path):
            return f"{ROOT_REPO_BLOB_URL}lib/{badge.message}/README.md"
        return f"lib/{badge.message}/README.md"
    if badge.label in {"paper", "OpenReview", "arXiv", "DOI"} and badge.message:
        return PAPER_LINKS[(badge.label, unquote(badge.message))]
    return None


def _assert_badge_links() -> None:
    for path in BADGE_DOCS:
        for badge in _iter_badges(path):
            if badge.label == "hub" and badge.message in {"n/a", "not-published"}:
                if badge.link is not None:
                    raise AssertionError(
                        f"{badge.path}:{badge.line}: hub badge {badge.message!r} must not link to an unpublished Hub repo"
                    )
                continue
            expected = _expected_link(badge)
            if expected is None:
                continue
            if badge.link != expected:
                raise AssertionError(
                    f"{path}:{badge.line}: badge {badge.label!r} link {badge.link!r} != {expected!r}"
                )


def _assert_prefix(path: Path, expected: list[str]) -> None:
    labels = _badge_labels(path)
    if labels[: len(expected)] != expected:
        raise AssertionError(
            f"{path}: badge order {labels[: len(expected)]} != {expected}"
        )


def _assert_model_order(path: Path) -> None:
    labels = _badge_labels(path)
    if not labels:
        raise AssertionError(f"{path}: no badges found")
    cursor = 0
    for aliases in MODEL_ORDER:
        positions = [
            i
            for i, label in enumerate(labels[cursor:], start=cursor)
            if label in aliases
        ]
        if not positions:
            if aliases == ("venue",):
                continue
            raise AssertionError(f"{path}: missing badge for {aliases}")
        cursor = positions[-1] + 1 if aliases[0] == "dataset" else positions[0] + 1

    text = path.read_text(encoding="utf-8")
    h1 = re.search(r"^# .+$", text, re.MULTILINE)
    if h1 is None:
        raise AssertionError(f"{path}: missing H1")
    after_h1 = text[h1.end() :].lstrip()
    if not after_h1.startswith("![") and not after_h1.startswith("[!["):
        raise AssertionError(f"{path}: badges must be directly below the H1")


def check() -> None:
    _assert_prefix(REPO_ROOT / "README.md", ROOT_ORDER)
    _assert_prefix(REPO_ROOT / "lib/laygen/README.md", LAYGEN_ORDER)
    _assert_prefix(REPO_ROOT / "lib/posgen/README.md", POSGEN_ORDER)
    for path in sorted((REPO_ROOT / "models").glob("*/README.md")):
        _assert_model_order(path)
    _assert_badge_links()


def main() -> int:
    try:
        check()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("README badge checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
