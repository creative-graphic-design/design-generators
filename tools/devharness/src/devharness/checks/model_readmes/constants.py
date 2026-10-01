"""Constants and repository paths for the model README checker."""

from __future__ import annotations

from typing import TypeAlias
from pathlib import Path


# The legacy script resolved its root from the script location. The CLI is
# intentionally run from the checkout, so the equivalent package root is cwd.
REPO_ROOT = Path.cwd()
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
    / "design-generators-model-conversion"
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
PyprojectValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | list["PyprojectValue"]
    | dict[str, "PyprojectValue"]
    | None
)

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
    "dlt": {
        "license": "apache-2.0",
        "datasets": [
            "creative-graphic-design/PubLayNet",
            "RICO13",
            "creative-graphic-design/magazine",
        ],
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
    "ltnet": {"license": "other", "datasets": ["COCO", "VG-MSDN"]},
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
    "posterllama": {
        "license": "other",
        "datasets": ["creative-graphic-design/CGL-Dataset"],
    },
    "posterllava": {
        "license": "other",
        "datasets": ["Ad Banner", "CGL", "PosterLayout", "QB-Poster"],
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
    "basnet": {"license": "apache-2.0", "datasets": ["SmartText demo"]},
}


EXPECTED_MODEL_NAMES = {
    "coarse-to-fine": "Coarse-to-Fine",
    "cgb-dm": "CGB-DM",
    "ds-gan": "DS-GAN",
    "dlt": "DLT",
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
    "ltnet": "LT-Net",
    "layoutdiffusion": "LayoutDiffusion",
    "layoutformerpp": "LayoutFormer++",
    "layoutganpp": "LayoutGAN++",
    "layoutprompter": "LayoutPrompter",
    "parse-then-place": "Parse-Then-Place",
    "posterllama": "PosterLlama",
    "posterllava": "PosterLLaVA",
    "postero": "PosterO",
    "ralf": "RALF",
    "smarttext": "SmartText",
    "basnet": "BASNet",
}

EXPECTED_REPOSITORY_LINKS = {
    "layousyn": "https://github.com/mlpc-ucsd/Lay-Your-Scene",
    "layout-gpt": "https://github.com/UCSB-AI/LayoutGPT",
    "layoutdiffusion": "https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion",
    "ltnet": "https://github.com/davidhalladay/LayoutTransformer",
    "layout-action": "https://github.com/BERYLSHEEP/LayoutActionProject",
    "layoutganpp": "https://github.com/ktrk115/const_layout",
    "layoutvae": "https://github.com/Layout-Generation/layout-generation",
    "layout-detr": "https://github.com/salesforce/LayoutDETR",
    "posterllama": "https://github.com/jaepoong/PosterLlama",
    "ralf": "https://github.com/CyberAgentAILab/RALF",
    "postero": "https://github.com/theKinsley/PosterO-CVPR2025",
    "posterllava": "https://github.com/PosterLLaVA/PosterLLaVA",
    "ds-gan": "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023",
    "cgb-dm": "https://github.com/yuli0103/LayoutDiT",
    "dlt": "https://github.com/wix-incubator/DLT",
    "smarttext": "https://github.com/intchous/SmartText",
    "basnet": "https://github.com/xuebinqin/BASNet",
    "flex-dm": "https://github.com/CyberAgentAILab/flex-dm",
    "housegan": "https://github.com/ennauata/housegan",
}

PROMPT_ONLY_SLUGS = {"layout-gpt", "layoutprompter", "postero"}
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
