"""Repository README contract tests."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_MODEL_READMES = REPO_ROOT / "scripts/check_model_readmes.py"
CHECK_README_BADGES = REPO_ROOT / "scripts/check_readme_badges.py"
DOCS_MODELS = REPO_ROOT / "docs" / "models.md"
DOCS_MODEL_DATASET_LABELS = {
    "Ad Banner": "ad_banner",
    "CGL": "cgl",
    "CGL-v2": "cgl_v2",
    "COCO": "coco",
    "COCO grounded": "coco-grounded",
    "Crello": "crello",
    "GRIT": "grit",
    "housegan-floorplan-vectorized": "housegan-floorplan-vectorized",
    "Magazine": "magazine",
    "NSR-1K": "nsr-1k",
    "PKU-PosterLayout": "pku_posterlayout",
    "PosterLayout": "posterlayout",
    "PubLayNet": "publaynet",
    "RICO13": "rico13",
    "RICO25": "rico25",
    "SmartText demo assets": "smarttext-demo",
    "VG-MSDN": "vg-msdn",
    "Web": "web",
    "WebUI": "webui",
}
DOCS_MODEL_TASK_COLORS = {
    frozenset({"content-agnostic-layout-generation"}): "2f80ed",
    frozenset({"content-aware-layout-generation"}): "eb5757",
    frozenset({"layout-evaluation"}): "6b7280",
    frozenset({"saliency-detection"}): "009688",
    frozenset(
        {"content-agnostic-layout-generation", "content-aware-layout-generation"}
    ): "9b51e0",
}


def _load_check_model_readmes() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_model_readmes", CHECK_MODEL_READMES
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_check_readme_badges() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_readme_badges", CHECK_README_BADGES
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_script(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_model_readme_contracts() -> None:
    _run_script("scripts/check_model_readmes.py")


def test_readme_badge_contracts() -> None:
    _run_script("scripts/check_readme_badges.py")


def test_citation_contract_accepts_matching_arxiv_metadata() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{yang2024posterllava,
  title = {PosterLLaVa: Constructing a Unified Multi-modal Layout Generator},
  author = {Tao Yang and Yingmin Luo},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2406.02884"
}
```
"""

    check_model_readmes._assert_citation_bibtex(Path("models/example/README.md"), text)


def test_citation_contract_rejects_arxiv_id_mismatch() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2303.08137},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2303.08137"
}
```
"""

    with pytest.raises(AssertionError, match="Citation arXiv ids"):
        check_model_readmes._assert_citation_bibtex(
            Path("models/example/README.md"), text
        )


def test_citation_contract_rejects_missing_arxiv_bibtex_fields() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  url = "https://arxiv.org/abs/2406.02884"
}
```
"""

    with pytest.raises(AssertionError, match="missing required fields"):
        check_model_readmes._assert_citation_bibtex(
            Path("models/example/README.md"), text
        )


def test_citation_contract_rejects_eprint_url_mismatch() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2303.08137"
}
```
"""

    with pytest.raises(AssertionError, match="does not match url"):
        check_model_readmes._assert_citation_bibtex(
            Path("models/example/README.md"), text
        )


def test_citation_contract_allows_conference_bibtex_without_eprint() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2303.08137)

## Citation

```bibtex
@inproceedings{example2023,
  title = {Example},
  author = {Example Author},
  booktitle = {Proceedings of Example Conference},
  year = {2023}
}
```
"""

    check_model_readmes._assert_citation_bibtex(Path("models/example/README.md"), text)


def test_static_v1_badges_reject_double_hyphen_query_values(tmp_path: Path) -> None:
    check_readme_badges = _load_check_readme_badges()
    readme = tmp_path / "README.md"
    readme.write_text(
        "![vendor-parity](https://img.shields.io/static/v1?"
        "label=vendor--parity&message=bit--exact&color=success&style=flat-square)\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must not contain '--'"):
        check_readme_badges._iter_badges(readme)


def test_root_models_table_accepts_linked_model_names_and_reproduction_badges(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Venue | Ckpt | Train |
| :--- | :---: | --- | --- |
| [`LayoutFormer++`](models/layoutformerpp/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutformerpp/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries
""",
        encoding="utf-8",
    )

    assert check_model_readmes._root_model_slugs(readme) == {"layoutformerpp"}


def test_root_models_table_rejects_metadata_columns(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Venue | Runtime | Datasets | Ckpt | Train |
| --- | --- | --- | --- | --- | --- |
| [`LayoutDM`](models/layout-dm/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | `🧨diffusers` | ![dataset: PubLayNet](https://img.shields.io/static/v1?label=%F0%9F%97%82%EF%B8%8F&message=PubLayNet&color=2f80ed) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries
""",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="Model, Venue, Ckpt, Train"):
        check_model_readmes._root_model_slugs(readme)


def test_model_readme_reproducibility_accepts_repo_root_link(tmp_path: Path) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "models" / "layout-dm" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        """# Model Card for LayoutDM

## Reproducibility

See [REPRODUCING.md](models/layout-dm/REPRODUCING.md) for commands.

## Environmental Impact
""",
        encoding="utf-8",
    )

    check_model_readmes._assert_readme_reproducibility_link(
        readme, readme.read_text(encoding="utf-8")
    )


def test_readme_badge_policy_derives_model_label_from_alt_prefix(
    tmp_path: Path,
) -> None:
    check_readme_badges = _load_check_readme_badges()
    readme = tmp_path / "README.md"
    readme.write_text(
        "[![model: LayoutGAN++](https://img.shields.io/static/v1?"
        "label=%F0%9F%A7%A0&message=LayoutGAN%2B%2B&color=blue)]"
        "(models/layoutganpp/README.md)\n",
        encoding="utf-8",
    )

    badges = check_readme_badges._iter_badges(readme)
    assert [
        (badge.label, badge.message, badge.color, badge.logo) for badge in badges
    ] == [("model", "LayoutGAN++", "blue", None)]


def test_root_readme_models_table_omits_generated_metadata_badges() -> None:
    check_readme_badges = _load_check_readme_badges()
    metadata_badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label in {"dataset", "framework", "task"}
    ]

    assert metadata_badges == []


def test_root_readme_badge_policy_enforces_library_badges() -> None:
    check_readme_badges = _load_check_readme_badges()
    library_badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label == "library"
    ]

    assert {
        (badge.message, badge.color, badge.logo, badge.link) for badge in library_badges
    } == {
        (
            "laygen",
            "2f80ed",
            None,
            "https://github.com/creative-graphic-design/design-generators/blob/main/lib/laygen/README.md",
        ),
        (
            "posgen",
            "00a88f",
            None,
            "https://github.com/creative-graphic-design/design-generators/blob/main/lib/posgen/README.md",
        ),
        (
            "traingen",
            "27ae60",
            None,
            "https://github.com/creative-graphic-design/design-generators/blob/main/lib/traingen/README.md",
        ),
        (
            "traingen-parity",
            "9b51e0",
            None,
            "https://github.com/creative-graphic-design/design-generators/blob/main/lib/traingen-parity/README.md",
        ),
    }


def test_root_readme_badges_do_not_use_label_arrow() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "label=>" not in text
    assert "label=%3E" not in text


def test_root_models_table_requires_training_link_when_file_exists(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Venue | Ckpt | Train |
| --- | --- | --- | --- |
| [`LayoutFlow`](models/layout-flow/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-flow/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layout-flow/TRAINING.md) |

## Libraries
""",
        encoding="utf-8",
    )

    assert check_model_readmes._root_model_slugs(readme) == {"layout-flow"}


def _model_workspace_slugs() -> set[str]:
    return {
        path.parent.name for path in (REPO_ROOT / "models").glob("*/pyproject.toml")
    }


def _docs_model_badge_rows() -> dict[str, dict[str, set[str]]]:
    docs_rows: dict[str, dict[str, set[str]]] = {}
    for line in DOCS_MODELS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ["):
            continue
        slug_match = re.search(r"api/models/([^/]+)/", line)
        if slug_match is None:
            continue
        badges: dict[str, set[str]] = {
            "framework": set(),
            "dataset": set(),
            "dataset_color": set(),
        }
        for match in re.finditer(
            r"!\[(framework|dataset): ([^\]]+)\]"
            r"\((https://img\.shields\.io/static/v1\?[^)]+)\)",
            line,
        ):
            axis = match.group(1)
            value = match.group(2)
            query = parse_qs(urlparse(match.group(3)).query)
            if axis == "framework":
                badges["framework"].add(value)
            else:
                badges["dataset"].add(DOCS_MODEL_DATASET_LABELS[value])
                badges["dataset_color"].add(query["color"][0])
        docs_rows[slug_match.group(1)] = badges
    return docs_rows


def _metadata_values(pyproject: Path) -> dict[str, set[str]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    metadata = data["tool"]["design-generators"]

    def values(key: str) -> set[str]:
        value = metadata[key]
        if isinstance(value, str):
            return {value}
        return set(value)

    tasks = values("task")
    return {
        "framework": values("framework"),
        "dataset": values("datasets"),
        "dataset_color": {DOCS_MODEL_TASK_COLORS[frozenset(tasks)]},
    }


def test_docs_models_metadata_badges_match_model_pyprojects() -> None:
    docs_rows = _docs_model_badge_rows()

    model_slugs = _model_workspace_slugs()
    missing_docs_rows = sorted(model_slugs.difference(docs_rows))
    extra_docs_rows = sorted(set(docs_rows).difference(model_slugs))
    assert not missing_docs_rows, (
        f"docs/models.md missing rows for model workspace members: {missing_docs_rows}"
    )
    assert not extra_docs_rows, (
        f"docs/models.md has rows for non-workspace model packages: {extra_docs_rows}"
    )

    for pyproject in sorted((REPO_ROOT / "models").glob("*/pyproject.toml")):
        assert docs_rows[pyproject.parent.name] == _metadata_values(pyproject)


def test_hugging_face_emoji_contract_allows_multiple_runtime_mentions(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and second `🤗transformers`.",
        encoding="utf-8",
    )

    check_model_readmes._assert_library_name_style(readme)


@pytest.mark.parametrize(
    ("emoji", "library", "expected_message"),
    [
        ("🤗", "transformers", "🤗 must annotate"),
        ("🧨", "diffusers", "🧨 must annotate"),
        ("🤖", "pydantic-ai", "🤖 must annotate"),
    ],
)
def test_runtime_emoji_contract_rejects_space_after_emoji(
    tmp_path: Path,
    emoji: str,
    library: str,
    expected_message: str,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    runtime_label = f"`{emoji} {library}`"
    readme.write_text(f"This package uses {runtime_label}.", encoding="utf-8")

    with pytest.raises(AssertionError, match=expected_message):
        check_model_readmes._assert_library_name_style(readme)


def test_hugging_face_emoji_contract_rejects_unattached_emoji(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and stray 🤗.",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        check_model_readmes._assert_library_name_style(readme)


def test_hugging_face_emoji_contract_rejects_emoji_outside_code_span(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🤗 [`transformers`](https://huggingface.co/docs/transformers/index).",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        check_model_readmes._assert_library_name_style(readme)


def test_diffusers_emoji_contract_rejects_unattached_emoji(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🤗transformers`](https://huggingface.co/docs/transformers/index) "
        "and stray 🧨.",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        check_model_readmes._assert_library_name_style(readme)


def test_diffusers_emoji_contract_rejects_emoji_outside_code_span(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🧨 [`diffusers`](https://huggingface.co/docs/diffusers/index).",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        check_model_readmes._assert_library_name_style(readme)


def test_pip_install_contract_reports_expected_direct_url_example() -> None:
    check_model_readmes = _load_check_model_readmes()
    section = """Clone this repository first.

```bash
uv sync --package layout-dm
```
"""

    with pytest.raises(AssertionError, match="Expected example") as exc_info:
        check_model_readmes._assert_pip_install_snippet(
            Path("models/layout-dm/README.md"),
            section,
            [
                ("laygen", "lib/laygen"),
                ("layout-dm", "models/layout-dm"),
            ],
            "How to Get Started",
        )

    message = str(exc_info.value)
    assert "laygen @ git+https://github.com/creative-graphic-design" in message
    assert "subdirectory=models/layout-dm" in message


def test_library_pip_install_contract_accepts_direct_url() -> None:
    check_model_readmes = _load_check_model_readmes()
    section = """Install directly from this repository.

```bash
pip install "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen"
```
"""

    check_model_readmes._assert_pip_install_snippet(
        Path("lib/laygen/README.md"),
        section,
        [("laygen", "lib/laygen")],
        "Install",
    )


def test_model_install_contract_requires_workspace_model_dependencies() -> None:
    check_model_readmes = _load_check_model_readmes()
    text = """# Model Card for SmartText

## How to Get Started with the Model

```bash
pip install \\
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \\
  "smarttext @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/smarttext"
```

## Training Details
"""

    with pytest.raises(AssertionError, match="models/basnet"):
        check_model_readmes._assert_model_pip_install_snippet(
            REPO_ROOT / "models" / "smarttext" / "README.md",
            text,
        )
