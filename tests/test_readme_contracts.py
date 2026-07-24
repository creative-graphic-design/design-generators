"""Repository README contract tests."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_MODEL_READMES = REPO_ROOT / "scripts/check_model_readmes.py"
CHECK_README_BADGES = REPO_ROOT / "scripts/check_readme_badges.py"
DOCS_MODELS = REPO_ROOT / "docs" / "models.md"


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


def test_root_models_table_accepts_linked_model_names_and_framework_badges(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Venue | Runtime | Datasets | Ckpt | Train |
| :--- | :---: | --- | --- | --- | --- |
| [`LayoutFormer++`](models/layoutformerpp/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | ![framework: transformers](https://img.shields.io/static/v1?label=.&message=transformers&color=yellow&logo=huggingface&logoColor=white) | [![dataset: RICO25](https://img.shields.io/static/v1?label=%F0%9F%97%82%EF%B8%8F&message=RICO25&color=2f80ed)](https://huggingface.co/datasets/creative-graphic-design/Rico) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutformerpp/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries
""",
        encoding="utf-8",
    )

    assert check_model_readmes._root_packages_runtime_by_slug(readme) == {
        "layoutformerpp": "transformers"
    }


def test_root_models_table_rejects_runtime_without_framework_badge(
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

    with pytest.raises(AssertionError, match="exactly one framework badge"):
        check_model_readmes._root_packages_runtime_by_slug(readme)


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


def test_root_readme_badge_policy_enforces_runtime_colors_and_logos() -> None:
    check_readme_badges = _load_check_readme_badges()
    runtime_badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label == "framework"
    ]

    assert {(badge.message, badge.color, badge.logo) for badge in runtime_badges} == {
        ("transformers", "yellow", "huggingface"),
        ("diffusers", "red", "huggingface"),
        ("pydantic-ai", "violet", "pydantic"),
    }


def test_root_readme_badge_policy_enforces_task_colored_dataset_badges() -> None:
    check_readme_badges = _load_check_readme_badges()
    dataset_badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label == "dataset" and badge.message in {"RICO25", "PubLayNet"}
    ]

    assert {(badge.message, badge.color, badge.logo) for badge in dataset_badges} == {
        ("RICO25", "2f80ed", None),
        ("RICO25", "9b51e0", None),
        ("PubLayNet", "2f80ed", None),
        ("PubLayNet", "9b51e0", None),
    }


def test_root_readme_badge_policy_enforces_task_legend_badges() -> None:
    check_readme_badges = _load_check_readme_badges()
    task_badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label == "task"
    ]

    assert {(badge.message, badge.color, badge.logo) for badge in task_badges} == {
        ("content-agnostic", "2f80ed", None),
        ("content-aware", "eb5757", None),
        ("mixed", "9b51e0", None),
    }


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

| Model | Venue | Runtime | Datasets | Ckpt | Train |
| --- | --- | --- | --- | --- | --- |
| [`LayoutFlow`](models/layout-flow/README.md) | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | ![framework: diffusers](https://img.shields.io/static/v1?label=.&message=diffusers&color=red&logo=huggingface&logoColor=white) | [![dataset: PubLayNet](https://img.shields.io/static/v1?label=%F0%9F%97%82%EF%B8%8F&message=PubLayNet&color=2f80ed)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-flow/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layout-flow/TRAINING.md) |

## Libraries
""",
        encoding="utf-8",
    )

    assert check_model_readmes._root_packages_runtime_by_slug(readme) == {
        "layout-flow": "diffusers"
    }


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
            "task": set(),
            "condition": set(),
            "dataset": set(),
        }
        for axis, value in re.findall(
            r"!\[(framework|task|condition|dataset): ([^\]]+)\]"
            r"\(https://img\.shields\.io/static/v1\?",
            line,
        ):
            badges[axis].add(value)
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

    return {
        "framework": values("framework"),
        "task": values("task"),
        "condition": values("conditions"),
        "dataset": values("datasets"),
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
