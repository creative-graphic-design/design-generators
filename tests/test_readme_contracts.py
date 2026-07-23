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
DOCS_MODELS = REPO_ROOT / "docs" / "models.md"


def _load_check_model_readmes() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_model_readmes", CHECK_MODEL_READMES
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def test_hugging_face_emoji_contract_rejects_second_mention(tmp_path: Path) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and second 🤗 `transformers`.",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="only on the first"):
        check_model_readmes._assert_library_name_style(readme)


def test_hugging_face_emoji_contract_rejects_unattached_emoji(
    tmp_path: Path,
) -> None:
    check_model_readmes = _load_check_model_readmes()
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and stray 🤗.",
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
