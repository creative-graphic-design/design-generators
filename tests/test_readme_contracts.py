"""Repository README contract tests."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
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


def _supported_checkpoint_ids(readme: Path) -> set[str]:
    text = readme.read_text(encoding="utf-8")
    match = re.search(r"^## Supported Checkpoints\s*$", text, re.MULTILINE)
    if match is None:
        return set()
    section = text[match.end() :]
    next_heading = re.search(r"\n## ", section)
    if next_heading is not None:
        section = section[: next_heading.start()]
    return set(re.findall(r"creative-graphic-design/[A-Za-z0-9_.+-]+", section))


def _model_workspace_slugs() -> set[str]:
    return {
        path.parent.name for path in (REPO_ROOT / "models").glob("*/pyproject.toml")
    }


def test_docs_models_hub_ids_match_model_readmes() -> None:
    docs_rows: dict[str, set[str]] = {}
    for line in DOCS_MODELS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| [`"):
            continue
        slug_match = re.search(r"api/models/([^/]+)/index\.md", line)
        assert slug_match is not None, line
        docs_rows[slug_match.group(1)] = set(
            re.findall(r"`(creative-graphic-design/[A-Za-z0-9_.+-]+)`", line)
        )

    model_slugs = _model_workspace_slugs()
    missing_docs_rows = sorted(model_slugs.difference(docs_rows))
    extra_docs_rows = sorted(set(docs_rows).difference(model_slugs))
    assert not missing_docs_rows, (
        f"docs/models.md missing rows for model workspace members: {missing_docs_rows}"
    )
    assert not extra_docs_rows, (
        f"docs/models.md has rows for non-workspace model packages: {extra_docs_rows}"
    )

    for readme in sorted((REPO_ROOT / "models").glob("*/README.md")):
        slug = readme.parent.name
        expected = _supported_checkpoint_ids(readme)
        if not expected:
            continue
        assert docs_rows[slug] == expected


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
