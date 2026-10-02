"""README badge contract tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_README_BADGES = REPO_ROOT / "scripts/check_readme_badges.py"


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


def test_root_readme_model_classification_badges_follow_policy() -> None:
    check_readme_badges = _load_check_readme_badges()
    badges = [
        badge
        for badge in check_readme_badges._iter_badges(REPO_ROOT / "README.md")
        if badge.label in {"task", "content"}
    ]

    assert len(badges) == 56
    assert {badge.label for badge in badges} == {"task", "content"}


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
            "lib/laygen/README.md",
        ),
        (
            "posgen",
            "00a88f",
            None,
            "lib/posgen/README.md",
        ),
        (
            "traingen",
            "27ae60",
            None,
            "lib/traingen/README.md",
        ),
        (
            "traingen-parity",
            "9b51e0",
            None,
            "lib/traingen-parity/README.md",
        ),
    }


def test_root_readme_badges_do_not_use_label_arrow() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "label=>" not in text
    assert "label=%3E" not in text
