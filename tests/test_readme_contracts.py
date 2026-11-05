"""README badge contract tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import pytest

from devharness.checks.model_readmes import root_readme


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


def _classification_badges(
    text: str, check_readme_badges: ModuleType
) -> list[tuple[str, str, str | None]]:
    badges: list[tuple[str, str, str | None]] = []
    for match in check_readme_badges.BADGE_RE.finditer(text):
        query_label = parse_qs(urlparse(match.group("url")).query).get("label", [None])[
            0
        ]
        if query_label is None:
            continue

        label = check_readme_badges._semantic_label(match.group("alt"), query_label)
        if label in {"task", "content"}:
            badges.append((label, match.group("url"), match.group("link")))

    return badges


def _assert_root_model_classification_badges_match_table(
    readme: Path, check_readme_badges: ModuleType
) -> None:
    text = readme.read_text(encoding="utf-8")
    table_lines = root_readme._root_packages_table_lines(text)
    table_badges: list[tuple[str, str, str | None]] = []
    for line in table_lines[2:]:
        cells = root_readme._split_markdown_table_row(line)
        table_badges.extend(_classification_badges(cells[1], check_readme_badges))
        table_badges.extend(_classification_badges(cells[2], check_readme_badges))

    assert _classification_badges(text, check_readme_badges) == table_badges, (
        f"{readme}: task/content badges must match the model table cells"
    )


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
    readme = REPO_ROOT / "README.md"
    badges = [
        badge
        for badge in check_readme_badges._iter_badges(readme)
        if badge.label in {"task", "content"}
    ]

    assert {badge.label for badge in badges} == {"task", "content"}
    _assert_root_model_classification_badges_match_table(readme, check_readme_badges)


@pytest.mark.parametrize(
    ("label", "message"),
    [("task", "task-agnostic"), ("content", "content-agnostic")],
)
def test_root_readme_model_classification_badges_reject_outside_table(
    tmp_path: Path, label: str, message: str
) -> None:
    check_readme_badges = _load_check_readme_badges()
    task_badge = (
        "![task: task-agnostic](https://img.shields.io/static/v1?label=task&"
        "message=task-agnostic&color=purple)"
    )
    content_badge = (
        "![content: content-agnostic](https://img.shields.io/static/v1?"
        "label=content&message=content-agnostic&color=purple)"
    )
    outside_badge = (
        f"![{label}: {message}](https://img.shields.io/static/v1?label={label}&"
        f"message={message}&color=purple)"
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        f"""## Models

| Model | Task | Content | Venue | Ckpt | Train |
| --- | --- | --- | --- | --- | --- |
| `Example` | {task_badge} | {content_badge} |  |  |  |

## Libraries

{outside_badge}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError, match="task/content badges must match the model table cells"
    ):
        _assert_root_model_classification_badges_match_table(
            readme, check_readme_badges
        )


def test_cikm_model_venue_badge_uses_neutral_catalog_color() -> None:
    check_readme_badges = _load_check_readme_badges()

    assert (
        check_readme_badges._expected_color(
            REPO_ROOT / "models" / "radm" / "README.md", "venue", "CIKM 2023"
        )
        == "6b7280"
    )


def test_docs_index_classification_badges_follow_root_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_readme_badges = _load_check_readme_badges()
    monkeypatch.setattr(check_readme_badges, "REPO_ROOT", tmp_path)
    docs_index = tmp_path / "docs" / "index.md"
    docs_index.parent.mkdir()
    docs_index.write_text(
        "![task: task-agnostic](https://img.shields.io/static/v1?label=%F0%9F%8E%AF&"
        "message=task-agnostic&color=purple)\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="task.*color"):
        check_readme_badges._iter_badges(docs_index)


def test_docs_index_image_generation_badges_use_distinct_icons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_readme_badges = _load_check_readme_badges()
    monkeypatch.setattr(check_readme_badges, "REPO_ROOT", tmp_path)
    docs_index = tmp_path / "docs" / "index.md"
    docs_index.parent.mkdir()
    docs_index.write_text(
        "![task: text-to-image](https://img.shields.io/static/v1?"
        "label=%F0%9F%8E%A8&message=text-to-image&color=6b7280)\n"
        "![content: text](https://img.shields.io/static/v1?"
        "label=%F0%9F%93%9D&message=text&color=2f80ed)\n"
        "![task: image-to-image](https://img.shields.io/static/v1?"
        "label=%F0%9F%8E%A8&message=image-to-image&color=6b7280)\n"
        "![content: layers + text](https://img.shields.io/static/v1?"
        "label=%F0%9F%A7%A9&message=layers%20%2B%20text&color=eb5757)\n",
        encoding="utf-8",
    )

    badges = check_readme_badges._iter_badges(docs_index)
    assert [(badge.label, badge.message) for badge in badges] == [
        ("task", "text-to-image"),
        ("content", "text"),
        ("task", "image-to-image"),
        ("content", "layers + text"),
    ]

    docs_index.write_text(
        "![task: text-to-image](https://img.shields.io/static/v1?"
        "label=%F0%9F%8E%AF&message=text-to-image&color=6b7280)\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="task.*icon"):
        check_readme_badges._iter_badges(docs_index)


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
