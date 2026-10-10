from __future__ import annotations

import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

AuditSummary = Callable[
    [Mapping[str, int], Mapping[str, set[str]]],
    tuple[dict[str, dict[str, int | bool]], bool],
]


@pytest.mark.parametrize(
    ("hashes_by_split", "expected_cross_split_identity"),
    [
        (
            {"train": {"same"}, "val": {"same"}, "test": {"same"}},
            True,
        ),
        (
            {"train": {"same"}, "val": {"other"}, "test": {"other"}},
            False,
        ),
    ],
)
def test_text_element_png_identity_reports_split_and_archive_equality(
    hashes_by_split: dict[str, set[str]], expected_cross_split_identity: bool
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_crello_png_dimensions.py"
    summarize = cast(
        AuditSummary,
        runpy.run_path(str(script))["summarize_text_element_png_identity"],
    )

    by_split, across_splits = summarize(
        {"train": 3, "val": 2, "test": 4}, hashes_by_split
    )

    assert by_split == {
        split: {
            "element_count": count,
            "unique_png_count": 1,
            "within_split_byte_identical": True,
        }
        for split, count in {"train": 3, "val": 2, "test": 4}.items()
    }
    assert across_splits is expected_cross_split_identity
