"""Tests for shared Hub publisher discovery and badge policy."""

from __future__ import annotations

import pytest
from devharness.hub_publishers import (
    HUB_BADGE_COLORS,
    HUB_BADGE_STATUS_PAIRS,
    assert_hub_badge_matches_publisher,
)


def test_hub_badge_status_pairs_are_explicit() -> None:
    assert HUB_BADGE_STATUS_PAIRS == (("planned-code-only", "published-code-only"),)
    assert HUB_BADGE_COLORS == {
        "planned-code-only": "orange",
        "published-code-only": "brightgreen",
    }


@pytest.mark.parametrize(
    ("package", "message", "publishers"),
    [
        ("collage-diffusion", "published-code-only", {"collage-diffusion"}),
        ("another-model", "planned-code-only", set()),
    ],
)
def test_hub_badge_status_matches_publisher_presence(
    package: str, message: str, publishers: set[str]
) -> None:
    assert_hub_badge_matches_publisher(package, [message], publishers)


@pytest.mark.parametrize(
    ("package", "message", "publishers", "diagnostic"),
    [
        (
            "collage-diffusion",
            "planned-code-only",
            {"collage-diffusion"},
            "must use a published Hub badge",
        ),
        (
            "another-model",
            "published-code-only",
            set(),
            "cannot be used without scripts/publish_to_hub.py",
        ),
    ],
)
def test_hub_badge_status_rejects_publisher_mismatches(
    package: str,
    message: str,
    publishers: set[str],
    diagnostic: str,
) -> None:
    with pytest.raises(AssertionError, match=diagnostic):
        assert_hub_badge_matches_publisher(package, [message], publishers)
