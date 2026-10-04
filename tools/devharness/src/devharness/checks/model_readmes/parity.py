"""Parity result tables, tolerance policy, and vendor badges."""

from __future__ import annotations

import re
from pathlib import Path

from .card import section


def _nonzero_number(text: str) -> bool:
    try:
        return float(text) != 0
    except ValueError:
        return False


def parity_requires_tolerance(section_text: str) -> bool:
    """Return whether parity results declare a nonzero numerical tolerance."""
    for match in re.finditer(r"\b[ra]tol\s*=?\s*`?([0-9.eE+-]+)`?", section_text):
        if _nonzero_number(match.group(1)):
            return True

    return False


def assert_vendor_parity_badge(path: Path, text: str) -> None:
    """Require the vendor-parity badge to describe the parity result section."""
    parity_section = section(text, "### Parity Results")
    badge = re.search(r"!\[vendor-parity\]\([^)]*[?&]message=([^&)]*)", text)
    if badge is None:
        raise AssertionError(f"{path}: missing vendor-parity badge")

    expected = (
        "not-run"
        if "not run" in parity_section
        else "practical-reproduction"
        if "practical training reproduction" in parity_section
        else "cpu-contract"
        if "vendor-parity CPU comparison" in parity_section
        else "tolerance-verified"
        if parity_requires_tolerance(parity_section)
        else "bit-exact"
    )
    actual = badge.group(1)
    accepted = {expected, expected.replace("--", "-")}
    if actual not in accepted:
        raise AssertionError(
            f"{path}: vendor-parity badge {actual!r} does not match Parity Results; expected {expected!r}"
        )


def assert_parity_table(path: Path, text: str) -> None:
    """Require numeric evidence in the model-card parity table."""
    parity_section = section(text, "### Parity Results")
    if "| ---" not in parity_section:
        raise AssertionError(f"{path}: Parity Results must contain a markdown table")

    rows = [
        line
        for line in parity_section.splitlines()
        if line.startswith("|") and not line.startswith("| ---")
    ]
    data_rows = rows[1:]
    if not data_rows:
        raise AssertionError(f"{path}: Parity Results table has no data rows")

    if not any(re.search(r"\d", row) for row in data_rows):
        raise AssertionError(
            f"{path}: Parity Results table must contain numeric evidence"
        )
