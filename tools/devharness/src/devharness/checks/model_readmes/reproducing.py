"""Reproducibility links and command contracts."""

from __future__ import annotations

from pathlib import Path

from .card import section
from .constants import PROMPT_ONLY_SLUGS


def assert_readme_reproducibility_link(path: Path, text: str) -> None:
    """Require a short absolute link from a model README to REPRODUCING.md."""
    reproducibility_section = section(text, "## Reproducibility")
    absolute_link = (
        "https://github.com/creative-graphic-design/design-generators/blob/main/"
        f"models/{path.parent.name}/REPRODUCING.md"
    )
    if absolute_link not in reproducibility_section:
        raise AssertionError(
            f"{path}: Reproducibility must link REPRODUCING.md as {absolute_link}"
        )

    if (
        "uv run --package " in reproducibility_section
        or "```" in reproducibility_section
    ):
        raise AssertionError(
            f"{path}: README Reproducibility must be a short link, not a walkthrough"
        )


def assert_reproducing_commands(path: Path, text: str) -> None:
    """Require ordered, runnable reproduction commands in REPRODUCING.md."""
    if "uv run --package " not in text:
        raise AssertionError(f"{path}: REPRODUCING.md must contain uv package commands")

    lower = text.lower()
    bad_command_shapes = ["python scripts/", "cd models/", "../.cache", "/tmp/"]
    for bad in bad_command_shapes:
        if bad in text:
            raise AssertionError(
                f"{path}: stale reproducibility command shape contains {bad!r}"
            )

    required_terms: list[str | tuple[str, ...]] = [
        "Workflow order:",
        "download",
        ("reference", "golden"),
        "pytest",
    ]
    if path.parent.name in PROMPT_ONLY_SLUGS:
        required_terms.extend([("prompt configuration", "save_pretrained"), "smoke"])
        if "convert checkpoints" in lower:
            raise AssertionError(
                f"{path}: prompt-only REPRODUCING.md must not mention converting checkpoints"
            )
    else:
        required_terms.extend(["convert", "from_pretrained"])

    for term in required_terms:
        alternatives = (term,) if isinstance(term, str) else term
        position = max(lower.find(alternative.lower()) for alternative in alternatives)
        if position == -1:
            raise AssertionError(f"{path}: missing reproducibility step {term!r}")

    if path.parent.name in {"coarse-to-fine", "layoutganpp"}:
        expected = (
            "Workflow order: download assets, generate references, convert checkpoints, "
            "run parity checks, then smoke-test local loading."
        )
        if expected not in text:
            raise AssertionError(
                f"{path}: reproducibility workflow must state reference -> conversion -> parity order"
            )
