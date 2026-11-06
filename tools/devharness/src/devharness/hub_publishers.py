"""Shared discovery and badge rules for model Hub publishers."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PUBLISHER_PATH = Path("scripts") / "publish_to_hub.py"
HUB_BADGE_STATUS_PAIRS = (("planned-code-only", "published-code-only"),)
HUB_BADGE_COLORS = {
    "planned-code-only": "orange",
    "published-code-only": "brightgreen",
}


@dataclass(frozen=True)
class HubPublisher:
    """A model package with its own Hugging Face publication procedure."""

    package: str
    directory: Path


def discover_publishers(repository_root: Path) -> list[HubPublisher]:
    """Return model packages that contain the package-local publisher script."""
    models_directory = repository_root / "models"
    publishers: list[HubPublisher] = []

    for package_directory in sorted(models_directory.iterdir()):
        if not package_directory.is_dir():
            continue

        if not (package_directory / "pyproject.toml").is_file():
            continue

        if not (package_directory / PUBLISHER_PATH).is_file():
            continue

        pyproject_path = package_directory / "pyproject.toml"
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        package_name = pyproject.get("project", {}).get("name")
        if not isinstance(package_name, str):
            raise TypeError(f"{pyproject_path} must define project.name")

        publishers.append(
            HubPublisher(
                package=package_name,
                directory=Path("models") / package_directory.name,
            )
        )

    return publishers


def assert_hub_badge_matches_publisher(
    package_name: str,
    messages: Sequence[str],
    publisher_directories: set[str],
) -> None:
    """Require the README Hub badge status to agree with publisher availability."""
    if len(messages) != 1:
        raise AssertionError(
            f"models/{package_name}/README.md: expected exactly one hub badge, found {len(messages)}"
        )

    message = messages[0]
    published_messages = {published for _, published in HUB_BADGE_STATUS_PAIRS}
    if package_name in publisher_directories:
        if message not in published_messages:
            expected_messages = ", ".join(sorted(published_messages))
            raise AssertionError(
                f"models/{package_name}/README.md: hub badge {message!r} must use a published Hub badge for its publisher script; expected one of {expected_messages}"
            )

        return

    if message.startswith("published-"):
        raise AssertionError(
            f"models/{package_name}/README.md: hub badge {message!r} cannot be used without {PUBLISHER_PATH.as_posix()}"
        )
