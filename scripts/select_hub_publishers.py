#!/usr/bin/env python3
"""Select changed model packages that define a Hugging Face publisher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PUBLISHER_PATH = Path("scripts/publish_to_hub.py")


@dataclass(frozen=True)
class HubPublisher:
    """A model package with its own Hugging Face publication procedure."""

    package: str
    directory: str


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

        pyproject = tomllib.loads((package_directory / "pyproject.toml").read_text())
        package_name = pyproject.get("project", {}).get("name")
        if not isinstance(package_name, str):
            raise TypeError(
                f"{package_directory / 'pyproject.toml'} must define project.name"
            )

        publishers.append(
            HubPublisher(
                package=package_name,
                directory=package_directory.relative_to(repository_root).as_posix(),
            )
        )

    return publishers


def select_publishers(
    repository_root: Path, changed_paths: Sequence[str] | None
) -> list[HubPublisher]:
    """Select configured publishers whose model package changed."""
    publishers = discover_publishers(repository_root)
    if changed_paths is None:
        return publishers

    changed_packages = {
        parts[1]
        for path in changed_paths
        if len(parts := Path(path).parts) > 1 and parts[0] == "models"
    }
    return [
        publisher
        for publisher in publishers
        if Path(publisher.directory).name in changed_packages
    ]


def _changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit for changed-package selection")
    parser.add_argument("--head", help="head commit for changed-package selection")
    args = parser.parse_args(argv)
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be provided together")

    changed_paths = (
        None if args.base is None else _changed_paths(args.base, args.head)
    )
    publishers = select_publishers(REPOSITORY_ROOT, changed_paths)
    print(
        json.dumps(
            {
                "include": [
                    {"package": publisher.package, "directory": publisher.directory}
                    for publisher in publishers
                ]
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
