#!/usr/bin/env python3
"""Select changed model packages that define a Hugging Face publisher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from devharness.hub_publishers import HubPublisher, discover_publishers

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def select_publishers(
    repository_root: Path, changed_paths: Sequence[str] | None
) -> list[HubPublisher]:
    """Select configured publishers whose model package changed."""
    publishers = discover_publishers(repository_root)
    if changed_paths is None:
        return publishers

    changed_packages: set[str] = set()
    for path in changed_paths:
        first_directory, separator, remaining_path = path.partition("/")
        if first_directory != "models" or not separator:
            continue

        package_directory, _, _ = remaining_path.partition("/")
        if package_directory:
            changed_packages.add(package_directory)

    return [
        publisher
        for publisher in publishers
        if publisher.directory.name in changed_packages
    ]


def _changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def is_package_change_current_on_main(
    package_directory: Path, event_revision: str, main_ref: str
) -> bool:
    """Return whether main has any newer commit that changes this package."""
    result = subprocess.run(
        [
            "git",
            "log",
            "--first-parent",
            "--format=%H",
            f"{event_revision}..{main_ref}",
            "--",
            package_directory.as_posix(),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return not result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit for changed-package selection")
    parser.add_argument("--head", help="head commit for changed-package selection")
    parser.add_argument(
        "--check-freshness",
        action="store_true",
        help="check whether main has a newer commit that changes one package",
    )
    parser.add_argument("--package-directory", type=Path)
    parser.add_argument("--event-revision")
    parser.add_argument("--main-ref", default="origin/main")
    args = parser.parse_args(argv)

    if args.check_freshness:
        if args.package_directory is None or args.event_revision is None:
            parser.error(
                "--check-freshness requires --package-directory and --event-revision"
            )

        if args.base is not None or args.head is not None:
            parser.error("--check-freshness cannot be combined with --base or --head")

        print(
            json.dumps(
                {
                    "is_current": is_package_change_current_on_main(
                        args.package_directory,
                        args.event_revision,
                        args.main_ref,
                    )
                },
                separators=(",", ":"),
            )
        )
        return 0

    if (
        args.package_directory is not None
        or args.event_revision is not None
        or args.main_ref != "origin/main"
    ):
        parser.error(
            "--package-directory, --event-revision, and --main-ref require --check-freshness"
        )

    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be provided together")

    changed_paths = None if args.base is None else _changed_paths(args.base, args.head)
    publishers = select_publishers(REPOSITORY_ROOT, changed_paths)
    print(
        json.dumps(
            {
                "include": [
                    {
                        "package": publisher.package,
                        "directory": publisher.directory.as_posix(),
                    }
                    for publisher in publishers
                ]
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
