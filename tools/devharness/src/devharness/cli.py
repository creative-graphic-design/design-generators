"""Command-line dispatch for repository development checks."""

from __future__ import annotations

import sys


USAGE = (
    "usage: devharness check {model-readmes,jaxtyping-annotations} [--write-baseline]"
)


def main() -> int:
    """Dispatch the supported repository checker command."""
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "check":
        print(USAGE, file=sys.stderr)
        return 2

    if args[1] == "model-readmes":
        try:
            from .checks.model_readmes import check as model_readmes
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

        return model_readmes.main(args[2:])

    if args[1] == "jaxtyping-annotations":
        from .checks.jaxtyping_annotations import check as jaxtyping_annotations

        return jaxtyping_annotations.main(args[2:])

    print(USAGE, file=sys.stderr)
    return 2
