"""Command-line dispatch for repository development checks."""

from __future__ import annotations

import sys


def main() -> int:
    """Dispatch the supported repository checker command."""
    args = sys.argv[1:]
    if len(args) < 2 or args[:2] != ["check", "model-readmes"]:
        print("usage: devharness check model-readmes", file=sys.stderr)
        return 2

    try:
        from .checks.model_readmes import check as model_readmes
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    return model_readmes.main(args[2:])
