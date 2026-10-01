"""Command-line dispatch for repository development checks."""

from __future__ import annotations

import sys

from .checks.model_readmes import check as model_readmes


def main() -> int:
    """Dispatch the supported repository checker command."""
    if sys.argv[1:] != ["check", "model-readmes"]:
        print("usage: devharness check model-readmes", file=sys.stderr)
        return 2

    return model_readmes.main()
