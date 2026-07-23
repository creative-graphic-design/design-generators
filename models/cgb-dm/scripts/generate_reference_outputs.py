"""Generate CGB-DM reference metadata for gated parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/cgb-dm/reference/metadata.json")
    parser.add_argument(
        "--dataset", choices=["pku_posterlayout", "cgl"], default="pku_posterlayout"
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Write lightweight reference metadata."""
    args = parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"dataset": args.dataset, "seed": args.seed, "stage": "S0-S2"}, indent=2
        ),
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
