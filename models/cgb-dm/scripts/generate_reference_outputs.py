"""Generate CGB-DM reference metadata for gated parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgb_dm.training.parity import write_vendor_order_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/cgb-dm/reference/metadata.json")
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--dataset", choices=["pku_posterlayout", "cgl"], default="pku_posterlayout"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="train")
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
    if args.data_root is not None:
        manifest_output = args.manifest_output or str(
            path.with_name(f"{args.dataset}_{args.split}_manifest.json")
        )
        manifest_path = write_vendor_order_manifest(
            data_root=args.data_root,
            output=manifest_output,
            dataset=args.dataset,
            split=args.split,
            seed=args.seed,
        )
        print(manifest_path)


if __name__ == "__main__":
    main()
