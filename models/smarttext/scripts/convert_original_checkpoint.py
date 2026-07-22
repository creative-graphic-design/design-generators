"""Convert original SmartText checkpoints to a local pipeline directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smarttext.conversion import convert_original_checkpoints
from smarttext.configuration_smarttext import SmartTextConfig


def main() -> None:
    """Run checkpoint conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smt-checkpoint", type=Path, required=True)
    parser.add_argument("--basnet-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/smarttext/converted/smarttext-smt"),
    )
    args = parser.parse_args()
    report = convert_original_checkpoints(
        smt_checkpoint=args.smt_checkpoint,
        basnet_checkpoint=args.basnet_checkpoint,
        output_dir=args.output_dir,
        config=SmartTextConfig(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
