"""Inspect TensorFlow variables in a Flex-DM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flex_dm.tf_checkpoint import list_tf_checkpoint_variables, tensorflow_version


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write checkpoint inventory JSON."""
    args = parse_args()
    checkpoint = args.job_dir / "checkpoints" / args.checkpoint_name
    variables = list_tf_checkpoint_variables(checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "tensorflow_version": tensorflow_version(),
                "variables": [
                    {"name": name, "shape": list(shape)} for name, shape in variables
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
