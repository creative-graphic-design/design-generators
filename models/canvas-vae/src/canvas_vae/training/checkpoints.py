"""Validate CanvasVAE Lightning checkpoints against the configured final epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def validate_final_checkpoint(
    checkpoint_path: str | Path,
    *,
    max_epochs: int,
    expected_global_step: int,
) -> tuple[int, int]:
    """Require a Lightning checkpoint from the configured final epoch and step."""
    if (
        type(max_epochs) is not int
        or type(expected_global_step) is not int
        or max_epochs < 1
        or expected_global_step < 0
    ):
        raise ValueError(
            "max_epochs must be positive and global_step cannot be negative"
        )

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a Lightning checkpoint mapping")

    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    if type(epoch) is not int or type(global_step) is not int:
        raise ValueError(
            "checkpoint must contain integer epoch and global_step metadata"
        )

    expected_epoch = max_epochs - 1
    if epoch != expected_epoch or global_step != expected_global_step:
        raise ValueError(
            f"expected final checkpoint at epoch={expected_epoch}, "
            f"global_step={expected_global_step}; found epoch={epoch}, "
            f"global_step={global_step}"
        )

    return epoch, global_step


def main() -> int:
    """Validate one checkpoint against its expected final training step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-epochs", required=True, type=int)
    parser.add_argument("--expected-global-step", required=True, type=int)
    args = parser.parse_args()

    try:
        epoch, global_step = validate_final_checkpoint(
            args.checkpoint,
            max_epochs=args.max_epochs,
            expected_global_step=args.expected_global_step,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"final checkpoint passed: epoch={epoch} global_step={global_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
