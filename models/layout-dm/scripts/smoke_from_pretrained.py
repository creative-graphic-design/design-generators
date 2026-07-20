"""Smoke-test LayoutDM local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - smoke scripts narrow pipeline output unions dynamically.

from layout_dm import LayoutDMPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load each converted checkpoint and run one deterministic step."""
    args = parse_args()
    for path in args.path:
        pipe = LayoutDMPipeline.from_pretrained(path)
        out = cast(
            Any,
            pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic"),
        )
        assert out.bbox.shape[-1] == 4
        assert out.labels.shape == out.mask.shape
        assert (path / "README.md").exists()
        print(path.name, out.bbox.shape, out.labels.shape)


if __name__ == "__main__":
    main()
