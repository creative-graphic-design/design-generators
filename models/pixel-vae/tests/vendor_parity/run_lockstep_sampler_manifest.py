"""Write the package sampler order and TFRecord positions for lockstep input."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import TypedDict

import numpy as np

from pixel_vae.training import PixelVAEDataModule


class SamplerManifest(TypedDict):
    sampler_order: list[int]
    record_locations: list[tuple[int, int, int]]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("batch size and steps must be positive")

    np.random.seed(0)
    data = PixelVAEDataModule(
        str(args.data_dir), batch_size=args.batch_size, num_workers=0
    )
    data.setup("fit")
    sampler = data.samplers["train"]
    sampler.set_epoch(0)
    sample_count = args.batch_size * args.steps
    sampler_order = list(islice(iter(sampler), sample_count))
    if len(sampler_order) != sample_count:
        raise RuntimeError("the sampler did not provide every lockstep index")

    locations = data.datasets["train"].locations
    manifest: SamplerManifest = {
        "sampler_order": sampler_order,
        "record_locations": [
            (
                locations[index].file_index,
                locations[index].payload_offset,
                locations[index].payload_size,
            )
            for index in sampler_order
        ],
    }
    args.output.write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
