"""Run a local House-GAN ``from_pretrained`` smoke test."""

from __future__ import annotations

import argparse
from typing import cast

from housegan import HouseGanPipeline
from laygen.modeling_outputs import LayoutGenerationOutput


def main() -> None:
    """Load a converted checkpoint and generate one sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Converted checkpoint root"
    )
    args = parser.parse_args()
    pipe = HouseGanPipeline.from_pretrained(args.checkpoint_dir, local_files_only=True)
    output = cast(
        LayoutGenerationOutput,
        pipe(
            condition_type="relation",
            scene_graph={
                "nodes": [
                    {"id": 0, "label": "living_room"},
                    {"id": 1, "label": "kitchen"},
                ],
                "edges": [{"source": 0, "target": 1, "predicate": "adjacent"}],
            },
            seed=0,
        ),
    )
    print(
        {
            "bbox_shape": tuple(output.bbox.shape),
            "labels_shape": tuple(output.labels.shape),
        }
    )


if __name__ == "__main__":
    main()
