"""Smoke test a converted LayoutDETR checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from layout_detr import LayoutDetrPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    pipe = LayoutDetrPipeline.from_pretrained(args.model_dir, local_files_only=True)
    output = pipe(
        Image.new("RGB", (64, 64), "white"),
        texts=["SALE", "SHOP NOW"],
        labels=["header", "button"],
        seed=0,
    )
    print(
        {
            "bbox_shape": tuple(output.bbox.shape),
            "labels_shape": tuple(output.labels.shape),
        }
    )


if __name__ == "__main__":
    main()
