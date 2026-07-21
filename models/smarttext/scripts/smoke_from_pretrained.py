"""Run a local SmartText ``save_pretrained`` to ``from_pretrained`` smoke test."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import cast

import torch
from PIL import Image, ImageFont

from laygen.modeling_outputs import LayoutGenerationOutput
from smarttext import (
    SmartTextBASNet,
    SmartTextConfig,
    SmartTextPipeline,
    SmartTextProcessor,
    SmartTextScorer,
)


def main() -> None:
    """Run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if args.checkpoint_dir is None:
            config = SmartTextConfig(grid_num=16, max_font_size=20, min_font_size=10)
            pipe = SmartTextPipeline(
                SmartTextScorer(config),
                SmartTextBASNet(config),
                SmartTextProcessor(config=config),
                config=config,
            )
            path = Path(tmp) / "smarttext-smt"
            pipe.save_pretrained(path)
        else:
            path = args.checkpoint_dir
        loaded = SmartTextPipeline.from_pretrained(path, local_files_only=True)
        out = cast(
            LayoutGenerationOutput,
            loaded(
                Image.new("RGB", (64, 64), "white"),
                prompt="ICME 2020",
                saliency=torch.zeros(64, 64),
                font=ImageFont.load_default(),
            ),
        )
        print({"bbox_shape": tuple(out.bbox.shape), "labels": out.id2label})


if __name__ == "__main__":
    main()
