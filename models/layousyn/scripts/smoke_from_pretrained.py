"""Smoke-test LayouSyn local ``from_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from layousyn import LayouSynPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Load a converted pipeline and run one denoising step."""
    args = parse_args()
    inputs = torch.load(args.inputs, map_location=args.device)
    pipe = LayouSynPipeline.from_pretrained(args.path).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    out = pipe(
        prompt="a person sitting on a bench",
        labels=[["person", "bench"]],
        caption_embeds=inputs["pipeline_caption_embeds"],
        caption_padding_mask=inputs["pipeline_caption_padding_mask"],
        concept_embeds=inputs["pipeline_concept_embeds"],
        aspect_ratio=inputs["pipeline_aspect_ratio"],
        num_inference_steps=1,
        guidance_scale=2.0,
        generator=torch.Generator(device=args.device).manual_seed(0),
    )
    print(out.bbox.shape, out.id2label)


if __name__ == "__main__":
    main()
