"""Generate LayoutDETR vendor reference tensors outside git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from layout_detr import LayoutDetrProcessor
from layout_detr.vendor_state import extract_generator_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, config, report = extract_generator_state(
        args.checkpoint,
        vendor_root=args.vendor_root,
        device="cpu",
    )
    processor = LayoutDetrProcessor(config=config)
    encoded = processor(
        images=Image.open(args.background).convert("RGB"),
        texts=args.texts.split("|"),
        labels=args.labels.split("|"),
    )
    generator = torch.Generator().manual_seed(args.seed)
    latents = torch.randn(
        (1, config.max_seq_length, config.z_dim),
        generator=generator,
    )
    torch.save({**dict(encoded), "latents": latents}, args.output_dir / "inputs.pt")
    meta = {
        "command": "generate_reference_outputs.py",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "background": str(args.background),
        "texts": args.texts,
        "labels": args.labels,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "conversion_report": report,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
