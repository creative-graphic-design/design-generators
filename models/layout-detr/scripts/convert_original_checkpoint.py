"""Convert the original LayoutDETR pickle into a local HF-style checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layout_detr import (
    LayoutDetrForConditionalGeneration,
    LayoutDetrPipeline,
    LayoutDetrProcessor,
)
from layout_detr.vendor_state import extract_generator_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    state, config, report = extract_generator_state(
        args.checkpoint,
        vendor_root=args.vendor_root,
        device="cpu",
    )
    model = LayoutDetrForConditionalGeneration(config)
    if args.allow_partial:
        compatible = {
            key: value
            for key, value in state.items()
            if key in model.state_dict()
            and tuple(model.state_dict()[key].shape) == tuple(value.shape)
        }
        model.load_state_dict(compatible, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    processor = LayoutDetrProcessor(config=config)
    pipe = LayoutDetrPipeline(model=model, processor=processor, config=config)
    pipe.save_pretrained(args.output_dir)
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
