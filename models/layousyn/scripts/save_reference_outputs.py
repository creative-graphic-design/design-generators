"""Generate vendor reference metadata for LayouSyn parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def save_reference_outputs(
    *,
    ckpt: str | Path,
    ckpt_config: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    caption: str,
    concepts: list[str],
    cfg_scale: float = 2.0,
    aspect_ratio: float = 1.0,
    num_sampling_steps: str = "40",
) -> None:
    """Save parity input metadata next to externally generated goldens.

    The heavyweight vendor execution path intentionally stays behind the
    ``vendor`` extra and is exercised by marked parity tests.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs.json").write_text(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "ckpt_config": str(ckpt_config),
                "seed": seed,
                "caption": caption,
                "concepts": concepts,
                "cfg_scale": cfg_scale,
                "aspect_ratio": aspect_ratio,
                "num_sampling_steps": num_sampling_steps,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse reference-generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--concept", action="append", dest="concepts", default=[])
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--aspect-ratio", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", default="40")
    return parser.parse_args()


def main() -> None:
    """Write reference metadata."""
    args = parse_args()
    save_reference_outputs(
        ckpt=args.ckpt,
        ckpt_config=args.ckpt_config,
        output_dir=args.output_dir,
        seed=args.seed,
        caption=args.caption,
        concepts=args.concepts,
        cfg_scale=args.cfg_scale,
        aspect_ratio=args.aspect_ratio,
        num_sampling_steps=args.num_sampling_steps,
    )


if __name__ == "__main__":
    main()
