"""Convert an original TensorFlow PixelVAE checkpoint to Transformers format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from laygen.common.vendor import vendor_root

from pixel_vae import PixelVAEConfig, PixelVAEModel
from pixel_vae.conversion import (
    convert_tensorflow_variables,
    extract_tensorflow_weights,
    tensorflow_state_sha256,
)

SOURCE_COMMIT = "bc1e2072ba3a253f1b099e8b0c604f6051e787da"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint conversion options."""
    parser = argparse.ArgumentParser(
        description="Convert PixelVAE weights from the pinned TensorFlow source."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional TensorFlow checkpoint prefix; omit to convert a seeded untrained state",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/pixel-vae/converted"),
        help="directory for save_pretrained output and conversion metadata (default: .cache/pixel-vae/converted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="TensorFlow initialization seed when --checkpoint is omitted (default: 0)",
    )
    parser.add_argument(
        "--source-commit",
        default=SOURCE_COMMIT,
        help=f"source repository commit recorded in metadata (default: {SOURCE_COMMIT})",
    )
    return parser.parse_args()


def main() -> None:
    """Build the original model, map its variables, and serialize both models."""
    args = parse_args()
    import tensorflow as tf

    vendor = vendor_root("canvas-vae", marker="src/pixel-vae/pixelvae/model.py")
    sys.path.insert(0, str(vendor / "src" / "pixel-vae"))
    from pixelvae.model import PixelVAE

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

    reference = PixelVAE(latent_dim=256, kl=100.0, l2=1e-6)
    if args.checkpoint is not None:
        reference.load_weights(str(args.checkpoint))

    source_weights = extract_tensorflow_weights(reference)
    model = PixelVAEModel(PixelVAEConfig())
    report = convert_tensorflow_variables(source_weights, model)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    model.encoder.save_pretrained(args.output_dir / "encoder")
    metadata = {
        "tensorflow_version": tf.__version__,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "encoder_state_sha256": tensorflow_state_sha256(
            {
                key: value
                for key, value in source_weights.items()
                if not key.startswith("decoder/")
            }
        ),
        "assigned_tensors": report.assigned_tensors,
        "missing_source_keys": report.missing_source_keys,
        "unexpected_source_keys": report.unexpected_source_keys,
    }
    (args.output_dir / "conversion.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
