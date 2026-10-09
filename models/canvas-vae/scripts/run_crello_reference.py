"""Run the original Crello transforms twice with Beam's in-process FnApiRunner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from canvas_vae.data import CRELLO_V1_SHA256
from traingen_parity.tensorflow_compat import install_assert_all_finite_compat


def main() -> None:
    """Generate two independent original image and document data runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument(
        "--run-indices", type=int, nargs="+", choices=(1, 2), default=(1, 2)
    )
    args = parser.parse_args()
    marker = args.source_dir / ".crello-v1-sha256"
    if marker.read_text("ascii").strip() != CRELLO_V1_SHA256:
        raise ValueError("source directory does not have the verified Crello v1 marker")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.vendor_root / "src" / "preprocess"))

    import apache_beam as beam
    import tensorflow as tf
    from apache_beam.runners.portability.fn_api_runner import FnApiRunner

    install_assert_all_finite_compat()

    from preprocess.transforms import create_transform

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Crello reference generation is CPU-only")

    for run_index in args.run_indices:
        run_dir = args.output_dir / f"run-{run_index}"
        for name in ("crello-image", "crello-document"):
            transform = create_transform(name)(
                input_path=str(args.source_dir),
                output_path=str(run_dir / name),
                assets_dir=None,
                num_shards=args.num_shards,
                encoder_path=str(args.encoder_path),
                max_seq_length=50,
            )
            with beam.Pipeline(runner=FnApiRunner()) as pipeline:
                pipeline | f"Generate-{run_index}-{name}" >> transform

    print("completed two CPU FnApiRunner preprocessing runs")


if __name__ == "__main__":
    main()
