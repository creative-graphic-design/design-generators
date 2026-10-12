"""Train and evaluate the original CanvasVAE TensorFlow code with a fixed seed.

Calls the unmodified ``canvasvae.train.train_and_evaluate`` after
``tf.keras.utils.set_random_seed(seed)``, with the TensorFlow 2.15 shims of
``generate_vendor_reference.py`` (Keras preprocessing import path and a
``BatchNormalization`` that ignores propagated masks) and TF32 disabled.
Arguments after ``--`` are passed to the original trainer, for example
``-- --num-epochs 1``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from generate_vendor_reference import import_original


def main() -> None:
    """Launch the original trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".cache/canvas-vae/original/data/rico"),
        help="Original TFRecord directory (default: %(default)s).",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=Path(".cache/canvas-vae/original/jobs/rico-seed0"),
        help="Output directory for logs, checkpoints, and test results (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="TensorFlow global seed (default: %(default)s).",
    )
    parser.add_argument(
        "original_args", nargs="*", help="Extra original trainer arguments after `--`."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tf, _, _ = import_original()
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    tf.keras.utils.set_random_seed(args.seed)

    from canvasvae.main import parse_args
    from canvasvae.train import train_and_evaluate

    original = parse_args(
        [
            "--dataset-name",
            "rico",
            "--data-dir",
            str(args.data_dir),
            "--job-dir",
            str(args.job_dir),
            *args.original_args,
        ]
    )
    train_and_evaluate(original)


if __name__ == "__main__":
    main()
