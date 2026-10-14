"""Train and evaluate the original CanvasVAE TensorFlow code with a fixed seed.

Calls the unmodified ``canvasvae.train.train_and_evaluate`` after
``tf.keras.utils.set_random_seed(seed)``, with the TensorFlow compatibility
shims of ``generate_vendor_reference.py`` and TF32 disabled.
Arguments after ``--`` are passed to the original trainer, for example
``-- --num-epochs 1``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_trainer_arguments(
    dataset_name: str,
    data_dir: Path,
    job_dir: Path,
    original_args: list[str],
) -> list[str]:
    """Build arguments for the unchanged vendor DataSpec and trainer."""
    return [
        "--dataset-name",
        dataset_name,
        "--data-dir",
        str(data_dir),
        "--job-dir",
        str(job_dir),
        *original_args,
    ]


def main() -> None:
    """Launch the original trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        choices=("rico", "crello-document"),
        default="rico",
        help="Original DataSpec dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Original TFRecord directory (defaults under .cache/canvas-vae/original/data).",
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
    data_dir = (
        args.data_dir or Path(".cache/canvas-vae/original/data") / args.dataset_name
    )
    logging.basicConfig(level=logging.INFO)
    from generate_vendor_reference import import_original

    tf, _, _ = import_original()
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    tf.keras.utils.set_random_seed(args.seed)

    from canvasvae.main import parse_args
    from canvasvae.train import train_and_evaluate

    original = parse_args(
        build_trainer_arguments(
            args.dataset_name,
            data_dir,
            args.job_dir,
            args.original_args,
        )
    )
    train_and_evaluate(original)


if __name__ == "__main__":
    main()
