"""Train original CanvasVAE and retain the original validation-best checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from generate_vendor_reference import disable_tensorflow_tf32, import_original

    tf, _, _ = import_original()
    disable_tensorflow_tf32(tf)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(args.seed)

    from canvas_vae.run_metadata import write_run_metadata
    from canvasvae.data import DataSpec
    from canvasvae.main import parse_args as parse_vendor_args
    from canvasvae.train import train

    write_run_metadata(args.job_dir / "run-metadata.json", "tensorflow")
    vendor_args = parse_vendor_args(
        [
            "--dataset-name",
            "crello-document",
            "--data-dir",
            str(args.data_dir),
            "--job-dir",
            str(args.job_dir),
            "--num-epochs",
            "500",
            "--batch-size",
            "1024",
            "--latent-dim",
            "512",
            "--num-blocks",
            "1",
            "--kl",
            "32",
            "--l2",
            "1e-6",
            "--validation-freq",
            "20",
        ]
    )
    dataspec = DataSpec(
        "crello-document",
        str(args.data_dir),
        batch_size=vendor_args.batch_size,
    )
    if os.environ.get("CANVAS_VAE_S4_TRAIN_STREAM") == "1":
        from canvas_vae.update_trace import install_tensorflow_s4_train_stream

        install_tensorflow_s4_train_stream(DataSpec)

    train(vendor_args, dataspec, return_best_model=True)
    best_prefix = args.job_dir / "checkpoints" / "best.ckpt"
    if not Path(f"{best_prefix}.index").is_file():
        raise RuntimeError(f"original top-k checkpoint is missing: {best_prefix}")
    print(f"completed original training seed={args.seed}; top-k={best_prefix}")


if __name__ == "__main__":
    main()
