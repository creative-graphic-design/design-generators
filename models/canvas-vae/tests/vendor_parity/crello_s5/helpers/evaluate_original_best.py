"""Evaluate an original top-k checkpoint with the unmodified TensorFlow evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from generate_vendor_reference import disable_tensorflow_tf32, import_original

    tf, _, _ = import_original()
    disable_tensorflow_tf32(tf)
    from canvasvae.data import DataSpec
    from canvasvae.main import parse_args as parse_vendor_args
    from canvasvae.models.vae import VAE
    from canvasvae.train import evaluate

    vendor_args = parse_vendor_args(
        [
            "--dataset-name",
            "crello-document",
            "--data-dir",
            str(args.data_dir),
            "--job-dir",
            str(args.output_dir),
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
    model = VAE(
        dataspec.make_input_columns(),
        latent_dim=vendor_args.latent_dim,
        decoder_type=vendor_args.decoder_type,
        num_blocks=vendor_args.num_blocks,
        block_type=vendor_args.block_type,
        kl=vendor_args.kl,
        l2=vendor_args.l2,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=vendor_args.learning_rate,
            clipnorm=1.0,
        )
    )
    first_batch = next(iter(dataspec.make_dataset("test")))
    model(first_batch, training=False)
    model.load_weights(str(args.checkpoint))

    records = []
    for seed in args.seeds:
        tf.keras.utils.set_random_seed(seed)
        vendor_args.job_dir = str(args.output_dir / f"evaluation-seed-{seed}")
        Path(vendor_args.job_dir).mkdir(parents=True, exist_ok=True)
        result = evaluate(vendor_args, dataspec, model)
        records.append({"evaluation_seed": seed, **result})

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_rule": "original validation-best top-1 best.ckpt",
        "test_documents": dataspec.size("test"),
        "evaluation_seeds": list(args.seeds),
        "results": records,
    }
    destination = args.output_dir / "s5-evaluation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evaluation": str(destination),
                "test_documents": summary["test_documents"],
            }
        )
    )


if __name__ == "__main__":
    main()
