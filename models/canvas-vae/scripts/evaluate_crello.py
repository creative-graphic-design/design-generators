"""Evaluate Crello CanvasVAE reconstruction and prior-generation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from laygen.common.randomness import randn

from canvas_vae import CanvasVAECrelloModel
from canvas_vae.data import CrelloBatch, CrelloSplit
from canvas_vae.metrics import (
    crello_field_statistics,
    crello_histogram_scores,
    crello_reconstruction_scores,
)
from canvas_vae.training.datamodule import CanvasVAECrelloDataModule


@torch.no_grad()
def main() -> None:
    """Write test-split reconstruction and random-generation scores as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory written by save_pretrained.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".cache/canvas-vae/crello/package-run-1"),
        help="Prepared Crello split directory (default: %(default)s).",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(".cache/canvas-vae/crello/fixture"),
        help="Verified embedding fixture directory (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Prior-generation seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Evaluation batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device (default: %(default)s)."
    )
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path.")
    args = parser.parse_args()

    model = CanvasVAECrelloModel.from_pretrained(args.checkpoint).to(args.device).eval()
    data = CanvasVAECrelloDataModule(
        data_dir=str(args.data_dir),
        fixture_dir=str(args.fixture_dir),
        batch_size=args.batch_size,
    )
    data.setup()
    documents = data.splits[CrelloSplit.test].documents
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    kl_values = []
    for batch in data.test_dataloader():
        batch = batch.to(args.device)
        output = model(batch)
        if output.kl_divergence is None:
            raise RuntimeError("evaluation forward did not return KL divergence")

        for key, values in crello_reconstruction_scores(
            batch, output, model.config
        ).items():
            sums[key] = sums.get(key, 0.0) + float(values.sum())
            counts[key] = counts.get(key, 0) + values.numel()

        kl_values.append(float(output.kl_divergence))

    results = {f"reconst_{key}": value / counts[key] for key, value in sums.items()}
    results["reconst_kl_divergence"] = sum(kl_values) / len(kl_values)
    if data.processor is None:
        raise RuntimeError("data module setup did not construct a processor")

    reference_batch = cast(CrelloBatch, data.processor(documents).to(args.device))
    reference = crello_field_statistics(reference_batch, model.config)
    for seed in args.seeds:
        latents = randn(
            (len(documents), model.config.latent_dim),
            generator=torch.Generator().manual_seed(seed),
            device=args.device,
        )
        generated = crello_field_statistics(
            reference_batch,
            model.config,
            model(latents=latents),
        )
        results.update(
            {
                f"random_seed{seed}_{key}": value
                for key, value in crello_histogram_scores(reference, generated).items()
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
