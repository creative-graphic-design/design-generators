"""Score a CanvasVAE pipeline on the RICO test split.

Reconstruction encodes every test layout, decodes the posterior mean with the
predicted element count, and averages per-layout field BLEU scores
(``reconst_total`` is their mean), component-grid pixel accuracy, and mean IoU.
Random generation decodes one standard-normal latent per test layout for each
evaluation seed and scores field histograms against the test split by
histogram intersection (``random_total`` is their mean).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from laygen.common.randomness import randn

from canvas_vae import CanvasVAEPipeline, CanvasVAEProcessor
from canvas_vae.metrics import field_histograms, histogram_scores, layout_scores, reconstruction_scores
from canvas_vae.modeling_canvas_vae import length_mask
from canvas_vae.processing_canvas_vae import load_rico_split
from canvas_vae.training import sequential_batches


@torch.no_grad()
def main() -> None:
    """Write evaluation results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory written by save_pretrained.")
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/canvas-vae/data/rico"), help="Prepared split directory (default: %(default)s).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random-generation evaluation seeds (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Reconstruction batch size (default: %(default)s).")
    parser.add_argument("--device", default="cpu", help="Torch device (default: %(default)s).")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path.")
    args = parser.parse_args()
    model = CanvasVAEPipeline.from_pretrained(args.checkpoint, local_files_only=True).model.to(args.device).eval()
    config = model.config
    processor = CanvasVAEProcessor(config.vocabularies, num_bins=config.num_bins)
    documents = load_rico_split(args.data_dir, "test")
    sums: dict[str, float] = {}
    kl_values = []
    for batch in sequential_batches(len(documents), args.batch_size):
        encoded = processor([documents[index]["elements"] for index in batch]).to(args.device)
        output = model(encoded["num_elements"], encoded["element_ids"])
        predicted = torch.stack([logits.argmax(-1) for logits in output.element_logits.values()], dim=-1)
        target_mask = length_mask(encoded["num_elements"] - 1, encoded["element_ids"].shape[1])
        scores = reconstruction_scores(encoded["element_ids"], target_mask, predicted, output.mask, config.field_sizes)
        scores |= layout_scores(encoded["element_ids"], target_mask, predicted, output.mask, grid_size=config.num_bins, num_labels=config.field_sizes["component"], background_id=config.primary_label_id)
        for key, values in scores.items():
            sums[key] = sums.get(key, 0.0) + float(values.sum())
        kl_values.append(float(output.kl_divergence))

    results = {f"reconst_{key}": value / len(documents) for key, value in sums.items()}
    results["reconst_kl_divergence"] = sum(kl_values) / len(kl_values)
    reference = processor([document["elements"] for document in documents])
    reference_histograms = field_histograms(reference["num_elements"], reference["element_ids"], config.field_sizes, config.max_length)
    for seed in args.seeds:
        latents = randn((len(documents), config.latent_dim), generator=torch.Generator().manual_seed(seed), device=args.device)
        output = model(latents=latents)
        predicted = torch.stack([logits.argmax(-1) for logits in output.element_logits.values()], dim=-1)
        generated = field_histograms(output.mask.sum(-1), predicted, config.field_sizes, config.max_length)
        results |= {f"random_seed{seed}_{key}": value for key, value in histogram_scores(reference_histograms, generated).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
