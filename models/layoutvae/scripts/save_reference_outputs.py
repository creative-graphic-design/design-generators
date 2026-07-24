"""Save LayoutVAE reference-output metadata for parity runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from layoutvae.conversion import load_original_modules


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FixedLatents(nn.Module):
    def __init__(self, latents: torch.Tensor) -> None:
        super().__init__()
        self.latents: torch.Tensor
        self.register_buffer("latents", latents)
        self.index = 0

    def forward(self, _inputs: object) -> torch.Tensor:
        value = self.latents[:, self.index, :]
        self.index += 1
        return value


class _FixedBboxNoise(nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise: torch.Tensor
        self.register_buffer("noise", noise)
        self.index = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs + self.noise[:, self.index, :]
        self.index += 1
        return value


class _FixedPoisson:
    def __init__(self, samples: torch.Tensor) -> None:
        self.samples = samples
        self.index = 0

    def __call__(self, _rate: torch.Tensor) -> "_FixedPoisson":
        return self

    def sample(self) -> torch.Tensor:
        value = self.samples[:, self.index].view(-1, 1)
        self.index += 1
        return value


def _fixed_inputs() -> dict[str, torch.Tensor]:
    return {
        "label_set": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
        ),
        "count_latents": torch.linspace(-0.75, 0.75, steps=2 * 6 * 32).reshape(
            2, 6, 32
        ),
        "count_samples": torch.tensor(
            [[0.0, 2.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 3.0, 4.0, 0.0, 0.0]]
        ),
        "bbox_latents": torch.linspace(-0.5, 0.5, steps=2 * 9 * 32).reshape(2, 9, 32),
        "bbox_noise": torch.linspace(0.0, 0.019, steps=2 * 9 * 4).reshape(2, 9, 4),
    }


def _labels_from_counts(
    class_counts: torch.Tensor, *, max_box: int = 9
) -> torch.Tensor:
    labels = torch.zeros(class_counts.shape[0], max_box, class_counts.shape[1])
    for batch_index, counts in enumerate(class_counts):
        position = 0
        for class_index in reversed(range(class_counts.shape[1])):
            for _ in range(int(counts[class_index].item())):
                if position >= max_box:
                    break
                labels[batch_index, position, class_index] = 1.0
                position += 1
    return labels


def _normalize_counts(class_counts: torch.Tensor, *, max_box: int = 9) -> torch.Tensor:
    counts = class_counts.clamp_min(0)
    denom = counts.sum(dim=1, keepdim=True).clamp_min(1)
    counts = torch.floor(max_box * (counts / denom))
    totals = counts.sum(dim=1)
    shortfall = max_box - totals
    counts[:, 0] = counts[:, 0] + torch.clamp(shortfall, min=0)
    return counts


def _run_original_count(
    count_module: nn.Module,
    label_set: torch.Tensor,
    count_latents: torch.Tensor,
    count_samples: torch.Tensor,
) -> torch.Tensor:
    count_module.eval()
    count_module.rep = _FixedLatents(count_latents)
    module = sys.modules[count_module.__class__.__module__]
    original_poisson = getattr(module, "Poisson")
    setattr(module, "Poisson", _FixedPoisson(count_samples))
    try:
        with torch.no_grad():
            return count_module(label_set, isTrain=False)
    finally:
        setattr(module, "Poisson", original_poisson)


def _run_original_bbox(
    bbox_module: nn.Module,
    class_counts: torch.Tensor,
    class_labels: torch.Tensor,
    bbox_latents: torch.Tensor,
    bbox_noise: torch.Tensor,
) -> torch.Tensor:
    bbox_module.eval()
    bbox_module.rep = _FixedLatents(bbox_latents)
    bbox_module.rep_mul = _FixedBboxNoise(bbox_noise)
    with torch.no_grad():
        output = bbox_module([class_counts, class_labels], isTrain=False)
    return output.permute(2, 0, 1)


def main() -> None:
    """Run the reference metadata CLI."""
    parser = argparse.ArgumentParser(
        description="Write LayoutVAE parity metadata and fixed label-set fixtures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("vendor/layout-generation-baselines/LayoutVAE"),
        help="Path to the LayoutVAE source root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutvae/reference"),
        help="Directory for reference metadata.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reference seed.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_inputs = _fixed_inputs()
    label_set = fixed_inputs["label_set"]
    torch.save(
        {"label_set": label_set, "seed": args.seed}, args.output_dir / "fixture.pt"
    )
    count_module, bbox_module = load_original_modules(args.source_root)
    class_counts = _run_original_count(
        count_module,
        label_set,
        fixed_inputs["count_latents"],
        fixed_inputs["count_samples"],
    )
    normalized_counts = _normalize_counts(class_counts)
    class_labels = _labels_from_counts(normalized_counts)
    raw_ltwh = _run_original_bbox(
        bbox_module,
        normalized_counts,
        class_labels,
        fixed_inputs["bbox_latents"],
        fixed_inputs["bbox_noise"],
    )
    torch.save(
        {
            **fixed_inputs,
            "class_counts": class_counts,
            "normalized_counts": normalized_counts,
            "class_labels": class_labels,
            "raw_ltwh": raw_ltwh,
        },
        args.output_dir / "fixed_forward.pt",
    )
    trained = args.source_root / "TrainedModel"
    metadata = {
        "seed": args.seed,
        "torch_version": torch.__version__,
        "fixtures": [["text", "figure"], ["title", "list"]],
        "count_checkpoint_sha256": _sha256(trained / "countvae.h5"),
        "bbox_checkpoint_sha256": _sha256(trained / "bboxvae.h5"),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
