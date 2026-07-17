"""Generate vendor LayoutFlow reference tensors used to audit parity locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import torch

from laygen.common.labels import DatasetName
from laygen.common.vendor import vendor_root
from layout_flow import LayoutFlowConfig
from layout_flow.configuration_layout_flow import normalize_dataset_name


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_ORIGINAL_DIR: Final[Path] = (
    REPO_ROOT / ".cache" / "layout-flow" / "original" / "checkpoints"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / ".cache" / "layout-flow" / "golden"
CHECKPOINT_NAMES: Final[dict[DatasetName, str]] = {
    DatasetName.publaynet: "checkpoint_PubLayNet_LayoutFlow.ckpt",
    DatasetName.rico25: "checkpoint_RICO_LayoutFlow.ckpt",
}
ALL_DATASETS: Final[str] = "all"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original LayoutFlow vendor backbone on deterministic inputs "
            "and write golden vector-field tensors plus metadata for parity review."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=[*sorted(str(dataset) for dataset in CHECKPOINT_NAMES), ALL_DATASETS],
        default=ALL_DATASETS,
        help="Dataset/checkpoint variant to generate, or all variants.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help="Directory containing the original LayoutFlow .ckpt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where golden .pt tensors and summary JSON are written.",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path("vendor/layout-flow"),
        help="Path to the read-only original LayoutFlow source checkout.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for vendor inference; CUDA_VISIBLE_DEVICES controls CUDA mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic synthetic parity inputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vendor_dir = vendor_root(
        "layout-flow",
        marker=Path("src/models/backbone/layoutdm_backbone.py"),
        path=args.vendor_dir,
    )
    sys.path.insert(0, str(vendor_dir))
    from src.models.backbone.layoutdm_backbone import LayoutDMBackbone

    datasets = (
        sorted(CHECKPOINT_NAMES, key=str)
        if args.dataset == ALL_DATASETS
        else [normalize_dataset_name(args.dataset)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset in datasets:
        dataset_value = str(dataset)
        config = LayoutFlowConfig(dataset_name=dataset_value)
        checkpoint = args.checkpoint_dir / CHECKPOINT_NAMES[dataset]
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw["state_dict"]
        vendor = LayoutDMBackbone(
            latent_dim=config.latent_dim,
            tr_enc_only=config.tr_enc_only,
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_pos_enc=config.use_pos_enc,
            num_cat=config.num_labels,
            attr_encoding=config.attr_encoding,
            seq_type=config.seq_type,
        ).to(args.device)
        vendor.load_state_dict(
            {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
        )
        vendor.eval()
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        sample = torch.randn(
            2,
            4,
            config.sample_dim,
            generator=generator,
            device=args.device,
            dtype=torch.float32,
        )
        cond_mask = torch.randint(
            0,
            2,
            (2, 4, config.sample_dim),
            generator=generator,
            device=args.device,
            dtype=torch.long,
        )
        timestep = torch.tensor([0.25, 0.75], device=args.device, dtype=torch.float32)
        with torch.no_grad():
            vector = vendor(sample[:, :, :4], sample[:, :, 4:], cond_mask, timestep)
        output_path = args.output_dir / f"{dataset_value}_vendor_vector_field.pt"
        torch.save(
            {
                "dataset": dataset_value,
                "seed": args.seed,
                "sample": sample.cpu(),
                "cond_mask": cond_mask.cpu(),
                "timestep": timestep.cpu(),
                "vector": vector.cpu(),
            },
            output_path,
        )
        summary.append(
            {
                "dataset": dataset_value,
                "checkpoint": str(checkpoint),
                "output": str(output_path),
                "shape": list(vector.shape),
            }
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
