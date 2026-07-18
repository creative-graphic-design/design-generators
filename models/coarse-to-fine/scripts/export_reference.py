"""Generate local vendor-reference tensors for Coarse-to-Fine parity tests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch


class VendorInferenceModel(Protocol):
    """Vendor model surface used by this local reference exporter."""

    def inference(
        self, device: torch.device, z: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class VendorConfig:
    """Minimal config namespace expected by the vendor model."""

    num_labels: int
    max_num_elements: int = 20
    discrete_x_grid: int = 128
    discrete_y_grid: int = 128
    d_model: int = 512
    d_z: int = 512
    n_layers: int = 4
    n_layers_decoder: int = 4
    n_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    eval_batch_size: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["rico25", "publaynet"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/ms-layout-generation/Coarse-to-Fine"),
        help="Initialized Coarse-to-Fine vendor checkout.",
    )
    return parser.parse_args()


def _vendor_model_class(vendor_root: Path) -> type[torch.nn.Module]:
    sys.path.insert(0, str(vendor_root.resolve()))
    from coarse2fine.c2f_model.model import C2FLayoutTransformer

    return cast(type[torch.nn.Module], C2FLayoutTransformer)


def _load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(checkpoint, map_location="cpu")
    return {str(key).removeprefix("module."): value for key, value in raw.items()}


def main() -> None:
    """Run vendor inference with a fixed latent and save reference tensors."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_labels = 25 if args.dataset == "rico25" else 5
    config = VendorConfig(num_labels=num_labels, eval_batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model_cls = _vendor_model_class(args.vendor_root)
    model = model_cls(config).to(device)
    model.load_state_dict(_load_state(args.checkpoint), strict=True)
    model.eval()
    latent_z = torch.randn((args.batch_size, 1, config.d_z), device=device)
    with torch.no_grad():
        generated = cast(VendorInferenceModel, model).inference(device, z=latent_z)
    artifact = args.output_dir / "reference.pt"
    torch.save(
        {
            "latent_z": latent_z.cpu(),
            "group_bounding_box_logits": generated["group_bounding_box"].cpu(),
            "label_in_one_group_logits": generated["label_in_one_group"].cpu(),
            "grouped_bbox_logits": generated["grouped_bboxes"].cpu(),
            "grouped_label_logits": generated["grouped_labels"].cpu(),
        },
        artifact,
    )
    metadata = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "artifact": str(artifact),
        "device": str(device),
        "config": asdict(config),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(artifact)


if __name__ == "__main__":
    main()
