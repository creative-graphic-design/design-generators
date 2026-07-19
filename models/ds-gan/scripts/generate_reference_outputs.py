"""Generate DS-GAN vendor-reference fixtures for parity tests."""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import DataLoader

from laygen.common.vendor import vendor_root

_VENDOR_MODEL: Final[Path] = Path("model.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the vendored PosterLayout DS-GAN generator with fixed NumPy and "
            "Torch seeds, then save raw class probabilities, boxes, and metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vendor-dir", type=Path, default=Path("vendor/posterlayout-cvpr2023")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".cache/ds-gan/original/DS-GAN-Epoch300.pth"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path(".cache/ds-gan/original/Dataset/test")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/ds-gan/fixtures/pku/reference_seed0.pt"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.current_device() != 0:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    vendor_dir = vendor_root(
        "posterlayout-cvpr2023", marker=_VENDOR_MODEL, path=args.vendor_dir
    )
    sys.path.insert(0, str(vendor_dir))
    from dataloader import canvas
    from infer import random_init
    from model import generator

    asset_root = args.checkpoint.resolve().parent
    _prepare_model_weight_dir(asset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = canvas(
        args.dataset_root / "image_canvas",
        args.dataset_root / "saliencymaps_pfpn",
        args.dataset_root / "saliencymaps_basnet",
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    config = {
        "backbone": "resnet50",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": 32 * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": 32,
    }
    with _pushd(asset_root):
        model = generator(config).eval().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = OrderedDict(
        (key.removeprefix("module."), value) for key, value in checkpoint.items()
    )
    model.load_state_dict(state_dict, strict=True)
    fixed_noise = random_init(args.batch_size, 32).to(device)
    pixel_batches = []
    initial_layouts = []
    class_probs = []
    boxes = []
    with torch.no_grad():
        for pixel_values in loader:
            batch_noise = fixed_noise[: pixel_values.shape[0]]
            cls, bbox = model(pixel_values.to(device), batch_noise)
            pixel_batches.append(pixel_values.cpu())
            initial_layouts.append(batch_noise.detach().cpu())
            class_probs.append(cls.detach().cpu())
            boxes.append(bbox.detach().cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "pixel_values": torch.cat(pixel_batches),
            "initial_layout": torch.cat(initial_layouts),
            "class_probs": torch.cat(class_probs),
            "bbox": torch.cat(boxes),
            "checkpoint": str(args.checkpoint),
            "vendor_commit": "44ee576471f3c06d06fc632f13e3f96d0c381847",
        },
        args.output,
    )
    print(args.output)


def _prepare_model_weight_dir(asset_root: Path) -> None:
    model_weight = asset_root / "model_weight"
    model_weight.mkdir(parents=True, exist_ok=True)
    for filename in ("resnet18-5c106cde.pth", "resnet50_a1_0-14fe96d1.pth"):
        source = asset_root / filename
        target = model_weight / filename
        if source.exists() and not target.exists():
            target.symlink_to(Path("..") / filename)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
