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
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from laygen.common.vendor import vendor_root

_VENDOR_MODEL: Final[Path] = Path("model.py")
_BILINEAR: Final = Image.Resampling.BILINEAR


class PosterLayoutTestCanvas(Dataset[torch.Tensor]):
    """Vendor-equivalent test canvas loader without CWD side effects."""

    def __init__(
        self,
        *,
        image_canvas_dir: Path,
        saliency_pfpn_dir: Path,
        saliency_basnet_dir: Path,
    ) -> None:
        """Initialize sorted file paths for public PKU test assets."""
        self.image_paths = sorted(image_canvas_dir.glob("*.png"), key=lambda p: p.name)
        self.saliency_pfpn_dir = saliency_pfpn_dir
        self.saliency_basnet_dir = saliency_basnet_dir

    def __len__(self) -> int:
        """Return the number of test canvases."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a vendor-preprocessed RGB plus saliency tensor."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        saliency_pfpn = Image.open(
            self.saliency_pfpn_dir / image_path.name.replace(".png", "_pred.png")
        ).convert("L")
        saliency_basnet = Image.open(
            self.saliency_basnet_dir / image_path.name
        ).convert("L")
        saliency = Image.fromarray(
            np.maximum(np.asarray(saliency_pfpn), np.asarray(saliency_basnet))
        )
        return torch.cat(
            (
                _pil_to_tensor(image, mode="RGB"),
                _pil_to_tensor(saliency, mode="L"),
            ),
            dim=0,
        )


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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    vendor_dir = vendor_root(
        "posterlayout-cvpr2023", marker=_VENDOR_MODEL, path=args.vendor_dir
    )
    sys.path.insert(0, str(vendor_dir))
    from model import generator

    asset_root = args.checkpoint.resolve().parent
    _prepare_model_weight_dir(asset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PosterLayoutTestCanvas(
        image_canvas_dir=args.dataset_root / "image_canvas",
        saliency_pfpn_dir=args.dataset_root / "saliencymaps_pfpn",
        saliency_basnet_dir=args.dataset_root / "saliencymaps_basnet",
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
    fixed_noise = _random_init(args.batch_size, 32).to(device)
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


def _random_init(batch_size: int, max_elem: int) -> torch.Tensor:
    probs = np.array([0.1, 0.8, 1.0, 1.0])
    class_ids = torch.tensor(
        np.random.choice(4, size=(batch_size, max_elem, 1), p=probs / probs.sum())
    )
    classes = torch.zeros((batch_size, max_elem, 4))
    classes.scatter_(-1, class_ids, 1)
    box_ltrb = torch.normal(0.5, 0.15, size=(batch_size, max_elem, 1, 4))
    left, top, right, bottom = box_ltrb.unbind(-1)
    boxes = torch.stack(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dim=-1,
    )
    return torch.concat([classes.unsqueeze(2), boxes], dim=2)


def _pil_to_tensor(image: Image.Image, *, mode: str) -> torch.Tensor:
    resized = image.resize((240, 350), _BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if mode == "L":
        return torch.from_numpy(array).unsqueeze(0)
    return torch.from_numpy(array).permute(2, 0, 1)


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
