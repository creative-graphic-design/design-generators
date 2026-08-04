"""Generate BASNet vendor reference tensors outside git."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image

from basnet import BASNetImageProcessor, normalize_saliency


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/smarttext"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, default=Path("vendor/smarttext/test_data/SMT")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/basnet/references")
    )
    parser.add_argument("--max-images", type=int, default=3)
    return parser.parse_args()


def _image_paths(image_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return paths[:max_images]


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def main() -> None:
    """Run vendor BASNet reference generation."""
    args = parse_args()
    for path in (args.vendor_dir, args.checkpoint, args.image_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    _configure_torch_determinism()
    sys.path.insert(0, str(args.vendor_dir.resolve()))

    from BASNet.model import BASNet  # type: ignore[import-not-found]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BASNetImageProcessor()
    paths = _image_paths(args.image_dir, args.max_images)
    images = [Image.open(path).convert("RGB") for path in paths]
    batch = processor.preprocess(images)
    pixel_values = batch["pixel_values"].to(device)
    model = BASNet(3, 1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        saliency = normalize_saliency(model(pixel_values)[0][:, 0, :, :]).detach().cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "pixel_values": batch["pixel_values"],
        "saliency": saliency,
        "image_paths": [str(path.resolve()) for path in paths],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "note": "Reference generated with the original BASNet implementation.",
    }
    torch.save(reference, args.output_dir / "saliency.pt")
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "vendor_dir": str(args.vendor_dir),
        "checkpoint": str(args.checkpoint),
        "image_dir": str(args.image_dir),
        "image_paths": reference["image_paths"],
        "cuda_visible_devices": reference["cuda_visible_devices"],
        "device": reference["device"],
        "case_count": len(paths),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
