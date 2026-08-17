"""Generate RADM original-code reference outputs from local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Record metadata for a future RADM original-code reference run. "
            "The heavyweight Detectron2 execution requires local vendor code, "
            "checkpoint, dataset, text features, and a selected GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vendor-root", type=Path, required=True, help="Local RADM vendor source root."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Local RADM checkpoint."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Local CGL dataset root."
    )
    parser.add_argument(
        "--text-feature-root", type=Path, required=True, help="Local text-feature root."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Reference metadata directory."
    )
    parser.add_argument("--seed", type=int, default=1, help="Reference seed.")
    parser.add_argument("--device", default="cuda:0", help="Selected reference device.")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Write reference metadata and explain the gated execution boundary."""
    args = build_parser().parse_args()
    marker = args.vendor_root / "train_net.py"
    if not marker.exists():
        raise RuntimeError(
            "RADM vendor execution is unavailable: expected train_net.py under --vendor-root"
        )
    for path in (args.checkpoint, args.dataset_root, args.text_feature_root):
        if not path.exists():
            raise RuntimeError(f"Missing required local asset: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": args.device,
        "seed": args.seed,
        "text_feature_root": str(args.text_feature_root),
        "vendor_root": str(args.vendor_root),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "metadata written; run the original Detectron2 evaluation wrapper in a GPU environment"
    )


if __name__ == "__main__":
    main()
