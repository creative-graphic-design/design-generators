"""Download original LayoutDETR assets from the vendor Google Drive links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ASSETS = {
    "model": {
        "id": "1iaKATX2Id9JnqDunDytVIK5l9HO0a0w-",
        "filename": "checkpoints/layoutdetr_ad_banner.pkl",
    },
    "up-detr": {
        "id": "1JhL1uwNJCaxMrIUx7UzQ3CMCHqmZpCnn",
        "filename": "pretrained/up-detr.pth",
    },
    "dataset": {
        "id": "1T09t4dX7zQ7J-8KxtJv1RkyjRNdilD1m",
        "filename": "data/ads_banner_dataset.zip",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        choices=["model", "up-detr", "dataset", "all"],
        action="append",
        default=["model"],
    )
    args = parser.parse_args()
    includes = set(ASSETS) if "all" in args.include else set(args.include)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit("Install with --extra download to use gdown") from exc
    for key in sorted(includes):
        spec = ASSETS[key]
        destination = args.output_dir / spec["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={spec['id']}"
        gdown.download(url, str(destination), quiet=False)
        manifest.append(
            {
                "asset": key,
                "google_drive_id": spec["id"],
                "path": str(destination),
                "size_bytes": destination.stat().st_size
                if destination.exists()
                else None,
            }
        )
    (args.output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
