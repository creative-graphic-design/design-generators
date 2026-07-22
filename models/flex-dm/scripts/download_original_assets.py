"""Download Flex-DM original GCS assets into a local cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

ASSETS = {
    ("weights", "crello"): (
        "https://storage.googleapis.com/ailab-public/flexdm/pretrained_weights/crello.zip",
        62_037_582,
    ),
    ("weights", "rico"): (
        "https://storage.googleapis.com/ailab-public/flexdm/pretrained_weights/rico.zip",
        51_152_425,
    ),
    ("data", "crello"): (
        "https://storage.googleapis.com/ailab-public/flexdm/preprocessed_data/crello.zip",
        3_661_085_903,
    ),
    ("data", "rico"): (
        "https://storage.googleapis.com/ailab-public/flexdm/preprocessed_data/rico.zip",
        26_714_323,
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/flex-dm/original")
    )
    parser.add_argument(
        "--dataset", choices=["crello", "rico", "all"], default="crello"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=["weights", "data"],
        default=["weights"],
    )
    parser.add_argument("--no-unpack", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Download requested assets and optionally unpack them."""
    args = parse_args()
    datasets = ["crello", "rico"] if args.dataset == "all" else [args.dataset]
    for asset_kind in args.assets:
        for dataset in datasets:
            url, expected_size = ASSETS[(asset_kind, dataset)]
            zip_dir = args.output_dir / "zips"
            zip_dir.mkdir(parents=True, exist_ok=True)
            zip_path = zip_dir / f"{asset_kind}-{dataset}.zip"
            if not zip_path.exists():
                print(f"Downloading {url} -> {zip_path}")
                urlretrieve(url, zip_path)
            size = zip_path.stat().st_size
            if size != expected_size:
                raise ValueError(
                    f"{zip_path} has {size} bytes, expected {expected_size}"
                )
            if args.no_unpack:
                continue
            target = args.output_dir / asset_kind / dataset
            target.mkdir(parents=True, exist_ok=True)
            with ZipFile(zip_path) as archive:
                archive.extractall(target)
            print(f"unpacked {zip_path} -> {target}")


if __name__ == "__main__":
    main()
