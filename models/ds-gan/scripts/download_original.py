"""Download original DS-GAN assets used by conversion and parity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import gdown

WEIGHTS_FOLDER_URL: Final[str] = (
    "https://drive.google.com/drive/folders/1UYJ34BhqgYztfh5n5A4GU4nqgboPtoWS"
)
TEST_DATA_URLS: Final[dict[str, str]] = {
    # Google Drive links listed on the PKU PosterLayout dataset page.
    "image_canvas": "https://drive.google.com/uc?id=1hcXueYYh2iY5XLtyTZFsXUZsI5JwFnaT",
    "saliencymaps_basnet": "https://drive.google.com/uc?id=1rSsIvoPfkj1s9W2wMq2jSIFnZuw4iEo7",
    "saliencymaps_pfpn": "https://drive.google.com/uc?id=1FDRU-2FFZHK2IZe83Py469MCAydVRKzU",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download original PosterLayout DS-GAN weights and, optionally, the "
            "public PKU PosterLayout test canvases/saliency maps used for parity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/ds-gan/original"),
        help="Directory where manually downloaded assets should be placed.",
    )
    parser.add_argument(
        "--include-test-data",
        action="store_true",
        help=(
            "Also download and extract public PKU PosterLayout test images and "
            "saliency maps. Layout annotations still require the upstream "
            "release agreement and are not downloaded by this script."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        url=WEIGHTS_FOLDER_URL,
        output=str(args.output_dir),
        quiet=False,
        resume=True,
    )
    if args.include_test_data:
        _download_test_data(args.output_dir)
    print(args.output_dir)


def _download_test_data(output_dir: Path) -> None:
    downloads = output_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    test_root = output_dir / "Dataset" / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    for name, url in TEST_DATA_URLS.items():
        archive = downloads / f"{name}.zip"
        gdown.download(url=url, output=str(archive), quiet=False, resume=True)
        with ZipFile(archive) as zip_file:
            zip_file.extractall(test_root)


if __name__ == "__main__":
    main()
