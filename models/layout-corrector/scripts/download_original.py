from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown


FILE_ID = "1og3l0enR67rDwiAN44K4RchcFYAgsbNq"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "layout_corrector_starter.zip"
    gdown.download(id=FILE_ID, output=str(archive), quiet=False)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(args.output_dir)


if __name__ == "__main__":
    main()
