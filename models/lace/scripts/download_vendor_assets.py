from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".cache/lace/original")
    parser.add_argument("--filename", default="model.tar.gz")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="puar-playground/LACE",
        filename=args.filename,
        repo_type="dataset",
        local_dir=output_dir,
    )
    if args.extract:
        with tarfile.open(path) as archive:
            archive.extractall(output_dir)
    print(path)


if __name__ == "__main__":
    main()
