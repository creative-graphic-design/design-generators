"""Download the RICO semantic-annotation archive and verify its MD5 checksum."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

RICO_URL = "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip"
RICO_MD5 = "5dd3372e2d99b342958136e394d4de79"


def md5sum(path: Path) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Download the archive unless a verified copy already exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/canvas-vae/data/semantic_annotations.zip"),
        help="Archive destination (default: %(default)s).",
    )
    args = parser.parse_args()
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(".part")
        with urllib.request.urlopen(RICO_URL) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.rename(args.output)

    checksum = md5sum(args.output)
    if checksum != RICO_MD5:
        raise SystemExit(f"MD5 mismatch for {args.output}: {checksum} != {RICO_MD5}")

    print(f"{args.output} md5={checksum}")


if __name__ == "__main__":
    main()
