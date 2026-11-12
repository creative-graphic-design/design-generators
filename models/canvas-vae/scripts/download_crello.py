"""Download and extract the byte-pinned Crello v1 source archive."""

from __future__ import annotations

import argparse

from canvas_vae.data import (
    CRELLO_V1_SIZE,
    download_crello_archive,
    extract_crello_archive,
    verify_crello_archive,
)


def main() -> None:
    """Fetch only the verified archive and extract it to an explicit path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", required=True, help="path for the private source zip"
    )
    parser.add_argument(
        "--extract-to", required=True, help="new directory for extracted records"
    )
    args = parser.parse_args()
    archive = download_crello_archive(args.archive)
    digest = verify_crello_archive(archive)
    extract_crello_archive(archive, args.extract_to)
    print(f"verified bytes={CRELLO_V1_SIZE} sha256={digest}")


if __name__ == "__main__":
    main()
