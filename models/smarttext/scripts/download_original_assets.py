"""Download original SmartText assets.

The BASNet/GDI checkpoint is large enough to trigger Google Drive's confirmation
flow. This script prefers ``gdown`` for both files and records a manifest with
file sizes and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

SMT_FILE_ID = "1zKVA9IGkPtmRkm-2_m7qriaEwVXBuaGX"
BASNET_FILE_ID = "1dN_lqywxefd_R4Q93lZck0kEkfKo-wkj"


def main() -> None:
    """Run the command-line downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/smarttext/original")
    )
    parser.add_argument(
        "--download", action="store_true", help="Actually download files with gdown."
    )
    parser.add_argument(
        "--smt-path", type=Path, default=None, help="Existing SMT.pth path to record."
    )
    parser.add_argument(
        "--basnet-path",
        type=Path,
        default=None,
        help="Existing gdi-basnet.pth path to record.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smt_path = args.smt_path or args.output_dir / "SMT.pth"
    basnet_path = args.basnet_path or args.output_dir / "gdi-basnet.pth"
    if args.download:
        import gdown

        gdown.download(id=SMT_FILE_ID, output=str(smt_path), quiet=False)
        gdown.download(id=BASNET_FILE_ID, output=str(basnet_path), quiet=False)
    manifest = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "assets": {
            "SMT.pth": _file_record(smt_path, SMT_FILE_ID),
            "gdi-basnet.pth": _file_record(basnet_path, BASNET_FILE_ID),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _file_record(path: Path, file_id: str) -> dict[str, object]:
    if not path.exists():
        return {"file_id": file_id, "path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "file_id": file_id,
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    main()
