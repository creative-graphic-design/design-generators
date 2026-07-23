"""Validate or fetch House-GAN original Dropbox assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Write an asset manifest for a manually downloaded assets directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir", required=True, help="Directory containing Dropbox assets"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory receiving asset_manifest.json"
    )
    parser.add_argument(
        "--dropbox-url",
        default="https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=1",
    )
    args = parser.parse_args()
    assets_dir = Path(args.assets_dir)
    files = sorted(path for path in assets_dir.rglob("*") if path.is_file())
    manifest = {
        "source_url": args.dropbox_url,
        "assets_dir": str(assets_dir),
        "files": [
            {
                "path": str(path.relative_to(assets_dir)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "has_dataset": any(
            path.name in {"train_data.npy", "housegan_clean_data.npy"} for path in files
        ),
        "checkpoints": [
            str(path.relative_to(assets_dir)) for path in files if path.suffix == ".pth"
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
