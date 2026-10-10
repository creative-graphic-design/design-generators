"""Require exact equality between two independent package cache preparations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    """Compare every prepared source, image, ID manifest, and vocabulary byte."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    args = parser.parse_args()
    first = _tree_hashes(args.first_run)
    second = _tree_hashes(args.second_run)
    if first != second:
        changed = sorted(
            key
            for key in first.keys() | second.keys()
            if first.get(key) != second.get(key)
        )
        raise ValueError(f"package cache replay differs for paths: {changed}")

    manifest = json.loads((args.first_run / "crello-manifest.json").read_text("utf-8"))
    print(
        json.dumps(
            {
                "stable": True,
                "files": len(first),
                "documents": manifest["documents"],
                "document_stage_png_count": manifest["document_stage_png_count"],
                "pixelvae_training_png_count": manifest["pixelvae_training_png_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
