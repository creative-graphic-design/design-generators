"""Record and verify the source tree used by sequential training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


def _git_output(source_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in the source tree")

    return result.stdout


def source_tree_sha256(source_root: Path) -> tuple[str, int]:
    index_entries = _git_output(source_root, "ls-files", "--stage", "-z").split(b"\0")
    untracked_paths = _git_output(
        source_root, "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    digest = hashlib.sha256()
    file_count = 0

    source_entries = []
    for entry in index_entries:
        if not entry:
            continue
        metadata, relative_path = entry.split(b"\t", 1)
        mode, object_id, _stage = metadata.split()
        if mode == b"160000":
            source_entries.append((relative_path, b"gitlink", object_id))
        else:
            source_entries.append((relative_path, b"file", b""))
    source_entries.extend(
        (relative_path, b"file", b"")
        for relative_path in untracked_paths
        if relative_path
    )

    for relative_path, entry_kind, object_id in sorted(set(source_entries)):
        digest.update(relative_path + b"\0" + entry_kind + b"\0")
        if entry_kind == b"gitlink":
            submodule_root = source_root / os.fsdecode(relative_path)
            if (submodule_root / ".git").exists():
                submodule_hash, submodule_file_count = source_tree_sha256(
                    submodule_root
                )
                digest.update(submodule_hash.encode() + b"\0")
                file_count += submodule_file_count
            else:
                digest.update(object_id + b"\0")
            file_count += 1
            continue

        path = source_root / os.fsdecode(relative_path)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            digest.update(b"deleted\0")
            file_count += 1
            continue

        digest.update(f"{stat.S_IMODE(metadata.st_mode):o}".encode() + b"\0")
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(os.fsencode(os.readlink(path)))
        elif stat.S_ISREG(metadata.st_mode):
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise RuntimeError(f"unsupported source-tree entry: {relative_path!r}")
        digest.update(b"\0")
        file_count += 1

    return digest.hexdigest(), file_count


def _source_commit(source_root: Path) -> str:
    return _git_output(source_root, "rev-parse", "HEAD").decode().strip()


def record_seed_zero(source_root: Path, manifest_path: Path) -> None:
    tree_hash, file_count = source_tree_sha256(source_root)
    manifest = {
        "schema_version": 1,
        "recorded_seed": 0,
        "source_commit": _source_commit(source_root),
        "source_tree_sha256": tree_hash,
        "file_count": file_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "recorded", **manifest}, sort_keys=True))


def verify_later_seed(source_root: Path, manifest_path: Path, seed: int) -> None:
    if seed <= 0:
        raise ValueError("verification requires a seed after seed 0")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("recorded_seed") != 0:
        raise ValueError("source-tree manifest must be recorded at seed 0")

    tree_hash, file_count = source_tree_sha256(source_root)
    if (
        manifest.get("source_tree_sha256") != tree_hash
        or manifest.get("file_count") != file_count
    ):
        raise ValueError(f"source-tree hash changed before seed {seed}")

    print(
        json.dumps(
            {
                "status": "verified",
                "seed": seed,
                "source_tree_sha256": tree_hash,
                "file_count": file_count,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("record", "verify"):
        command = subparsers.add_parser(action)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if action == "record":
            command.add_argument("--seed", type=int, choices=(0,), required=True)
        else:
            command.add_argument("--seed", type=int, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    try:
        if args.action == "record":
            record_seed_zero(source_root, manifest_path)
        else:
            verify_later_seed(source_root, manifest_path, args.seed)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"source-tree guard failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
