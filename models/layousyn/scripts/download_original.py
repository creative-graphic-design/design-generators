"""Download original Lay-Your-Scene checkpoint assets."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main(
    repo_id: str = "dsrivastavv/Lay-Your-Scene",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Download a Hugging Face snapshot for conversion/parity."""
    return snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


if __name__ == "__main__":
    print(main())
