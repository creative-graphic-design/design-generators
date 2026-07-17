from __future__ import annotations

import argparse
from pathlib import Path

from layout_corrector.conversion import discover_seed_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--starter-dir", type=Path, required=True)
    parser.add_argument("--corrector-job-dir", type=Path, required=True)
    parser.add_argument("--layout-dm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()
    seed_dirs = discover_seed_dirs(args.corrector_job_dir)
    if not seed_dirs:
        raise FileNotFoundError(
            f"No corrector seed dirs found in {args.corrector_job_dir}"
        )
    raise NotImplementedError(
        "Checkpoint tensor conversion is intentionally gated until starter-kit "
        "LayoutDM artifacts are available locally for parity validation."
    )


if __name__ == "__main__":
    main()
