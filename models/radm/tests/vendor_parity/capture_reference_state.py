"""Capture original RADM initialized training state without running training."""

from __future__ import annotations

import argparse
from pathlib import Path

from reference_adapter import RADMReferenceAdapter, write_reference_metadata


def main() -> None:
    """Build the selected reference graph and write metadata to a cache path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--text-feature-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = RADMReferenceAdapter(
        vendor_root=args.vendor_root,
        dataset_root=args.dataset_root,
        text_feature_root=args.text_feature_root,
        device=args.device,
    ).build_initialized_state()
    write_reference_metadata(state, args.output)
    print(f"reference initialized-state metadata written to {args.output}")


if __name__ == "__main__":
    main()
