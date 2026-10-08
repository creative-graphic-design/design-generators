"""Freeze PixelVAE CPU tolerances from three independent calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from pixel_vae.testing import calibrate_limits

DEFAULT_INPUT_DIR: Final[Path] = Path(".cache/pixel-vae/parity/calibration")
DEFAULT_OUTPUT: Final[Path] = DEFAULT_INPUT_DIR / "limits.json"


def parse_args() -> argparse.Namespace:
    """Parse calibration input and output paths."""
    parser = argparse.ArgumentParser(
        description="Freeze pre-registered PixelVAE CPU limits from three independent repeats."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"directory containing repeat-1..3.json (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"frozen limits JSON path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Validate repeat independence and write the frozen numeric limits."""
    args = parse_args()
    records = []
    for repeat in range(1, 4):
        path = args.input_dir / f"repeat-{repeat}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") != "calibration" or record.get("repeat") != repeat:
            raise ValueError(f"unexpected calibration record metadata: {path}")
        records.append((path, record))

    digests = {record.get("encoder_state_sha256") for _, record in records}
    process_ids = {record.get("process_id") for _, record in records}
    source_commits = {record.get("source_commit") for _, record in records}
    versions = {record.get("tensorflow_version") for _, record in records}
    configurations = {
        json.dumps(record.get("configuration"), sort_keys=True) for _, record in records
    }
    input_ids = {
        json.dumps(
            {
                key: record.get(key)
                for key in (
                    "s1_image_ids",
                    "s2_image_ids",
                    "s3_image_ids",
                    "s4_image_ids",
                )
            },
            sort_keys=True,
        )
        for _, record in records
    }
    if len(digests) != 1 or None in digests:
        raise ValueError("calibration repeats do not share one encoder-state digest")
    if len(process_ids) != 3 or None in process_ids:
        raise ValueError(
            "calibration repeats must come from three independent processes"
        )
    if len(source_commits) != 1 or None in source_commits:
        raise ValueError("calibration repeats do not share one vendor source commit")
    if len(versions) != 1 or None in versions:
        raise ValueError("calibration repeats do not share one TensorFlow version")
    if len(configurations) != 1 or "null" in configurations:
        raise ValueError(
            "calibration repeats do not share the production configuration"
        )
    if len(input_ids) != 1:
        raise ValueError("calibration repeats do not use the same ordered input IDs")

    metric_repeats = [record["metrics"] for _, record in records]
    limits = calibrate_limits(metric_repeats)
    result = {
        "formula": "ceil2(1.5 * max(repeat 1, repeat 2, repeat 3)); ceil2 rounds upward to two significant figures",
        "encoder_state_sha256": next(iter(digests)),
        "source_commit": next(iter(source_commits)),
        "repeat_process_ids": sorted(process_ids),
        "repeat_artifacts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path, _ in records
        },
        "repeat_metrics": metric_repeats,
        "limits": limits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
