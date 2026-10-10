"""Freeze PixelVAE CPU tolerances from three independent calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pixel_vae.testing import JSONValue

DEFAULT_INPUT_DIR: Final[Path] = Path(".cache/pixel-vae/parity/calibration")
DEFAULT_OUTPUT: Final[Path] = DEFAULT_INPUT_DIR / "limits.json"
DEFAULT_LOWER_BOUNDS: Final[Path] = Path("models/pixel-vae/parity-limit-floors.json")
CALIBRATION_SELECTION_RULE: Final[str] = (
    "deduplicated canonical train PNGs in UTF-8 document_id/source element order; "
    "repeat r uses document-stage indexes [2(r-1), 2r) and filtered training-stage "
    "indexes [6(r-1), 6r); each element type uses unique index (r-1) mod n, recording "
    "its unique count and whether it wrapped; distinct image IDs are required only "
    "when that type has at least three unique PNGs"
)
LOWER_BOUNDS_SOURCE: Final[str] = (
    "Originally registered floor, not a limit frozen by failed held-out runs "
    "2b0c16e or b755f92"
)


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
    parser.add_argument(
        "--lower-bounds",
        type=Path,
        default=DEFAULT_LOWER_BOUNDS,
        help=f"registered metric floors JSON path (default: {DEFAULT_LOWER_BOUNDS})",
    )
    return parser.parse_args()


def main() -> None:
    """Validate repeat independence and write the frozen numeric limits."""
    from pixel_vae.testing import calibrate_limits

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen calibration limits: {args.output}"
        )

    records = []
    input_selection_paths = []
    s4_selections_by_repeat: list[dict[str, tuple[str, int]]] = []
    for repeat in range(1, 4):
        path = args.input_dir / f"repeat-{repeat}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") != "calibration" or record.get("repeat") != repeat:
            raise ValueError(f"unexpected calibration record metadata: {path}")
        records.append((path, record))

        selection = record.get("calibration_selection")
        expected_document_indices = list(range((repeat - 1) * 2, repeat * 2))
        expected_training_indices = list(range((repeat - 1) * 6, repeat * 6))
        if not isinstance(selection, dict):
            raise ValueError(f"calibration selection metadata is missing: {path}")
        selection_path = args.input_dir / f"repeat-{repeat}.inputs.json"
        selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_record != {"mode": "calibration", "repeat": repeat, **selection}:
            raise ValueError(
                f"pre-comparison input selection differs: {selection_path}"
            )
        input_selection_paths.append(selection_path)
        if selection.get("rule") != CALIBRATION_SELECTION_RULE:
            raise ValueError(f"calibration input rule differs: {path}")
        if selection.get("document_png_indices") != expected_document_indices:
            raise ValueError(f"unexpected document input indices: {path}")
        if selection.get("training_png_indices") != expected_training_indices:
            raise ValueError(f"unexpected training input indices: {path}")
        if selection.get("s1_image_ids") != record.get("s1_image_ids"):
            raise ValueError(
                f"calibration document IDs differ from the parity record: {path}"
            )
        if selection.get("training_image_ids") != record.get("s3_image_ids"):
            raise ValueError(
                f"calibration training IDs differ from the parity record: {path}"
            )
        if record.get("s2_image_ids") != selection.get("training_image_ids", [])[:2]:
            raise ValueError(
                f"calibration training IDs differ from the selected inputs: {path}"
            )
        s4_selections_by_repeat.append(
            _validate_s4_type_selections(selection, record, repeat, path)
        )

    s1_id_sets = [set(record["s1_image_ids"]) for _, record in records]
    training_id_sets = [
        set(record["calibration_selection"]["training_image_ids"])
        for _, record in records
    ]
    if any(len(ids) != 2 for ids in s1_id_sets) or any(
        first & second
        for index, first in enumerate(s1_id_sets)
        for second in s1_id_sets[index + 1 :]
    ):
        raise ValueError(
            "calibration document input pairs must contain distinct image IDs"
        )
    if any(len(ids) != 6 for ids in training_id_sets) or any(
        first & second
        for index, first in enumerate(training_id_sets)
        for second in training_id_sets[index + 1 :]
    ):
        raise ValueError("calibration training inputs must be distinct across repeats")
    s4_element_types = {
        json.dumps(record.get("s4_element_types"), sort_keys=True)
        for _, record in records
    }
    if len(s4_element_types) != 1:
        raise ValueError("per-type image checks do not cover the same element types")
    for element_type in s4_selections_by_repeat[0]:
        selections = [repeat[element_type] for repeat in s4_selections_by_repeat]
        unique_counts = {selection[1] for selection in selections}
        if len(unique_counts) != 1:
            raise ValueError(
                f"calibration per-type unique counts differ for {element_type}"
            )
        unique_count = selections[0][1]
        if unique_count >= 3 and len({selection[0] for selection in selections}) != 3:
            raise ValueError(
                f"calibration per-type inputs must be distinct for {element_type}"
            )

    digests = {record.get("encoder_state_sha256") for _, record in records}
    process_ids = {record.get("process_id") for _, record in records}
    source_commits = {record.get("source_commit") for _, record in records}
    versions = {record.get("tensorflow_version") for _, record in records}
    configurations = {
        json.dumps(record.get("configuration"), sort_keys=True) for _, record in records
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
    metric_repeats = [record["metrics"] for _, record in records]
    lower_bounds_record = json.loads(args.lower_bounds.read_text(encoding="utf-8"))
    if lower_bounds_record.get("source") != LOWER_BOUNDS_SOURCE:
        raise ValueError("calibration floors are not the originally registered floors")
    lower_bounds = lower_bounds_record.get("limits")
    if not isinstance(lower_bounds, dict):
        raise ValueError("registered calibration floors are missing")
    if not isinstance(lower_bounds_record.get("historical_calibration"), dict):
        raise ValueError("registered calibration history is missing")
    if not isinstance(lower_bounds_record.get("historical_failed_heldout"), list):
        raise ValueError("historical held-out failures are missing")
    limits = calibrate_limits(metric_repeats, lower_bounds)
    result = {
        "formula": "max(L, ceil2(1.5 * M)); M is the maximum over three distinct-input runs; ceil2 rounds up to two significant figures",
        "lower_bounds_source": lower_bounds_record["source"],
        "lower_bounds": lower_bounds,
        "historical_calibration": lower_bounds_record["historical_calibration"],
        "historical_failed_heldout": lower_bounds_record["historical_failed_heldout"],
        "encoder_state_sha256": next(iter(digests)),
        "source_commit": next(iter(source_commits)),
        "calibration_selection_rule": CALIBRATION_SELECTION_RULE,
        "calibration_input_ids": {
            str(repeat): {
                "s1_image_ids": record["s1_image_ids"],
                "training_image_ids": record["s3_image_ids"],
                "s4_image_ids": record["s4_image_ids"],
                "s4_type_selections": record["calibration_selection"][
                    "s4_type_selections"
                ],
            }
            for repeat, (_, record) in enumerate(records, start=1)
        },
        "repeat_process_ids": sorted(process_ids),
        "repeat_artifacts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path, _ in records
        },
        "input_selection_artifacts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_selection_paths
        },
        "repeat_metrics": metric_repeats,
        "limits": limits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _validate_s4_type_selections(
    selection: Mapping[str, JSONValue],
    record: Mapping[str, JSONValue],
    repeat: int,
    path: Path,
) -> dict[str, tuple[str, int]]:
    if selection.get("s4_element_type_occurrence") != repeat - 1:
        raise ValueError(f"unexpected per-type input occurrence: {path}")
    if selection.get("s4_image_ids") != record.get("s4_image_ids"):
        raise ValueError(
            f"calibration per-type IDs differ from the parity record: {path}"
        )
    type_selections = selection.get("s4_type_selections")
    if not isinstance(type_selections, list) or not all(
        isinstance(item, dict) for item in type_selections
    ):
        raise ValueError(f"calibration per-type selection details are missing: {path}")

    expected_types: list[str] = []
    expected_ids: list[str] = []
    per_type: dict[str, tuple[str, int]] = {}
    for item in type_selections:
        element_type = item.get("element_type")
        image_id = item.get("image_id")
        unique_index = item.get("unique_png_index")
        unique_count = item.get("unique_png_count")
        wrapped = item.get("wrapped")
        if (
            not isinstance(element_type, str)
            or not isinstance(image_id, str)
            or type(unique_index) is not int
            or type(unique_count) is not int
            or unique_count < 1
            or not isinstance(wrapped, bool)
        ):
            raise ValueError(f"invalid per-type selection details: {path}")
        if element_type in per_type:
            raise ValueError(f"duplicate per-type image selection: {path}")
        if unique_index != (repeat - 1) % unique_count:
            raise ValueError(f"unexpected wrapped index for {element_type}: {path}")
        if wrapped != ((repeat - 1) >= unique_count):
            raise ValueError(f"unexpected wrap flag for {element_type}: {path}")
        per_type[element_type] = (image_id, unique_count)
        expected_types.append(element_type)
        expected_ids.append(image_id)

    if expected_types != record.get("s4_element_types"):
        raise ValueError(
            f"calibration element types differ from the parity record: {path}"
        )
    if expected_ids != selection.get("s4_image_ids"):
        raise ValueError(f"calibration per-type IDs differ from the selection: {path}")
    if type_selections != record.get("s4_type_selections"):
        raise ValueError(
            f"calibration per-type details differ from the parity record: {path}"
        )
    return per_type


if __name__ == "__main__":
    main()
