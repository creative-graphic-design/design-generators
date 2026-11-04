"""Compare the sparse full-run loss events without interpolating missing updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42004, 42005, 42006)
DATASETS = {"magazine": "magazine", "rico13": "rico", "publaynet": "publaynet"}
TAGS = {
    "generator": ("Train/Loss_G", "train_loss_g"),
    "discriminator": ("Train/Loss_D", "train_loss_d"),
}
OFFSETS = range(-2, 3)


class OffsetResult(TypedDict):
    pairs: int
    bitwise_equal: int


class SeriesComparison(TypedDict):
    vendor_events: int
    package_events: int
    same_step_pairs: OffsetResult
    event_index_offsets: dict[str, OffsetResult]
    step_offsets: dict[str, OffsetResult]


class TaggedSeriesComparison(SeriesComparison):
    vendor_tag: str
    package_tag: str


class EventFileRecord(TypedDict):
    path: str
    sha256: str


class RunRecord(TypedDict):
    dataset: str
    training_seed: int
    vendor_event_file: EventFileRecord
    package_event_file: EventFileRecord
    losses: dict[str, TaggedSeriesComparison]


class AggregateLoss(TypedDict):
    event_index_offsets: dict[str, OffsetResult]
    step_offsets: dict[str, OffsetResult]


class ComparisonReport(TypedDict):
    conclusion: str
    runs: list[RunRecord]
    aggregate: dict[str, AggregateLoss]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal_float32(left: float, right: float) -> bool:
    return struct.pack("<f", left) == struct.pack("<f", right)


def _offset_result(
    vendor: list[tuple[int, float]],
    package: list[tuple[int, float]],
    offset: int,
    *,
    by_event_index: bool,
) -> OffsetResult:
    if by_event_index:
        pairs = [
            (vendor[index + offset][1], value)
            for index, (_, value) in enumerate(package)
            if 0 <= index + offset < len(vendor)
        ]
    else:
        vendor_by_step = {step: value for step, value in vendor}
        pairs = [
            (vendor_by_step[step + offset], value)
            for step, value in package
            if step + offset in vendor_by_step
        ]
    return {
        "pairs": len(pairs),
        "bitwise_equal": sum(_equal_float32(left, right) for left, right in pairs),
    }


def compare_series(
    vendor: list[tuple[int, float]], package: list[tuple[int, float]]
) -> SeriesComparison:
    vendor_steps = [step for step, _ in vendor]
    package_steps = [step for step, _ in package]
    if len(set(vendor_steps)) != len(vendor_steps) or len(set(package_steps)) != len(
        package_steps
    ):
        raise ValueError("loss event steps must be unique within each arm")

    return {
        "vendor_events": len(vendor),
        "package_events": len(package),
        "same_step_pairs": _offset_result(vendor, package, 0, by_event_index=False),
        "event_index_offsets": {
            str(offset): _offset_result(vendor, package, offset, by_event_index=True)
            for offset in OFFSETS
        },
        "step_offsets": {
            str(offset): _offset_result(vendor, package, offset, by_event_index=False)
            for offset in OFFSETS
        },
    }


def _read_scalars(path: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def _event_file(directory: Path) -> Path:
    paths = sorted(directory.glob("events.out.tfevents.*"))
    if len(paths) != 1:
        raise ValueError(f"expected one event file in {directory}, found {len(paths)}")
    return paths[0]


def _sum_offset(
    runs: list[RunRecord],
    loss_name: str,
    field: Literal["event_index_offsets", "step_offsets"],
    offset: str,
) -> OffsetResult:
    return {
        "pairs": sum(run["losses"][loss_name][field][offset]["pairs"] for run in runs),
        "bitwise_equal": sum(
            run["losses"][loss_name][field][offset]["bitwise_equal"] for run in runs
        ),
    }


def build_report() -> ComparisonReport:
    runs: list[RunRecord] = []
    for dataset, vendor_dataset in DATASETS.items():
        for seed in SEEDS:
            vendor_dir = (
                ROOT
                / ".cache/layoutganpp/vendor-work/output"
                / vendor_dataset
                / "LayoutGAN++"
                / f"s5-{dataset}-original-seed-{seed}"
            )
            package_dir = (
                ROOT
                / ".cache/layoutganpp/package-rerun-295e2c6-20261026"
                / dataset
                / f"package-seed-{seed}"
                / "lightning_logs/version_0"
            )
            vendor_path = _event_file(vendor_dir)
            package_path = _event_file(package_dir)
            vendor_scalars = _read_scalars(vendor_path)
            package_scalars = _read_scalars(package_path)
            by_tag: dict[str, TaggedSeriesComparison] = {}
            for name, (vendor_tag, package_tag) in TAGS.items():
                if (
                    vendor_tag not in vendor_scalars
                    or package_tag not in package_scalars
                ):
                    raise ValueError(f"missing {name} scalar tag for {dataset}/{seed}")
                comparison = compare_series(
                    vendor_scalars[vendor_tag], package_scalars[package_tag]
                )
                by_tag[name] = {
                    **comparison,
                    "vendor_tag": vendor_tag,
                    "package_tag": package_tag,
                }

            runs.append(
                {
                    "dataset": dataset,
                    "training_seed": seed,
                    "vendor_event_file": {
                        "path": vendor_path.relative_to(ROOT).as_posix(),
                        "sha256": _sha256(vendor_path),
                    },
                    "package_event_file": {
                        "path": package_path.relative_to(ROOT).as_posix(),
                        "sha256": _sha256(package_path),
                    },
                    "losses": by_tag,
                }
            )

    aggregate: dict[str, AggregateLoss] = {}
    for loss_name in TAGS:
        aggregate[loss_name] = {
            "event_index_offsets": {
                str(offset): _sum_offset(
                    runs, loss_name, "event_index_offsets", str(offset)
                )
                for offset in OFFSETS
            },
            "step_offsets": {
                str(offset): _sum_offset(runs, loss_name, "step_offsets", str(offset))
                for offset in OFFSETS
            },
        }

    return {
        "conclusion": "sparse_event_logs_have_no_same_update_loss_pairs",
        "runs": runs,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/layoutganpp/s5/full-run-loss-event-comparison.json"),
    )
    args = parser.parse_args()
    report = build_report()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(_sha256(output))


if __name__ == "__main__":
    main()
