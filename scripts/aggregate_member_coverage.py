"""Aggregate workspace member coverage artifacts into Shields endpoint JSON."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MemberCoverage:
    """Coverage percentage for one workspace member."""

    package: str
    percent: float


def _read_member_coverage(path: Path) -> MemberCoverage:
    data = json.loads(path.read_text(encoding="utf-8"))
    percent = float(data["totals"]["percent_covered"])
    return MemberCoverage(package=path.stem, percent=percent)


def _badge_color(percent: float) -> str:
    if percent >= 90:
        return "brightgreen"
    if percent >= 80:
        return "green"
    if percent >= 70:
        return "yellow"
    if percent >= 60:
        return "orange"
    return "red"


def aggregate(input_dir: Path, output_dir: Path) -> list[MemberCoverage]:
    """Write coverage badge endpoint files and return member coverage rows."""

    coverage_files = sorted(input_dir.glob("*.json"))
    if not coverage_files:
        raise FileNotFoundError(f"no coverage JSON files found under {input_dir}")

    rows = sorted(
        (_read_member_coverage(path) for path in coverage_files),
        key=lambda row: row.package,
    )
    minimum = min(rows, key=lambda row: row.percent)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"min {minimum.percent:.1f}% per member",
        "color": _badge_color(round(minimum.percent, 1)),
    }
    (output_dir / "coverage-badge.json").write_text(
        json.dumps(endpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "minimum": {
            "package": minimum.package,
            "percent": round(minimum.percent, 4),
        },
        "members": [
            {"package": row.package, "percent": round(row.percent, 4)} for row in rows
        ],
    }
    (output_dir / "coverage-members.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = aggregate(args.input_dir, args.output_dir)
    minimum = min(rows, key=lambda row: row.percent)
    print(
        f"Minimum member coverage: {minimum.percent:.1f}% "
        f"({minimum.package}) across {len(rows)} members."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
