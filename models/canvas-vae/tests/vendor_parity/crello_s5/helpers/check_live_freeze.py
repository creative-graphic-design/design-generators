from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _freeze(path: Path) -> tuple[dict[str, str], list[str]]:
    pins: dict[str, str] = {}
    direct: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(.+)", line)
        if match:
            pins[_canonical(match.group(1))] = match.group(2)
        elif " @ " in line:
            name, url = line.split(" @ ", maxsplit=1)
            direct.append(f"{_canonical(name)} @ {url}")
    return pins, sorted(direct)


def _editable_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"#egg=([^&]+)", line)
        if match:
            names.add(_canonical(match.group(1)))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path, required=True)
    args = parser.parse_args()

    expected, expected_direct = _freeze(args.expected)
    actual, actual_direct = _freeze(args.actual)
    changed = {
        name: {"expected": version, "actual": actual.get(name)}
        for name, version in expected.items()
        if actual.get(name) != version
    }
    allowed_extra = _editable_names(args.expected) | {
        "canvas-vae",
        "laygen",
        "pip",
        "setuptools",
        "traingen",
        "traingen-parity",
        "wheel",
    }
    extra = sorted(set(actual) - set(expected) - allowed_extra)
    missing = sorted(set(expected) - set(actual))
    if changed or extra or missing or expected_direct != actual_direct:
        raise SystemExit(
            json.dumps(
                {
                    "changed": changed,
                    "extra_unlocked_packages": extra,
                    "missing_packages": missing,
                    "direct_url_records_match": expected_direct == actual_direct,
                },
                sort_keys=True,
            )
        )

    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.actual, args.copy_to)
    print(
        json.dumps(
            {
                "pinned_packages_matched": len(expected),
                "allowed_environment_tools": sorted(set(actual) - set(expected)),
                "direct_url_records_match": True,
                "full_freeze": str(args.copy_to),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
