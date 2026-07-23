#!/usr/bin/env bash

# @file scripts/list_test_members.sh
# @brief Print workspace members that have tests as compact JSON.
# @stdout A JSON array of objects with `dir` and `package` fields.
# @example
#   scripts/list_test_members.sh

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python - "${repo_root}" <<'PY'
import json
import pathlib
import sys
import tomllib

repo_root = pathlib.Path(sys.argv[1])
members: list[dict[str, str]] = []

for parent in ("lib", "models"):
    parent_dir = repo_root / parent
    if not parent_dir.is_dir():
        continue

    for pyproject_file in sorted(parent_dir.glob("*/pyproject.toml")):
        member_dir = pyproject_file.parent
        if not (member_dir / "tests").is_dir():
            continue

        with pyproject_file.open("rb") as file:
            project = tomllib.load(file)["project"]

        members.append(
            {
                "dir": member_dir.relative_to(repo_root).as_posix(),
                "package": project["name"],
            }
        )

print(json.dumps(members, separators=(",", ":")))
PY
