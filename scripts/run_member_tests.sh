#!/usr/bin/env bash

# @file scripts/run_member_tests.sh
# @brief Run pytest for one uv workspace member.
# @description
#   The default mode matches CI and enforces member coverage. Set
#   `MEMBER_TEST_COVERAGE=0` to keep local pre-commit runs coverage-free.
#   Set `MEMBER_TEST_COVERAGE_XML_DIR` to write pytest-cov XML for CI Codecov
#   upload.
# @arg $1 member_dir Workspace member directory such as `models/layout-dm`.
# @stdout Pytest output for the selected workspace member.
# @example
#   scripts/run_member_tests.sh models/layout-dm

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/run_member_tests.sh <member_dir>" >&2
  exit 2
fi

member_dir="${1%/}"
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${repo_root}"

pyproject_file="${member_dir}/pyproject.toml"
tests_dir="${member_dir}/tests"

if [ ! -f "${pyproject_file}" ]; then
  echo "workspace member pyproject not found: ${pyproject_file}" >&2
  exit 2
fi

if [ ! -d "${tests_dir}" ]; then
  echo "workspace member has no tests: ${member_dir}"
  exit 0
fi

package_name="$(
  PYPROJECT_FILE="${pyproject_file}" python - <<'PY'
import os
import tomllib

with open(os.environ["PYPROJECT_FILE"], "rb") as file:
    print(tomllib.load(file)["project"]["name"])
PY
)"

import_package="$(
  MEMBER_DIR="${member_dir}" python - <<'PY'
import os
import pathlib

src = pathlib.Path(os.environ["MEMBER_DIR"]) / "src"
packages = sorted(
    path.name
    for path in src.iterdir()
    if path.is_dir() and (path / "__init__.py").exists()
)
print(packages[0])
PY
)"

optional_extras="$(
  PYPROJECT_FILE="${pyproject_file}" python - <<'PY'
import os
import tomllib

with open(os.environ["PYPROJECT_FILE"], "rb") as file:
    data = tomllib.load(file)

print(" ".join(sorted(data.get("project", {}).get("optional-dependencies", {}))))
PY
)"

uv_run_args=(--package "${package_name}")
if [ "${MEMBER_TEST_RUNNER_DEPS:-1}" = "0" ]; then
  uv_run_args+=(--no-sync)
fi
if [[ " ${optional_extras} " == *" agents "* ]]; then
  uv_run_args+=(--extra agents)
fi
if [[ " ${optional_extras} " == *" torch "* ]]; then
  uv_run_args+=(--extra torch)
fi
if [[ " ${optional_extras} " == *" diffusion "* ]]; then
  uv_run_args+=(--extra diffusion)
fi

if [ "${MEMBER_TEST_RUNNER_DEPS:-1}" != "0" ]; then
  uv_run_args+=(--with "beartype>=0.22.9,<0.23")
  if [ "${package_name}" = "posgen" ]; then
    uv_run_args+=(--with-editable "${repo_root}/lib/laygen")
  fi
fi

pytest_args=(
  "${tests_dir}"
  -m
  "not vendor_parity and not integration"
)

if [ "${MEMBER_TEST_COVERAGE:-1}" != "0" ]; then
  if [ "${MEMBER_TEST_RUNNER_DEPS:-1}" != "0" ]; then
    uv_run_args+=(--with "pytest-cov>=7.1.0")
  fi

  fail_under="$(
    PYPROJECT_FILE="${pyproject_file}" \
    DEFAULT_FAIL_UNDER="${WORKSPACE_COVERAGE_FAIL_UNDER:-90}" \
      python - <<'PY'
import os
import sys
import tomllib

path = os.environ["PYPROJECT_FILE"]
floor = float(os.environ["DEFAULT_FAIL_UNDER"])
with open(path, "rb") as file:
    data = tomllib.load(file)

value = float(data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under", floor))
if value < floor:
    sys.exit(
        "coverage fail_under for {} is {:g}, below required floor {:g}".format(
            path,
            value,
            floor,
        )
    )

print("{:g}".format(value))
PY
  )"

  pytest_args+=(
    "--cov=${import_package}"
    "--cov-report=term-missing"
    "--cov-fail-under=${fail_under}"
  )

  if [ -n "${MEMBER_TEST_COVERAGE_XML_DIR:-}" ]; then
    mkdir -p "${MEMBER_TEST_COVERAGE_XML_DIR}"
    pytest_args+=(
      "--cov-report=xml:${MEMBER_TEST_COVERAGE_XML_DIR}/${package_name}.xml"
    )
  fi
fi

uv run "${uv_run_args[@]}" pytest "${pytest_args[@]}"
