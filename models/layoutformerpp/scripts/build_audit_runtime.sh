#!/usr/bin/env bash

# @file models/layoutformerpp/scripts/build_audit_runtime.sh
# @brief Build the dedicated LayoutFormer++ parity runtime.
# @description
#   Installs the committed runtime requirements into a fresh repository-local
#   environment and verifies its normalized freeze against the committed value.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --package layoutformerpp "$script_dir/build_audit_runtime.py" "$@"
