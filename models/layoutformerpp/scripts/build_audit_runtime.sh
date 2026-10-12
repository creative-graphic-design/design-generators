#!/usr/bin/env bash

# @file models/layoutformerpp/scripts/build_audit_runtime.sh
# @brief Build the dedicated LayoutFormer++ parity runtime.
# @description
#   Installs the workspace packages and pinned reference dependencies into the
#   repository-local audited runtime used by the staged parity campaign.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --package layoutformerpp "$script_dir/build_audit_runtime.py" "$@"
