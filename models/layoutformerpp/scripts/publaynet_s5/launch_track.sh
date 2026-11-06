#!/usr/bin/env bash

# @file models/layoutformerpp/scripts/publaynet_s5/launch_track.sh
# @brief Validate and start one PubLayNet full-run track.
# @arg $1 campaign path relative to the repository root.
# @arg $2 family: unconditional, completion, refinement, label, label_size, or relation.
# @arg $3 training seed.
# @arg $4 arm: original or package.
# @arg $5 physical GPU index.
# @example
#   bash models/layoutformerpp/scripts/publaynet_s5/launch_track.sh "$CAMPAIGN_REL" label 0 package "$GPU_INDEX"

set -euo pipefail

if [[ $# -ne 5 ]]; then
    printf 'usage: launch_track.sh CAMPAIGN_REL FAMILY SEED ARM GPU_INDEX\n' >&2
    exit 2
fi

package_env="${PACKAGE_ENV:?PACKAGE_ENV is required}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$package_env/bin/python" "$script_dir/launch_track.py" "$@"
