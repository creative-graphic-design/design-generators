#!/usr/bin/env bash
# @file models/canvas-vae/scripts/colab_uv_setup.sh
# @brief Prepare a pinned CanvasVAE checkout and sync its requested uv extras on Colab.
# @option --source-url URL Git remote used to clone the source.
# @option --source-branch NAME Branch used to fetch the pinned commit.
# @option --source-commit SHA Full commit hash required for the checkout.
# @option --repo-dir PATH Remote checkout directory.
# @option --package NAME Workspace package to sync.
# @option --extra NAME Repeatable uv extra to install.
# @option --submodules Initialize recursive git submodules.
# @option --uv-environment PATH Set UV_PROJECT_ENVIRONMENT for the sync.
# @example
#   bash colab_uv_setup.sh --source-url URL --source-branch BRANCH --source-commit SHA --repo-dir /content/repo --package canvas-vae --extra training --extra vendor --submodules
set -Eeuo pipefail

function usage() {
    cat >&2 <<'USAGE'
usage: colab_uv_setup.sh --source-url URL --source-branch NAME --source-commit SHA --repo-dir PATH --package NAME --extra NAME [--extra NAME]... [--submodules] [--uv-environment PATH]
USAGE
    exit 2
}

source_url=''
source_branch=''
source_commit=''
repo_dir=''
package=''
uv_environment=''
initialize_submodules=false
extras=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source-url | --source-branch | --source-commit | --repo-dir | --package | --extra | --uv-environment)
            [ "$#" -ge 2 ] || usage
            case "$1" in
                --source-url) source_url="$2" ;;
                --source-branch) source_branch="$2" ;;
                --source-commit) source_commit="$2" ;;
                --repo-dir) repo_dir="$2" ;;
                --package) package="$2" ;;
                --extra) extras+=("$2") ;;
                --uv-environment) uv_environment="$2" ;;
            esac
            shift 2
            ;;
        --submodules)
            initialize_submodules=true
            shift
            ;;
        *) usage ;;
    esac
done

[ -n "$source_url" ] && [ -n "$source_branch" ] && [ -n "$repo_dir" ] && [ -n "$package" ] && [ "${#extras[@]}" -gt 0 ] || usage
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$source_url" == https://* ]] || usage
for extra in "${extras[@]}"; do
    [[ "$extra" =~ ^[A-Za-z0-9_-]+$ ]] || usage
done
git check-ref-format --branch "$source_branch" >/dev/null

if [ -e "$repo_dir" ]; then
    head="$(git -C "$repo_dir" rev-parse HEAD)"
    [ "$head" = "$source_commit" ] || { echo 'refusing to reuse an unpinned repository directory' >&2; exit 78; }
    [ -z "$(git -C "$repo_dir" status --porcelain)" ] || { echo 'refusing to reuse a dirty repository directory' >&2; exit 78; }
else
    mkdir -p "$(dirname "$repo_dir")"
    git clone --quiet --no-checkout --branch "$source_branch" "$source_url" "$repo_dir"
    git -C "$repo_dir" fetch origin "$source_commit"
    git -C "$repo_dir" checkout --quiet --detach "$source_commit"
fi

head="$(git -C "$repo_dir" rev-parse HEAD)"
[ "$head" = "$source_commit" ] || { echo 'source checkout differs from the requested commit' >&2; exit 78; }
[ -z "$(git -C "$repo_dir" status --porcelain)" ] || { echo 'source checkout is dirty' >&2; exit 78; }
if [ "$initialize_submodules" = true ]; then
    git -C "$repo_dir" submodule update --init --recursive
fi

unset PYTHONPATH MPLBACKEND UV_SYSTEM_PYTHON UV_PROJECT_ENVIRONMENT
if [ -n "$uv_environment" ]; then
    export UV_PROJECT_ENVIRONMENT="$uv_environment"
fi
uv_arguments=(sync --frozen --project "$repo_dir" --package "$package")
for extra in "${extras[@]}"; do
    uv_arguments+=(--extra "$extra")
done
uv "${uv_arguments[@]}"
printf 'colab_setup_passed commit=%s package=%s\n' "$head" "$package"
