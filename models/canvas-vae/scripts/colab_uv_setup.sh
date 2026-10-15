#!/usr/bin/env bash
# @file models/canvas-vae/scripts/colab_uv_setup.sh
# @brief Prepare a pinned CanvasVAE checkout and sync its requested uv dependencies on Colab.
# @option --source-url URL Git remote used to clone the source.
# @option --source-branch NAME Branch used to fetch the pinned commit.
# @option --source-commit SHA Full commit hash required for the checkout.
# @option --repo-dir PATH Remote checkout directory.
# @option --package NAME Workspace package to sync.
# @option --all-packages Sync every workspace package instead of one package.
# @option --extra NAME Repeatable uv extra to install.
# @option --group NAME Repeatable uv dependency group to install.
# @option --submodules Initialize recursive git submodules.
# @option --uv-environment PATH Required explicit sync environment; emit its shell export.
# @stdout A shell-quoted UV_PROJECT_ENVIRONMENT export.
# @example
#   eval "$(bash colab_uv_setup.sh --source-url URL --source-branch BRANCH --source-commit SHA --repo-dir /content/repo --package canvas-vae --extra training --extra vendor --submodules --uv-environment /content/vendor-env)"
#   eval "$(bash colab_uv_setup.sh --source-url URL --source-branch BRANCH --source-commit SHA --repo-dir /content/repo --all-packages --group docs --uv-environment /content/ci-env)"
set -Eeuo pipefail

function usage() {
    cat >&2 <<'USAGE'
usage: colab_uv_setup.sh --source-url URL --source-branch NAME --source-commit SHA --repo-dir PATH (--package NAME --extra NAME [--extra NAME]... | --all-packages) [--group NAME]... --uv-environment PATH [--submodules]
USAGE
    exit 2
}

source_url=''
source_branch=''
source_commit=''
repo_dir=''
package=''
all_packages=false
uv_environment=''
initialize_submodules=false
extras=()
groups=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source-url | --source-branch | --source-commit | --repo-dir | --package | --extra | --group | --uv-environment)
            [ "$#" -ge 2 ] || usage
            case "$1" in
                --source-url) source_url="$2" ;;
                --source-branch) source_branch="$2" ;;
                --source-commit) source_commit="$2" ;;
                --repo-dir) repo_dir="$2" ;;
                --package) package="$2" ;;
                --extra) extras+=("$2") ;;
                --group) groups+=("$2") ;;
                --uv-environment) uv_environment="$2" ;;
            esac
            shift 2
            ;;
        --submodules)
            initialize_submodules=true
            shift
            ;;
        --all-packages)
            all_packages=true
            shift
            ;;
        *) usage ;;
    esac
done

[ -n "$source_url" ] && [ -n "$source_branch" ] && [ -n "$repo_dir" ] && [ -n "$uv_environment" ] || usage
if [ "$all_packages" = true ]; then
    [ -z "$package" ] && [ "${#extras[@]}" -eq 0 ] || usage
else
    [ -n "$package" ] && [ "${#extras[@]}" -gt 0 ] || usage
fi
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$source_url" == https://* ]] || usage
[[ "$repo_dir" == /* ]] || usage
[[ "$uv_environment" == /* ]] || usage
case "/$uv_environment/" in
    */../* | */./*) usage ;;
esac
case "$uv_environment/" in
    "$repo_dir/.venv/"*)
    echo 'refusing repository .venv as the uv environment' >&2
    exit 78
        ;;
esac
for extra in "${extras[@]}"; do
    [[ "$extra" =~ ^[A-Za-z0-9_-]+$ ]] || usage
done
for group in "${groups[@]}"; do
    [[ "$group" =~ ^[A-Za-z0-9_-]+$ ]] || usage
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
    git -C "$repo_dir" submodule update --init --recursive >&2
fi

if [ -e "$repo_dir/.venv" ] || [ -L "$repo_dir/.venv" ]; then
    echo 'repository .venv exists; refusing environment drift' >&2
    exit 78
fi

unset PYTHONPATH MPLBACKEND UV_SYSTEM_PYTHON UV_PROJECT_ENVIRONMENT
export UV_PROJECT_ENVIRONMENT="$uv_environment"
uv_arguments=(sync --frozen --project "$repo_dir")
if [ "$all_packages" = true ]; then
    uv_arguments+=(--all-packages)
else
    uv_arguments+=(--package "$package")
    for extra in "${extras[@]}"; do
        uv_arguments+=(--extra "$extra")
    done
fi
for group in "${groups[@]}"; do
    uv_arguments+=(--group "$group")
done
uv "${uv_arguments[@]}" >&2
check_arguments=(run --frozen --no-sync --project "$repo_dir")
if [ "$all_packages" = true ]; then
    check_arguments+=(--all-packages)
else
    check_arguments+=(--package "$package")
    for extra in "${extras[@]}"; do
        check_arguments+=(--extra "$extra")
    done
fi
for group in "${groups[@]}"; do
    check_arguments+=(--group "$group")
done
if [ "$all_packages" = true ]; then
    uv "${check_arguments[@]}" python -c \
        'import os, sys; assert os.path.realpath(sys.prefix) == os.path.realpath(os.environ["UV_PROJECT_ENVIRONMENT"])' >&2
    printf 'colab_setup_passed commit=%s all_packages=true\n' "$head" >&2
else
    uv "${check_arguments[@]}" python \
        "$repo_dir/models/canvas-vae/scripts/check_colab_nccl.py" \
        --repo-dir "$repo_dir" \
        --expected-environment "$uv_environment" >&2
    printf 'colab_setup_passed commit=%s package=%s\n' "$head" "$package" >&2
fi
printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "$uv_environment"
