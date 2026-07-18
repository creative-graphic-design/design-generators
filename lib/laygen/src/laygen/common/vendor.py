"""Vendor repository path helpers shared by parity scripts."""

from __future__ import annotations

from pathlib import Path


def vendor_root(
    repo: str,
    *,
    marker: str | Path | None = None,
    path: str | Path | None = None,
    repo_root: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve an initialized vendor submodule checkout.

    Args:
        repo: Repository directory name under `vendor/`.
        marker: Optional file that must exist inside the vendor checkout.
        path: Optional user-supplied path. Defaults to `vendor/<repo>`.
        repo_root: Optional repository root override for tests.
        cwd: Optional current working directory override for tests.

    Returns:
        Resolved path to the vendor checkout.

    Raises:
        FileNotFoundError: If the checkout or marker cannot be found.

    Examples:
        >>> vendor_root("const-layout")  # doctest: +SKIP
        PosixPath('...')
    """
    repo_dir = Path("vendor") / repo
    requested = Path(path) if path is not None else repo_dir
    project_root = repo_root or Path(__file__).resolve().parents[4]
    current = cwd or Path.cwd()
    marker_path = Path(marker) if marker is not None else None

    candidates = _candidate_roots(requested, repo_dir, project_root, current)
    seen: set[Path] = set()
    initialized_missing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if marker_path is None:
            if resolved.exists():
                return resolved
        elif (resolved / marker_path).exists():
            return resolved
        elif resolved.exists():
            initialized_missing.append(resolved)

    searched = "\n".join(f"- {path}" for path in seen)
    hint = f"Run `git submodule update --init vendor/{repo}` from the repository root."
    if initialized_missing:
        missing = "\n".join(f"- {path}" for path in initialized_missing)
        raise FileNotFoundError(
            f"Found vendor/{repo}, but required marker `{marker_path}` is missing. "
            f"{hint}\nChecked initialized paths:\n{missing}"
        )
    raise FileNotFoundError(
        f"Could not find initialized vendor/{repo}. {hint}\nSearched:\n{searched}"
    )


def _candidate_roots(
    requested: Path, repo_dir: Path, repo_root: Path, cwd: Path
) -> list[Path]:
    candidates = [requested] if requested.is_absolute() else []
    candidates.extend(
        [
            cwd / requested,
            repo_root / requested,
            repo_root / repo_dir,
        ]
    )
    if "=" in repo_root.name:
        sibling = repo_root.with_name(repo_root.name.split("=", maxsplit=1)[0])
        candidates.extend([sibling / requested, sibling / repo_dir])
    if "=" in cwd.name:
        sibling = cwd.with_name(cwd.name.split("=", maxsplit=1)[0])
        candidates.extend([sibling / requested, sibling / repo_dir])
    return candidates
