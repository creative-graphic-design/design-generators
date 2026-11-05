"""Fix the root README models badge from the workspace model members."""

from __future__ import annotations

from .card import model_member_count
from .constants import REPO_ROOT
from .root_readme import update_root_model_badge_count


def main() -> int:
    """Update the models badge and fail when the README needed a change."""
    readme_path = REPO_ROOT / "README.md"
    model_count = model_member_count()
    if not update_root_model_badge_count(readme_path, model_count):
        return 0

    print(f"{readme_path}: models badge updated to {model_count}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
