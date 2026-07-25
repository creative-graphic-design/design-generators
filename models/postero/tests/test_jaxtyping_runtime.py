"""Runtime import smoke for jaxtyping-annotated PosterO modules."""

from jaxtyping import install_import_hook


def test_jaxtyping_import_hook_accepts_postero() -> None:
    with install_import_hook("postero", "beartype.beartype"):
        from postero.agent import PosterOAgent

    assert PosterOAgent.__name__ == "PosterOAgent"
