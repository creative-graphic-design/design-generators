"""Label helpers reserved for future position-generation datasets."""


def normalize_label(label: str) -> str:
    """Normalize a position-generation label key.

    Args:
        label: Raw label string.

    Returns:
        Lowercase label key with hyphens converted to underscores.

    Raises:
        ValueError: This function does not raise.

    Examples:
        >>> normalize_label("Anchor-Point")
        'anchor_point'
    """

    return label.strip().lower().replace("-", "_")
