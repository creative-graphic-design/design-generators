"""Label helpers reserved for future position-generation datasets."""


def normalize_label(label: str) -> str:
    """Normalize a position-generation label key."""
    return label.strip().lower().replace("-", "_")
