"""Configuration values shared by LayoutGAN++ training checks."""

from enum import StrEnum, auto


class LayoutGANPPSeedMode(StrEnum):
    """Seed policy used by the package training data module."""

    default = auto()
    deterministic = auto()
