"""Restore the interpolation alias expected by the RADM reference checkout."""

from PIL import Image


if not hasattr(Image, "LINEAR"):
    setattr(Image, "LINEAR", Image.Resampling.BILINEAR)
