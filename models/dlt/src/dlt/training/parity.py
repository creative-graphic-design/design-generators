"""Trace-point names DLT records in a training step for package-versus-reference parity checks."""

from __future__ import annotations

TRACE_POINTS: tuple[str, ...] = (
    "box",
    "box_cond",
    "cat",
    "mask_box",
    "mask_cat",
    "noise",
    "t",
    "noised_box",
    "noised_cat",
    "pred_box",
    "pred_cat",
    "masked_l2",
    "masked_ce",
    "loss",
)
