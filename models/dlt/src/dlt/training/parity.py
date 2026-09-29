"""DLT trace-point names for shared S0-S2 parity helpers."""

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
