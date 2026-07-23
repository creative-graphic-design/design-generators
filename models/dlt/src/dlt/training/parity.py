"""DLT S0-S2 parity adapter structures."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lightning_module import DLTTrainingModule


@dataclass(frozen=True)
class DLTStepTrace:
    """Comparable DLT training-step tensors."""

    tensors: dict[str, torch.Tensor]


class DLTSyntheticStepTraceAdapter:
    """Trace adapter for local S0-S2 parity smoke checks."""

    trace_points = (
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

    def trace_training_step(
        self, module: DLTTrainingModule, batch: dict[str, torch.Tensor]
    ) -> DLTStepTrace:
        """Run and collect a DLT training-step trace."""
        module.training_step(batch, 0)
        return DLTStepTrace(
            {
                key: module.latest_step_trace[key]
                for key in self.trace_points
                if key in module.latest_step_trace
            }
        )
