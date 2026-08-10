"""DLT S0-S2 parity adapter structures."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import DLTExample
from .dataset import DLTStepTrace as DLTStepTraceTensors
from .lightning_module import DLTTrainingModule


@dataclass(frozen=True)
class DLTStepTrace:
    """Comparable DLT training-step tensors."""

    tensors: DLTStepTraceTensors


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
        self, module: DLTTrainingModule, batch: DLTExample
    ) -> DLTStepTrace:
        """Run and collect a DLT training-step trace."""
        module.training_step(batch, 0)
        trace = module.latest_step_trace
        return DLTStepTrace(
            {
                "box": trace["box"],
                "box_cond": trace["box_cond"],
                "cat": trace["cat"],
                "mask_box": trace["mask_box"],
                "mask_cat": trace["mask_cat"],
                "noise": trace["noise"],
                "t": trace["t"],
                "noised_box": trace["noised_box"],
                "noised_cat": trace["noised_cat"],
                "pred_box": trace["pred_box"],
                "pred_cat": trace["pred_cat"],
                "masked_l2": trace["masked_l2"],
                "masked_ce": trace["masked_ce"],
                "loss": trace["loss"],
            }
        )
