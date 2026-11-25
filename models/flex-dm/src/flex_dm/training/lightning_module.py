"""Lightning training module for Flex-DM masked document modeling."""

from __future__ import annotations

from collections.abc import Mapping

from jaxtyping import Float, Int, Shaped
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from traingen.lightning.steps import finish_training_step, log_validation_loss
from traingen.optim import KerasAdam, clip_gradients_by_norm

from ..modeling_flex_dm import FlexDmForMaskedDocumentModeling
from .masking import prepare_training_batch, sample_training_tasks


class FlexDmTrainingModule(LightningModule):
    """Train a converted Flex-DM checkpoint with the original Keras update."""

    def __init__(self, model_name_or_path: str) -> None:
        """Load the converted starting checkpoint."""
        super().__init__()
        self.model = FlexDmForMaskedDocumentModeling.from_pretrained(model_name_or_path)
        self.latest_step_trace: dict[str, Shaped[Tensor, "..."]] = {}

    def prepare_batch(
        self, batch: Mapping[str, Shaped[Tensor, "..."]]
    ) -> tuple[
        dict[str, Shaped[Tensor, "..."]],
        dict[str, Shaped[Tensor, "..."]],
        dict[str, Shaped[Tensor, "..."]],
        Int[Tensor, "batch"],
    ]:
        """Sample a task and prepare its inputs, labels, and masks."""
        task_ids = sample_training_tasks(
            self.model.config.task_names,
            self.model.config.masking_method,
            batch_size=batch["length"].shape[0],
        ).to(batch["length"].device)
        targets, modified_inputs, masks = prepare_training_batch(
            batch,
            self.model.config.input_columns,
            task_ids,
            attribute_groups=self.model.config.attribute_groups,
            input_dtype=self.model.config.input_dtype,
        )
        return modified_inputs, targets, masks, task_ids

    def training_step(
        self, batch: dict[str, Shaped[Tensor, "..."]], batch_idx: int
    ) -> Float[Tensor, ""]:
        """Compute one source-matched masked training objective."""
        del batch_idx
        inputs, labels, masks, task_ids = self.prepare_batch(batch)
        output = self.model(
            inputs=inputs,
            labels=labels,
            masks=masks,
            task_ids=task_ids,
        )
        if (
            output.loss is None
            or output.loss_components is None
            or output.l2_loss is None
        ):
            raise RuntimeError("Flex-DM training requires task losses and L2 loss")

        losses = {**output.loss_components, "l2": output.l2_loss}
        trace = {"task_ids": task_ids.detach()}
        total, self.latest_step_trace = finish_training_step(self, losses, trace)
        return total

    def validation_step(
        self, batch: dict[str, Shaped[Tensor, "..."]], batch_idx: int
    ) -> Float[Tensor, ""]:
        """Evaluate the same masked objective without updating weights."""
        del batch_idx
        inputs, labels, masks, task_ids = self.prepare_batch(batch)
        output = self.model(
            inputs=inputs,
            labels=labels,
            masks=masks,
            task_ids=task_ids,
        )
        if output.loss is None:
            raise RuntimeError("Flex-DM validation requires a loss")

        log_validation_loss(self, output.loss)
        return output.loss

    def configure_optimizers(self) -> Optimizer:
        """Return Keras 2 Adam at the released Flex-DM learning rate."""
        return KerasAdam(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Apply Keras per-parameter clipping before the Adam update."""
        del optimizer, gradient_clip_algorithm
        if gradient_clip_val not in (None, 0.0):
            raise ValueError("Flex-DM uses per-parameter clip_norm=1.0")

        # Match the reference's independent Keras clipnorm for each parameter.
        clip_gradients_by_norm(self.model.parameters(), max_norm=1.0)
