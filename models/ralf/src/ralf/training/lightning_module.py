"""LightningModule implementing package-local RALF training."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping, Sequence
from enum import IntEnum
from types import SimpleNamespace
from typing import ClassVar, Protocol, cast

import torch
from jaxtyping import Float, Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from ..configuration_ralf import RalfConfig
from ..modeling_ralf import (
    RalfForConditionalLayoutGeneration,
    RalfRelationElement,
    RalfRelationLocation,
    RalfRelationSize,
    RalfRelationshipTable,
    _consume_relation_graph_rng,
)
from ..retrieval import RalfRetrievedBatch
from ..tokenization_ralf import RalfLayoutTokenizer
from .datamodule import RalfTrainingBatch


class _GradientTraceHook(Protocol):
    """Diagnostic callback surface used by the training parity runner."""

    def on_package_gradients_clipped(self, pl_module: "RalfTrainingModule") -> None:
        """Observe package gradients after Lightning applies clipping."""


class _RelationTableUnpickler(pickle.Unpickler):
    """Load only the three enum globals used by serialized relation tables."""

    _allowed_globals: ClassVar[dict[tuple[str, str], type[IntEnum]]] = {
        (
            "image2layout.train.helpers.relationships",
            "RelElement",
        ): RalfRelationElement,
        (
            "image2layout.train.helpers.relationships",
            "RelLoc",
        ): RalfRelationLocation,
        (
            "image2layout.train.helpers.relationships",
            "RelSize",
        ): RalfRelationSize,
    }

    def find_class(self, module: str, name: str) -> type[IntEnum]:
        try:
            return self._allowed_globals[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"unsupported global in relation table: {module}.{name}"
            ) from exc


_RELATION_TABLE_PICKLE_MODULE = SimpleNamespace(
    __name__="ralf_relation_table_pickle",
    Unpickler=_RelationTableUnpickler,
    load=pickle.load,
)


def _load_relationship_table(
    path: str | os.PathLike[str],
) -> RalfRelationshipTable:
    """Load a relation table without importing its original enum classes."""
    payload = torch.load(
        path,
        map_location="cpu",
        pickle_module=_RELATION_TABLE_PICKLE_MODULE,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"relationship table at {path} is not a mapping")

    return cast(RalfRelationshipTable, payload)


class RalfTrainingModule(LightningModule):
    """Own the package model and expose its training loop to Lightning."""

    def __init__(
        self,
        *,
        config: RalfConfig,
        model: RalfForConditionalLayoutGeneration | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        clip_max_norm: float = 0.1,
        epochs: int = 70,
        scheduler: str = "multi_step",
        scheduler_milestones: tuple[float, ...] = (0.7,),
        condition_type: str = "unconditional",
        relationship_table_path: str | None = None,
    ) -> None:
        """Initialize the package-local Lightning training module."""
        super().__init__()
        self.ralf_config = config
        relationship_table = None
        if condition_type == "relation":
            table_path = relationship_table_path or os.environ.get(
                "RALF_RELATIONSHIP_TABLE_PATH"
            )
            existing_table = None if model is None else model.relationship_table
            if table_path is None and existing_table is not None:
                relationship_table = existing_table
            elif table_path is None:
                raise ValueError("relation training requires a relationship table path")

            else:
                relationship_table = _load_relationship_table(table_path)

        self.relationship_table = relationship_table
        self.model = model or RalfForConditionalLayoutGeneration(
            config, relationship_table=relationship_table
        )
        if (
            model is not None
            and relationship_table is not None
            and model.relationship_table is not relationship_table
        ):
            self.model.configure_relationship_table(relationship_table)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_max_norm = clip_max_norm
        self.epochs = epochs
        self.scheduler = scheduler
        self.scheduler_milestones = tuple(scheduler_milestones)
        self.condition_type = condition_type
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, ...]] = {}
        self._gradient_trace_hook: _GradientTraceHook | None = None

    def forward(
        self,
        **batch: Shaped[torch.Tensor, ...] | RalfRetrievedBatch | Sequence[str | int],
    ) -> CausalLMOutput:
        """Run the package model on one training batch."""
        retrieved = cast(RalfRetrievedBatch | None, batch.pop("retrieved", None))
        model_batch = cast(RalfTrainingBatch, batch)
        if self.condition_type == "refinement":
            model_batch, condition_kwargs = self._prepare_refinement_layout(model_batch)
        else:
            condition_kwargs = self._condition_kwargs(model_batch)

        return self.model(
            input_ids=model_batch["input_ids"],
            labels=model_batch["labels"],
            attention_mask=model_batch["attention_mask"],
            pixel_values=model_batch["pixel_values"],
            saliency=model_batch["saliency"],
            retrieved=retrieved,
            condition_type=self.condition_type,
            **condition_kwargs,
        )

    def _condition_kwargs(
        self, batch: RalfTrainingBatch
    ) -> dict[str, Shaped[torch.Tensor, ...]]:
        """Build the full layout condition before decoder shifting."""
        if self.condition_type == "refinement":
            _, condition_kwargs = self._prepare_refinement_layout(batch)
            return condition_kwargs
        elif self.condition_type in {"label", "label_size", "completion", "relation"}:
            encoded = RalfLayoutTokenizer(self.ralf_config).encode_layout(
                labels=batch["layout_labels"],
                bbox=batch["layout_bbox"],
                mask=batch["layout_mask"],
            )
            if self.condition_type == "relation":
                if self.relationship_table is None:
                    raise RuntimeError("relation training has no relationship table")

                _consume_relation_graph_rng(batch["layout_mask"])
        else:
            return {}

        condition_kwargs: dict[str, Shaped[torch.Tensor, ...]] = {
            "constraint_input_ids": cast(
                Shaped[torch.Tensor, "batch tokens"], encoded["input_ids"]
            ),
            "constraint_mask": cast(
                Shaped[torch.Tensor, "batch tokens"], encoded["attention_mask"]
            ),
        }
        if self.condition_type == "relation":
            condition_kwargs["sample_ids"] = cast(
                Shaped[torch.Tensor, "batch"], batch.get("sample_ids", [])
            )

        return condition_kwargs

    def _prepare_refinement_layout(
        self, batch: RalfTrainingBatch
    ) -> tuple[RalfTrainingBatch, dict[str, Shaped[torch.Tensor, "..."]]]:
        """Prepare the perturbed layout used for refinement training."""
        bbox = batch["layout_bbox"].clone()
        for bbox_index in range(bbox.size(-1)):
            component = bbox[..., bbox_index]
            noise = torch.normal(
                0.0,
                0.01,
                size=component.shape,
                dtype=component.dtype,
            ).to(component.device)
            perturbed = (component + noise).clamp(0.0, 1.0)
            perturbed[~batch["layout_mask"]] = 0.0
            bbox[..., bbox_index] = perturbed

        encoded = RalfLayoutTokenizer(self.ralf_config).encode_layout(
            labels=batch["layout_labels"],
            bbox=bbox,
            mask=batch["layout_mask"],
        )
        prepared_batch = cast(RalfTrainingBatch, dict(batch))
        return prepared_batch, {
            "constraint_input_ids": cast(
                Shaped[torch.Tensor, "batch tokens"], encoded["input_ids"]
            ),
            "constraint_mask": cast(
                Shaped[torch.Tensor, "batch tokens"], encoded["attention_mask"]
            ),
        }

    def training_step(
        self,
        batch: RalfTrainingBatch,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Run one teacher-forced package training step."""
        del batch_idx
        model_batch = self._model_batch(batch)
        if self.condition_type == "refinement":
            model_batch, condition_kwargs = self._prepare_refinement_layout(model_batch)
        else:
            condition_kwargs = self._condition_kwargs(model_batch)

        output = self.model(
            input_ids=model_batch["input_ids"],
            labels=model_batch["labels"],
            attention_mask=model_batch["attention_mask"],
            pixel_values=model_batch["pixel_values"],
            saliency=model_batch["saliency"],
            retrieved=model_batch["retrieved"],
            condition_type=self.condition_type,
            **condition_kwargs,
        )
        if output.loss is None:
            raise RuntimeError("RALF package model returned no training loss")

        loss = output.loss
        self.latest_step_trace = {
            "train_loss": loss.detach(),
            "logits": output.logits.detach(),
        }
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: RalfTrainingBatch,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Evaluate one validation batch with the same teacher forcing."""
        del batch_idx
        model_batch = self._model_batch(batch)
        if self.condition_type == "refinement":
            model_batch, condition_kwargs = self._prepare_refinement_layout(model_batch)
        else:
            condition_kwargs = self._condition_kwargs(model_batch)

        output = self.model(
            input_ids=model_batch["input_ids"],
            labels=model_batch["labels"],
            attention_mask=model_batch["attention_mask"],
            pixel_values=model_batch["pixel_values"],
            saliency=model_batch["saliency"],
            retrieved=model_batch["retrieved"],
            condition_type=self.condition_type,
            **condition_kwargs,
        )
        if output.loss is None:
            raise RuntimeError("RALF package model returned no validation loss")

        self.log("val_loss", output.loss, on_epoch=True, prog_bar=True)
        return output.loss

    def optim_groups(
        self,
    ) -> list[dict[str, list[Shaped[torch.Tensor, ...]] | float]]:
        """Return decay and learning-rate groups matching the reference loop."""
        decay: set[str] = set()
        no_decay: set[str] = set()

        whitelist = (
            nn.Linear,
            nn.MultiheadAttention,
            nn.Conv2d,
            nn.Conv1d,
            nn.Parameter,
        )
        blacklist = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)

        for module_name, module in self.model.named_modules():
            for param_name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue

                name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(name)
                elif param_name.endswith("weight") and isinstance(module, whitelist):
                    decay.add(name)
                elif param_name.endswith("weight") and isinstance(module, blacklist):
                    no_decay.add(name)
                elif "weight_ih" in param_name or "weight_hh" in param_name:
                    decay.add(name)
                elif "bias_ih" in param_name or "bias_hh" in param_name:
                    no_decay.add(name)

        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if missing := set(parameters) - decay - no_decay:
            raise RuntimeError(f"unassigned trainable parameters: {sorted(missing)}")

        if overlap := decay & no_decay:
            raise RuntimeError(
                f"parameters assigned to both optimizer groups: {sorted(overlap)}"
            )

        custom_prefix = "encoder.extractor.body"
        groups: list[dict[str, list[Shaped[torch.Tensor, ...]] | float]] = []
        ordered_groups = (
            (decay, self.weight_decay, True),
            (no_decay, 0.0, True),
            (decay, self.weight_decay, False),
            (no_decay, 0.0, False),
        )
        for names, group_decay, custom_group in ordered_groups:
            selected = {
                name for name in names if name.startswith(custom_prefix) is custom_group
            }
            if not selected:
                continue

            group_lr = self.learning_rate * 0.1 if custom_group else self.learning_rate
            groups.append(
                {
                    "params": [parameters[name] for name in sorted(selected)],
                    "weight_decay": group_decay,
                    "lr": group_lr,
                }
            )

        return groups

    def scheduler_epochs(self) -> int:
        """Return the epoch count used to construct this run's scheduler."""
        trainer = self._trainer
        if trainer is None:
            return self.epochs

        if trainer.max_epochs is None:
            raise RuntimeError("Lightning trainer max_epochs must be set")

        return trainer.max_epochs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build AdamW and the epoch-level MultiStepLR."""
        optimizer = torch.optim.AdamW(
            self.optim_groups(), lr=self.learning_rate, foreach=False
        )
        if self.scheduler == "none":
            return optimizer

        epochs = self.scheduler_epochs()
        milestones = [int(value * epochs) for value in self.scheduler_milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Apply the fixed norm clip used by the training configuration.

        The original implementation clips only when the configured norm is
        positive, so a non-positive value disables clipping instead of
        multiplying every gradient by zero.
        """
        del optimizer, gradient_clip_val, gradient_clip_algorithm
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)

        trace_hook = getattr(self, "_gradient_trace_hook", None)
        if trace_hook is not None:
            trace_hook.on_package_gradients_clipped(self)

    @staticmethod
    def _model_batch(
        batch: RalfTrainingBatch,
    ) -> RalfTrainingBatch:
        batch_retrieved = batch["retrieved"]
        return {
            "input_ids": batch["input_ids"].long(),
            "labels": batch["labels"].long(),
            "attention_mask": batch["attention_mask"].bool(),
            "pixel_values": batch["pixel_values"].float(),
            "saliency": batch["saliency"].float(),
            "layout_labels": batch["layout_labels"].long(),
            "layout_bbox": batch["layout_bbox"].float(),
            "layout_mask": batch["layout_mask"].bool(),
            "sample_ids": batch.get("sample_ids", []),
            "retrieved": batch_retrieved,
        }
