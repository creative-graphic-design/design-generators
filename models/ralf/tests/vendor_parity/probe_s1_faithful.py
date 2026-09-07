"""Run the faithful, fixed-batch CGL label S1 lockstep diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
PARITY_ROOT = REPO_ROOT / "models" / "ralf" / "tests" / "vendor_parity"
sys.path.insert(0, str(PARITY_ROOT))

import run_training_stages as stages  # noqa: E402
from run_training_stages import (  # noqa: E402
    _load_context,
    _loss_pair,
    _models,
    _move_batch,
)
from training_reference import (  # noqa: E402
    VendorTrainingModel,
    vendor_preprocess,
)
from ralf.configuration_ralf import RalfConfigTaskName  # noqa: E402
from ralf.modeling_ralf import (  # noqa: E402
    RalfForConditionalLayoutGeneration,
    RalfRelationshipTable,
)
from ralf.retrieval import RalfRetrievedBatch  # noqa: E402


class ProbeDivergence(RuntimeError):
    """Raised when the diagnostic finds its first non-identical tensor."""


def _digest(value: Shaped[Tensor, "..."]) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_hashes() -> dict[str, str]:
    return {
        "torch_cpu": _digest(torch.get_rng_state()),
        "torch_cuda": _digest(torch.cuda.get_rng_state()),
    }


def _location(delta: Tensor) -> tuple[int, ...]:
    flat_index = delta.reshape(-1).argmax()
    return tuple(int(index) for index in torch.unravel_index(flat_index, delta.shape))


def _compare(
    name: str,
    package: Shaped[Tensor, "..."],
    vendor: Shaped[Tensor, "..."],
) -> None:
    if package.shape != vendor.shape or package.dtype != vendor.dtype:
        raise ProbeDivergence(
            f"{name}: shape/dtype package={tuple(package.shape)}/{package.dtype} "
            f"vendor={tuple(vendor.shape)}/{vendor.dtype}"
        )
    if torch.equal(package, vendor):
        print(f"MATCH {name}: shape={tuple(package.shape)} dtype={package.dtype}")
        return

    delta = (package.detach().float() - vendor.detach().float()).abs()
    index = _location(delta)
    raise ProbeDivergence(
        f"{name}: first_index={index}; package_value={package[index].item()}; "
        f"vendor_value={vendor[index].item()}; max_abs_diff={delta.max().item()}; "
        f"shape={tuple(package.shape)}; dtype={package.dtype}"
    )


def _randperm_source() -> str:
    for frame in reversed(traceback.extract_stack(limit=12)):
        if "task_preprocessor.py" in frame.filename:
            return "vendor.task_preprocessor"
        if "modeling_ralf.py" in frame.filename:
            return "package.modeling_ralf"
    return "unknown"


class _Recorder:
    def __init__(self) -> None:
        self.phase = "setup"
        self.checkpoints: list[dict[str, object]] = []
        self.seed_events: list[dict[str, object]] = []
        self.randperm: list[dict[str, object]] = []
        self.vendor_conditions: dict[str, dict[str, Shaped[Tensor, "..."]]] = {}
        self.package_condition: dict[str, Shaped[Tensor, "..."]] | None = None

    def checkpoint(self, name: str) -> None:
        self.checkpoints.append({"name": name, "phase": self.phase, **_rng_hashes()})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the faithful S1 probe")
    context_args = argparse.Namespace(
        dataset="cgl",
        condition="label",
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    recorder = _Recorder()
    original_reseed = stages.reseed
    original_randperm = torch.randperm

    def seeded(seed: int) -> None:
        recorder.seed_events.append(
            {"phase": recorder.phase, "seed": seed, "when": "before", **_rng_hashes()}
        )
        original_reseed(seed)
        recorder.seed_events.append(
            {"phase": recorder.phase, "seed": seed, "when": "after", **_rng_hashes()}
        )

    def randperm(
        n: int | torch.SymInt,
        *,
        generator: torch.Generator | None = None,
        out: Tensor | None = None,
        dtype: torch.dtype | None = None,
        layout: torch.layout | None = None,
        device: torch.device | str | int | None = None,
        pin_memory: bool = False,
        requires_grad: bool = False,
    ) -> Tensor:
        before = _rng_hashes()
        output = original_randperm(
            n,
            generator=generator,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        after = _rng_hashes()
        recorder.randperm.append(
            {
                "phase": recorder.phase,
                "n": int(n),
                "source": _randperm_source(),
                "cpu_changed": before["torch_cpu"] != after["torch_cpu"],
                "cuda_changed": before["torch_cuda"] != after["torch_cuda"],
            }
        )
        return output

    try:
        recorder.phase = "load_context"
        config, _data, context = _load_context(context_args)
        device = torch.device(args.device)
        recorder.phase = "models"
        package_module, vendor_model, _ = _models(
            config, args.cache_dir, device, args.seed, "label"
        )
        package_model = package_module.model
        batch = context["batch"]

        vendor_training_model = cast(VendorTrainingModel, vendor_model)
        original_vendor_preprocess = vendor_training_model.preprocess
        preprocess_count = 0

        def vendor_preprocess_hook(
            self: VendorTrainingModel, batch_value: Mapping[str, object]
        ) -> tuple[dict[str, object], dict[str, object]]:
            del self
            nonlocal preprocess_count
            preprocess_count += 1
            name = f"vendor_preprocess_{preprocess_count}"
            recorder.phase = name
            recorder.checkpoint(f"before_{name}")
            result = original_vendor_preprocess(batch_value)
            inputs, _ = result
            recorder.vendor_conditions[name] = {
                "seq_layout_const": cast(
                    Shaped[Tensor, "..."], inputs["seq_layout_const"]
                )
                .detach()
                .cpu()
                .clone(),
                "seq_layout_const_pad_mask": cast(
                    Shaped[Tensor, "..."], inputs["seq_layout_const_pad_mask"]
                )
                .detach()
                .cpu()
                .clone(),
            }
            recorder.checkpoint(f"after_{name}")
            return result

        original_prepare = package_model._prepare_conditional_inputs

        def package_prepare(
            self: RalfForConditionalLayoutGeneration,
            *,
            pixel_values: Float[Tensor, "batch channels height width"] | None,
            saliency: Float[Tensor, "batch 1 height width"] | None,
            retrieved: RalfRetrievedBatch | None,
            batch_size: int,
            condition_type: RalfConfigTaskName | None = None,
            constraint_input_ids: Int[Tensor, "batch tokens"] | None = None,
            constraint_mask: Bool[Tensor, "batch tokens"] | None = None,
            constraint_element_mask: Bool[Tensor, "batch elements"] | None = None,
            relationship_table: RalfRelationshipTable | None = None,
            sample_ids: Int[Tensor, "batch"]
            | Sequence[int | str]
            | int
            | str
            | None = None,
        ) -> dict[str, Shaped[Tensor, ...] | Mapping[str, Shaped[Tensor, ...]]]:
            del self
            recorder.phase = "package_condition"
            recorder.checkpoint("before_package_condition")
            result = original_prepare(
                pixel_values=pixel_values,
                saliency=saliency,
                retrieved=retrieved,
                batch_size=batch_size,
                condition_type=condition_type,
                constraint_input_ids=constraint_input_ids,
                constraint_mask=constraint_mask,
                constraint_element_mask=constraint_element_mask,
                relationship_table=relationship_table,
                sample_ids=sample_ids,
            )
            result_tensors = cast(Mapping[str, Shaped[Tensor, ...]], result)
            recorder.package_condition = {
                "seq_layout_const": result_tensors["seq_layout_const"]
                .detach()
                .cpu()
                .clone(),
                "seq_layout_const_pad_mask": result_tensors["seq_layout_const_pad_mask"]
                .detach()
                .cpu()
                .clone(),
            }
            recorder.checkpoint("after_package_condition")
            return result

        original_package_forward = package_model.forward

        def package_forward(
            self: RalfForConditionalLayoutGeneration,
            input_ids: Int[Tensor, "batch tokens"] | None = None,
            pixel_values: Float[Tensor, "batch channels height width"] | None = None,
            saliency: Float[Tensor, "batch 1 height width"] | None = None,
            attention_mask: Bool[Tensor, "batch tokens"] | None = None,
            labels: Int[Tensor, "batch tokens"] | None = None,
            retrieved: RalfRetrievedBatch | None = None,
            condition_type: RalfConfigTaskName | None = None,
            constraint_input_ids: Int[Tensor, "batch tokens"] | None = None,
            constraint_mask: Bool[Tensor, "batch tokens"] | None = None,
            constraint_element_mask: Bool[Tensor, "batch elements"] | None = None,
            return_dict: bool | None = None,
            **kwargs: str | float | bool | None,
        ) -> CausalLMOutput | tuple[Float[Tensor, ...], ...]:
            del self
            recorder.phase = "package_forward"
            recorder.checkpoint("before_package_forward")
            result = original_package_forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                saliency=saliency,
                attention_mask=attention_mask,
                labels=labels,
                retrieved=retrieved,
                condition_type=condition_type,
                constraint_input_ids=constraint_input_ids,
                constraint_mask=constraint_mask,
                constraint_element_mask=constraint_element_mask,
                return_dict=return_dict,
                **kwargs,
            )
            recorder.checkpoint("after_package_forward")
            return result

        original_vendor_train_loss = vendor_training_model.train_loss

        def vendor_train_loss(
            self: VendorTrainingModel,
            inputs: Mapping[str, object],
            targets: Mapping[str, object],
        ) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
            del self
            recorder.phase = "vendor_forward"
            recorder.checkpoint("before_vendor_forward")
            result = original_vendor_train_loss(inputs, targets)
            recorder.checkpoint("after_vendor_forward")
            return result

        with (
            patch.object(stages, "reseed", seeded),
            patch.object(torch, "randperm", randperm),
            patch.object(
                vendor_model,
                "preprocess",
                MethodType(vendor_preprocess_hook, vendor_training_model),
            ),
            patch.object(
                package_model,
                "_prepare_conditional_inputs",
                MethodType(package_prepare, package_model),
            ),
            patch.object(
                package_model,
                "forward",
                MethodType(package_forward, package_model),
            ),
            patch.object(
                vendor_model,
                "train_loss",
                MethodType(vendor_train_loss, vendor_training_model),
            ),
        ):
            recorder.phase = "s1_initial_vendor_preprocess"
            initial_vendor_inputs, initial_vendor_targets = vendor_preprocess(
                vendor_model, batch
            )
            package_batch = _move_batch(batch, device)
            _compare(
                "s1_prepared_input_ids",
                package_batch["input_ids"].cpu(),
                cast(Shaped[Tensor, "..."], initial_vendor_inputs["seq"]),
            )
            _compare(
                "s1_prepared_labels",
                package_batch["labels"].cpu(),
                cast(Shaped[Tensor, "..."], initial_vendor_targets["seq"]),
            )

            recorder.phase = "s1_loss_pair"
            package_loss, vendor_loss, package_logits, vendor_logits = _loss_pair(
                package_module, vendor_model, batch, device, args.seed
            )
            if recorder.package_condition is None:
                raise RuntimeError("package condition was not captured")
            _compare(
                "condition.seq_layout_const",
                recorder.package_condition["seq_layout_const"],
                recorder.vendor_conditions["vendor_preprocess_2"]["seq_layout_const"],
            )
            _compare(
                "condition.seq_layout_const_pad_mask",
                recorder.package_condition["seq_layout_const_pad_mask"],
                recorder.vendor_conditions["vendor_preprocess_2"][
                    "seq_layout_const_pad_mask"
                ],
            )
            _compare(
                "logits", package_logits.detach().cpu(), vendor_logits.detach().cpu()
            )
            _compare("loss", package_loss.detach().cpu(), vendor_loss.detach().cpu())
    except ProbeDivergence as exc:
        print(f"FIRST_DIVERGENCE {exc}")
        return 1
    finally:
        pass

    print(f"RALF_FILE {__import__('ralf').__file__}")
    print("SEED_EVENTS")
    for event in recorder.seed_events:
        print(event)
    print("CHECKPOINTS")
    for checkpoint in recorder.checkpoints:
        print(checkpoint)
    print("RANDPERM_ACCOUNTING")
    for index, draw in enumerate(recorder.randperm):
        print(f"draw_index={index} {draw}")
    for name, condition in recorder.vendor_conditions.items():
        print(
            f"VENDOR_CONDITION name={name} "
            f"seq_sha256={_digest(condition['seq_layout_const'])} "
            f"pad_mask_sha256={_digest(condition['seq_layout_const_pad_mask'])}"
        )
    print("S1_FAITHFUL_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
