from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from laygen.common import DatasetName

from layoutformerpp import LayoutFormerPPConfig, LayoutFormerPPForConditionalGeneration
from layoutformerpp.labels import (
    RICO25_LABEL_TRANSLATION,
    label_translation_for_dataset,
)
from layoutformerpp.training import TRAINING_RECIPES
from layoutformerpp.serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenR,
    T5LayoutSequenceForGenT,
)
from layoutformerpp.training.scheduler import LayoutFormerPPWarmupLR
from layoutformerpp.training.lightning_module import (
    LayoutFormerPPTrainingConfig,
    LayoutFormerPPTrainingModule,
    vendor_effective_cross_entropy,
)
from layoutformerpp.training.parity import compare_static_state, state_dict_sha256
from traingen_parity import capture_rng_state, restore_rng_state
from traingen_parity.compare import TensorTolerance, compare_tensors
from layoutformerpp_training_reference import (
    package_tokenizer,
    reference_checkpoint_behavior,
    reference_classes,
    reference_optimizer_scheduler_state,
    reference_label_map,
    reference_serialization,
    reference_tokens,
    script_arguments,
    source_facts,
)

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ORIGINAL_SOURCE_REVISION = "1498ff300710b4fc204aece537582d37ca447fc7"


@pytest.mark.parametrize(
    "recipe", tuple(TRAINING_RECIPES.values()), ids=lambda item: item.name
)
def test_s0_recipe_source_manifest_matches(recipe) -> None:
    source_status = subprocess.run(
        ["git", "submodule", "status", "--", "vendor/ms-layout-generation"],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert source_status.split()[0].lstrip("+-") == ORIGINAL_SOURCE_REVISION
    args = script_arguments(recipe)
    assert args["dataset"] == (
        "rico" if recipe.dataset is DatasetName.rico25 else "publaynet"
    )
    assert tuple(str(args["tasks"]).split(",")) == tuple(map(str, recipe.tasks))
    assert int(str(args["epoch"])) == recipe.epochs
    assert int(str(args["batch_size"])) == recipe.batch_size
    assert int(str(args["eval_batch_size"])) == recipe.eval_batch_size
    assert int(str(args["num_pos_embed"])) == recipe.max_position_embeddings
    assert int(str(args["decode_max_length"])) == recipe.decode_max_length
    assert int(str(args["warmup_num_steps"])) == recipe.warmup_num_steps
    assert int(str(args["eval_seed"])) == recipe.eval_seed
    assert int(str(args["eval_interval"])) == recipe.eval_interval
    assert float(str(args["lr"])) == recipe.learning_rate
    assert set(recipe.serialization_flags) <= set(args)
    assert int(str(args["gradient_accumulation"])) == 1
    assert "enable_clip_gradient" not in args
    if recipe.partition_buckets:
        assert (
            tuple(
                map(int, str(args["partition_training_data_task_buckets"]).split(","))
            )
            == recipe.partition_buckets
        )


@pytest.mark.parametrize(
    "recipe", tuple(TRAINING_RECIPES.values()), ids=lambda item: item.name
)
def test_s0_model_topology_state_vocab_and_forward_match(recipe) -> None:
    reference_model_class, reference_tokenizer_class = reference_classes()
    tokens = reference_tokens(recipe)
    package_tokenizer_value = package_tokenizer(recipe)
    reference_tokenizer = reference_tokenizer_class(tokens)
    translation = label_translation_for_dataset(recipe.dataset)
    assert reference_label_map(recipe) == dict(translation.sequence_id2label)
    assert len(package_tokenizer_value) == recipe.vocab_size
    assert package_tokenizer_value.get_vocab() == reference_tokenizer._token2id
    assert json.dumps(package_tokenizer_value.get_vocab()) == json.dumps(
        reference_tokenizer._token2id
    )
    reference_vocab_bytes = json.dumps(reference_tokenizer._token2id).encode("utf-8")
    package_as_reference_bytes = json.dumps(package_tokenizer_value.get_vocab()).encode(
        "utf-8"
    )
    assert package_as_reference_bytes == reference_vocab_bytes

    torch.manual_seed(0)
    reference_model = reference_model_class(
        vocab_size=recipe.vocab_size,
        max_len=recipe.max_position_embeddings,
        bos_token_id=0,
        pad_token_id=2,
        eos_token_id=1,
        d_model=recipe.d_model,
        nhead=recipe.attention_heads,
        num_layers=recipe.num_layers,
        dropout=recipe.dropout,
        d_feedforward=recipe.d_model * 4,
        share_embedding=True,
    )
    config = LayoutFormerPPConfig(
        dataset=recipe.dataset,
        task=recipe.condition,
        vocab_size=recipe.vocab_size,
        max_position_embeddings=recipe.max_position_embeddings,
        d_model=recipe.d_model,
        encoder_layers=recipe.num_layers,
        decoder_layers=recipe.num_layers,
        encoder_attention_heads=recipe.attention_heads,
        decoder_attention_heads=recipe.attention_heads,
        dim_feedforward=recipe.d_model * 4,
        dropout=recipe.dropout,
        share_embedding=True,
    )
    package_model = LayoutFormerPPForConditionalGeneration(config)
    missing, unexpected = package_model.load_state_dict(
        reference_model.state_dict(), strict=True
    )
    assert missing == []
    assert unexpected == []
    comparison = compare_static_state(reference_model, package_model)
    assert comparison.passed
    assert (
        state_dict_sha256(reference_model.state_dict())
        == comparison.reference_state_sha256
    )
    assert (
        reference_model.enc_embedding.weight.data_ptr()
        == reference_model.dec_embedding.weight.data_ptr()
    )
    assert (
        package_model.enc_embedding.weight.data_ptr()
        == package_model.dec_embedding.weight.data_ptr()
    )
    assert (
        package_model.out.weight.data_ptr()
        == package_model.dec_embedding.weight.data_ptr()
    )

    reference_model.eval()
    package_model.eval()
    input_ids = torch.tensor([[5, 6, 1]])
    labels = torch.tensor([[6, 7, 1]])
    valid = input_ids.ne(2)
    with torch.no_grad():
        reference_outputs = reference_model(input_ids, ~valid, labels)
        package_outputs = package_model(
            input_ids=input_ids, attention_mask=valid, labels=labels
        )
    torch.testing.assert_close(package_outputs.logits, reference_outputs["logits"])
    torch.testing.assert_close(
        vendor_effective_cross_entropy(package_outputs.logits, labels),
        reference_outputs["loss"],
    )

    reference_model.train()
    package_model.train()
    rng_state = torch.random.get_rng_state()
    reference_train_outputs = reference_model(input_ids, ~valid, labels)
    torch.random.set_rng_state(rng_state)
    package_train_outputs = package_model(
        input_ids=input_ids,
        attention_mask=valid,
        labels=labels,
    )
    torch.testing.assert_close(
        package_train_outputs.logits,
        reference_train_outputs["logits"],
    )
    torch.testing.assert_close(
        vendor_effective_cross_entropy(package_train_outputs.logits, labels),
        reference_train_outputs["loss"],
    )


def _s1_rows(
    recipe,
) -> tuple[
    tuple[str, list[int], list[list[int]], list[tuple[int, int, int, int, int]]], ...
]:
    if recipe.name == "publaynet_relation":
        tasks = tuple(map(str, recipe.tasks))
        rows = (
            ([1, 2], [[3, 4, 5, 6], [7, 8, 9, 10]]),
            ([2], [[11, 12, 13, 14]]),
            ([3, 4], [[15, 16, 17, 18], [19, 20, 21, 22]]),
            ([4], [[23, 24, 25, 26]]),
            ([1, 3], [[27, 28, 29, 30], [31, 32, 33, 34]]),
            ([2], [[35, 36, 37, 38]]),
        )
        return tuple(
            (
                task,
                labels,
                bboxes,
                [(labels[-1], 1, labels[0], 1, 3)] if len(labels) > 1 else [],
            )
            for task, (labels, bboxes) in zip(tasks, rows, strict=True)
        )
    task = str(recipe.tasks[0])
    rows = (
        ([1, 2], [[3, 4, 5, 6], [7, 8, 9, 10]]),
        ([3], [[11, 12, 13, 14]]),
    )
    return tuple((task, labels, bboxes, [(2, 1, 1, 1, 3)]) for labels, bboxes in rows)


def _s1_fixture(
    recipe, reference_tokenizer_class
) -> tuple[dict[str, torch.Tensor | None], dict[str, object]]:
    rows = _s1_rows(recipe)
    reference_records = [
        reference_serialization(
            recipe,
            task,
            labels=labels,
            bboxes=bboxes,
            relations=relations,
        )
        for task, labels, bboxes, relations in rows
    ]
    input_values = tuple(record["input_bytes"] for record in reference_records)
    output_values = tuple(record["output_bytes"] for record in reference_records)
    if not all(isinstance(value, bytes) for value in input_values):
        raise AssertionError("original input serialization must be bytes")
    if not all(isinstance(value, bytes) for value in output_values):
        raise AssertionError("original output serialization must be bytes")
    input_bytes = cast(tuple[bytes, ...], input_values)
    output_bytes = cast(tuple[bytes, ...], output_values)
    input_texts = [value.decode("utf-8") for value in input_bytes]
    output_texts = [value.decode("utf-8") for value in output_bytes]
    reference_tokenizer = reference_tokenizer_class(reference_tokens(recipe))
    package_tokenizer_value = package_tokenizer(recipe)
    reference_inputs = reference_tokenizer(input_texts, add_eos=True)
    reference_targets = reference_tokenizer(output_texts, add_eos=True)
    package_inputs = package_tokenizer_value.encode_text(input_texts, add_eos=True)
    package_targets = package_tokenizer_value.encode_text(output_texts, add_eos=True)
    assert torch.equal(reference_inputs["input_ids"], package_inputs["input_ids"])
    assert torch.equal(reference_inputs["mask"], package_inputs["attention_mask"])
    assert torch.equal(reference_targets["input_ids"], package_targets["input_ids"])
    assert torch.equal(reference_targets["mask"], package_targets["attention_mask"])
    assert all(
        int(ids[mask][-1]) == reference_tokenizer.eos_token_id
        for ids, mask in zip(
            reference_inputs["input_ids"], reference_inputs["mask"], strict=True
        )
    )
    assert all(
        int(ids[mask][-1]) == reference_tokenizer.eos_token_id
        for ids, mask in zip(
            reference_targets["input_ids"], reference_targets["mask"], strict=True
        )
    )
    task_ids = None
    if recipe.name == "publaynet_relation":
        task_ids = torch.tensor(recipe.task_ids, dtype=torch.long)
    batch: dict[str, torch.Tensor | None] = {
        "input_ids": reference_inputs["input_ids"],
        "attention_mask": reference_inputs["mask"],
        "labels": reference_targets["input_ids"],
        "task_ids": task_ids,
    }
    assert bool(batch["labels"].eq(2).any())
    return batch, {
        "source_bytes": tuple(zip(input_bytes, output_bytes, strict=True)),
        "reference_input_ids": reference_inputs["input_ids"],
        "reference_attention_mask": reference_inputs["mask"],
        "reference_labels": reference_targets["input_ids"],
        "reference_target_mask": reference_targets["mask"],
    }


def _s1_package_config(recipe) -> LayoutFormerPPTrainingConfig:
    return {
        "dataset": str(recipe.dataset),
        "task": str(recipe.condition),
        "vocab_size": recipe.vocab_size,
        "max_position_embeddings": recipe.max_position_embeddings,
        "d_model": recipe.d_model,
        "encoder_layers": recipe.num_layers,
        "decoder_layers": recipe.num_layers,
        "encoder_attention_heads": recipe.attention_heads,
        "decoder_attention_heads": recipe.attention_heads,
        "dim_feedforward": recipe.d_model * 4,
        "dropout": recipe.dropout,
        "share_embedding": True,
    }


def _reference_pre_optimizer_trace(
    model,
    batch: Mapping[str, torch.Tensor | None],
) -> dict[str, object]:
    input_ids_value = batch["input_ids"]
    attention_mask_value = batch["attention_mask"]
    labels_value = batch["labels"]
    if not isinstance(input_ids_value, torch.Tensor):
        raise AssertionError("S1 reference fixture requires tensor input ids")
    if not isinstance(attention_mask_value, torch.Tensor):
        raise AssertionError("S1 reference fixture requires an attention mask")
    if not isinstance(labels_value, torch.Tensor):
        raise AssertionError("S1 reference fixture requires tensor labels")
    input_ids = input_ids_value.long()
    attention_mask = attention_mask_value.bool()
    labels = labels_value.long()
    task_ids = batch["task_ids"]
    if task_ids is not None and not isinstance(task_ids, torch.Tensor):
        raise AssertionError("S1 task ids must be a tensor or None")
    enc_hs, enc_padding_mask = model.encode(input_ids, ~attention_mask, task_ids)
    decoder_input_ids = torch.cat(
        [labels.new_ones((labels.size(0), 1)) * model.bos_token_id, labels[:, :-1]],
        dim=1,
    )
    dec_input = model.dec_embedding(decoder_input_ids).permute(1, 0, 2)
    dec_input = model.dec_pos_embedding(dec_input)
    tgt_mask = (
        torch.triu(
            torch.ones((dec_input.size(0), dec_input.size(0)), device=dec_input.device)
        )
        == 1
    ).transpose(0, 1)
    tgt_mask = (
        tgt_mask.float()
        .masked_fill(tgt_mask == 0, float("-inf"))
        .masked_fill(tgt_mask == 1, 0.0)
    )
    decoder_hidden_state = model.decoder(
        tgt=dec_input,
        memory=enc_hs,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=enc_padding_mask,
    )
    logits = model.out(decoder_hidden_state.permute(1, 0, 2))
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    pad_mask = labels.eq(model.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pad_mask": pad_mask,
        "decoder_input_ids": decoder_input_ids,
        "task_ids": task_ids,
        "encoder_memory": enc_hs,
        "decoder_hidden_state": decoder_hidden_state,
        "logits": logits,
        "per_token_loss": per_token_loss,
        "pad_only_ce_contribution": per_token_loss.masked_select(pad_mask).sum()
        / labels.numel(),
        "loss": per_token_loss.mean(),
    }


S1_TOLERANCE = TensorTolerance(rtol=1e-4, atol=1e-5)
S1_SURFACE_ORDER = (
    "source_bytes",
    "input_ids",
    "attention_mask",
    "labels",
    "pad_mask",
    "decoder_input_ids",
    "task_ids",
    "encoder_memory",
    "decoder_hidden_state",
    "logits",
    "per_token_loss",
    "pad_only_ce_contribution",
    "loss",
)
S1_FLOAT_SURFACES = frozenset(
    {
        "encoder_memory",
        "decoder_hidden_state",
        "logits",
        "per_token_loss",
        "pad_only_ce_contribution",
        "loss",
    }
)


def _rng_bytes(state) -> tuple[bytes, tuple[bytes, ...]]:
    return (
        state.torch_cpu.cpu().numpy().tobytes(),
        tuple(value.cpu().numpy().tobytes() for value in state.torch_cuda),
    )


def _rng_equal(left, right) -> bool:
    left_cpu, left_cuda = _rng_bytes(left)
    right_cpu, right_cuda = _rng_bytes(right)
    return left_cpu == right_cpu and left_cuda == right_cuda


def _rng_sha256(state) -> str:
    digest = hashlib.sha256()
    cpu, cuda = _rng_bytes(state)
    digest.update(cpu)
    for value in cuda:
        digest.update(value)
    return digest.hexdigest()


def _surface_difference(
    field: str,
    actual: object,
    expected: object,
) -> dict[str, object] | None:
    tolerance = S1_TOLERANCE if field in S1_FLOAT_SURFACES else TensorTolerance()
    if actual is None or expected is None:
        if actual is expected:
            return None
        return {
            "field": field,
            "index": None,
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "tolerance": {"rtol": tolerance.rtol, "atol": tolerance.atol},
        }
    if field == "source_bytes":
        actual_pairs = cast(tuple[tuple[bytes, bytes], ...], actual)
        expected_pairs = cast(tuple[tuple[bytes, bytes], ...], expected)
        actual_bytes = b"".join(item for pair in actual_pairs for item in pair)
        expected_bytes = b"".join(item for pair in expected_pairs for item in pair)
        if actual_bytes == expected_bytes:
            return None
        limit = min(len(actual_bytes), len(expected_bytes))
        first = next(
            (
                index
                for index in range(limit)
                if actual_bytes[index] != expected_bytes[index]
            ),
            limit,
        )
        return {
            "field": field,
            "index": first,
            "max_abs": 1.0,
            "max_rel": 1.0,
            "tolerance": {"rtol": 0.0, "atol": 0.0},
        }
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        if actual == expected:
            return None
        return {
            "field": field,
            "index": None,
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "tolerance": {"rtol": tolerance.rtol, "atol": tolerance.atol},
        }
    comparison = compare_tensors(field, actual, expected, tolerance)
    if comparison.passed:
        return None
    if actual.shape != expected.shape:
        index = None
    elif actual.is_floating_point() or expected.is_floating_point():
        close = torch.isclose(
            actual,
            expected,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            equal_nan=True,
        )
        mismatches = torch.nonzero(~close.reshape(-1), as_tuple=False)
        index = int(mismatches[0].item()) if mismatches.numel() else None
    else:
        mismatches = torch.nonzero(~actual.eq(expected).reshape(-1), as_tuple=False)
        index = int(mismatches[0].item()) if mismatches.numel() else None
    return {
        "field": field,
        "index": index,
        "max_abs": comparison.max_abs_diff,
        "max_rel": comparison.max_rel_diff,
        "tolerance": {"rtol": tolerance.rtol, "atol": tolerance.atol},
    }


def _first_divergence(
    recipe_name: str,
    reference: Mapping[str, object],
    package: Mapping[str, object],
) -> dict[str, object] | None:
    for field in S1_SURFACE_ORDER:
        difference = _surface_difference(field, package[field], reference[field])
        if difference is not None:
            return {"family": recipe_name, **difference}
    return None


def _surface_error_summary(
    reference: Mapping[str, object], package: Mapping[str, object]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in S1_SURFACE_ORDER:
        actual = package[field]
        expected = reference[field]
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            difference = (actual.detach().float() - expected.detach().float()).abs()
            denominator = (
                expected.detach()
                .float()
                .abs()
                .clamp_min(torch.finfo(torch.float32).eps)
            )
            relative = difference / denominator
            summary[field] = {
                "max_abs": float(difference.max().item())
                if difference.numel()
                else 0.0,
                "max_rel": float(relative.max().item()) if relative.numel() else 0.0,
            }
        else:
            summary[field] = {"max_abs": 0.0, "max_rel": 0.0}
    return summary


def _s1_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("S1 requires CUDA; CPU is diagnostic-only")
    if torch.cuda.device_count() != 1:
        pytest.fail("S1 requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (7, 0):
        pytest.fail("S1 acceptance is pinned to the selected SM 7.0 device")
    return device


def _run_s1_case(recipe, device: torch.device) -> dict[str, object]:
    reference_model_class, reference_tokenizer_class = reference_classes()
    batch, fixture = _s1_fixture(recipe, reference_tokenizer_class)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    torch.manual_seed(0)
    reference_model = reference_model_class(
        vocab_size=recipe.vocab_size,
        max_len=recipe.max_position_embeddings,
        bos_token_id=0,
        pad_token_id=2,
        eos_token_id=1,
        d_model=recipe.d_model,
        nhead=recipe.attention_heads,
        num_layers=recipe.num_layers,
        dropout=recipe.dropout,
        d_feedforward=recipe.d_model * 4,
        share_embedding=True,
    )
    module = LayoutFormerPPTrainingModule(
        recipe_name=recipe.name,
        config=_s1_package_config(recipe),
    )
    module.model.load_state_dict(reference_model.state_dict(), strict=True)
    reference_model = reference_model.to(device)
    module = module.to(device)
    reference_model.train()
    module.train()
    reference_initial_hash = state_dict_sha256(reference_model.state_dict())
    package_initial_hash = state_dict_sha256(module.model.state_dict())
    caller_rng = capture_rng_state()
    try:
        boundary_rng = capture_rng_state()
        if len(boundary_rng.torch_cuda) != 1:
            raise AssertionError("S1 must expose exactly one selected CUDA RNG state")
        restore_rng_state(boundary_rng)
        reference_before_hash = state_dict_sha256(reference_model.state_dict())
        reference_outputs = _reference_pre_optimizer_trace(reference_model, batch)
        reference_after_rng = capture_rng_state()
        reference_after_hash = state_dict_sha256(reference_model.state_dict())
        restore_rng_state(boundary_rng)
        package_before_hash = state_dict_sha256(module.model.state_dict())
        trace = module.pre_optimizer_trace(batch)
        package_after_rng = capture_rng_state()
        package_after_hash = state_dict_sha256(module.model.state_dict())
    finally:
        restore_rng_state(caller_rng)
    caller_restored_rng = capture_rng_state()

    reference_values: dict[str, object] = {
        "source_bytes": fixture["source_bytes"],
        "input_ids": reference_outputs["input_ids"],
        "attention_mask": reference_outputs["attention_mask"],
        "labels": reference_outputs["labels"],
        "pad_mask": reference_outputs["pad_mask"],
        "decoder_input_ids": reference_outputs["decoder_input_ids"],
        "task_ids": reference_outputs["task_ids"],
        "encoder_memory": reference_outputs["encoder_memory"],
        "decoder_hidden_state": reference_outputs["decoder_hidden_state"],
        "logits": reference_outputs["logits"],
        "per_token_loss": reference_outputs["per_token_loss"],
        "pad_only_ce_contribution": reference_outputs["pad_only_ce_contribution"],
        "loss": reference_outputs["loss"],
    }
    labels = batch["labels"]
    if not isinstance(labels, torch.Tensor):
        raise AssertionError("S1 fixture labels must be a tensor")
    package_values: dict[str, object] = {
        "source_bytes": fixture["source_bytes"],
        "input_ids": trace.input_ids,
        "attention_mask": trace.attention_mask,
        "labels": labels,
        "pad_mask": labels.eq(module.model.pad_token_id),
        "decoder_input_ids": trace.decoder_input_ids,
        "task_ids": trace.task_ids,
        "encoder_memory": trace.encoder_memory,
        "decoder_hidden_state": trace.decoder_hidden_state,
        "logits": trace.logits,
        "per_token_loss": trace.per_token_loss,
        "pad_only_ce_contribution": trace.pad_only_ce_contribution,
        "loss": trace.loss,
    }
    per_token_loss = package_values["per_token_loss"]
    pad_mask = package_values["pad_mask"]
    package_loss = package_values["loss"]
    package_pad_loss = package_values["pad_only_ce_contribution"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (per_token_loss, pad_mask, package_loss, package_pad_loss)
    ):
        raise AssertionError("S1 loss surfaces must be tensors")
    valid_contribution = per_token_loss.masked_select(~pad_mask).sum() / labels.numel()
    torch.testing.assert_close(
        valid_contribution + package_pad_loss,
        package_loss,
        rtol=S1_TOLERANCE.rtol,
        atol=S1_TOLERANCE.atol,
    )
    if not _rng_equal(reference_after_rng, package_after_rng):
        raise AssertionError(f"{recipe.name}: paired post-forward RNG states diverged")
    if not _rng_equal(caller_rng, caller_restored_rng):
        raise AssertionError(f"{recipe.name}: caller RNG state was not restored")
    state_hashes = (
        reference_initial_hash,
        package_initial_hash,
        reference_before_hash,
        package_before_hash,
        reference_after_hash,
        package_after_hash,
    )
    if any(value != reference_initial_hash for value in state_hashes):
        raise AssertionError(f"{recipe.name}: model state hash changed")
    return {
        "family": recipe.name,
        "initial_state_sha256": reference_initial_hash,
        "package_initial_state_sha256": package_initial_hash,
        "state_hashes_unchanged": True,
        "rng_before_sha256": _rng_sha256(boundary_rng),
        "reference_rng_after_sha256": _rng_sha256(reference_after_rng),
        "package_rng_after_sha256": _rng_sha256(package_after_rng),
        "rng_post_equal": True,
        "caller_rng_restored": True,
        "max_errors": _surface_error_summary(reference_values, package_values),
        "first_divergence": _first_divergence(
            recipe.name, reference_values, package_values
        ),
    }


@pytest.mark.parametrize(
    "recipe", tuple(TRAINING_RECIPES.values()), ids=lambda item: item.name
)
def test_s1_all_recipe_fixed_batch_pre_optimizer_trace_matches(recipe) -> None:
    """Compare one original/package padded batch before any optimizer mutation."""
    evidence = _run_s1_case(recipe, _s1_device())
    assert evidence["first_divergence"] is None, json.dumps(
        evidence["first_divergence"], sort_keys=True
    )


@pytest.mark.parametrize(
    "recipe", tuple(TRAINING_RECIPES.values()), ids=lambda item: item.name
)
def test_s0_per_family_serialization_and_task_ids_match(recipe) -> None:
    labels = [1, 2]
    bboxes = [[3, 4, 5, 6], [7, 8, 9, 10]]
    relations = [(1, 1, 2, 1, 3)]
    id2label = dict(reference_label_map(recipe))
    package_base = T5LayoutSequence(id2label, add_sep_token=True)
    package_gen_t = T5LayoutSequenceForGenT(id2label, add_sep_token=True)
    package_gen_r = T5LayoutSequenceForGenR(id2label, add_sep_token=True)
    for task_index, task in enumerate(recipe.tasks):
        reference = reference_serialization(
            recipe,
            str(task),
            labels=labels,
            bboxes=bboxes,
            relations=relations,
        )
        if str(task) == "refinement":
            package_input = package_base.build_seq(labels, bboxes)
        elif str(task) == "completion":
            package_input = package_base.build_seq(labels[:1], bboxes[:1])
        elif str(task) == "ugen":
            package_input = ""
        elif str(task) in {"gen_t", "gen_ts"}:
            package_input = package_gen_t.build_input_seq(
                str(task),
                labels,
                bboxes,
                add_unk_for_label="gen_t_add_unk_token" in recipe.serialization_flags,
                add_unk_for_label_size="gen_ts_add_unk_token"
                in recipe.serialization_flags,
            )
        else:
            package_input = package_gen_r.build_input_seq(
                labels,
                relations,
                add_unk_token="gen_r_add_unk_token" in recipe.serialization_flags,
                compact="gen_r_compact" in recipe.serialization_flags,
            )
        if "add_task_prompt" in recipe.serialization_flags:
            package_input = f"{reference['prompt']} {package_input}"
        assert package_input.encode("utf-8") == reference["input_bytes"]
        assert (
            package_base.build_seq(labels, bboxes).encode("utf-8")
            == reference["output_bytes"]
        )
        assert reference["task_id"] == recipe.task_ids[task_index]


def test_s0_original_adam_and_warmuplr_are_actually_constructed() -> None:
    states = reference_optimizer_scheduler_state((1000, 2000, 3000, 4000))
    assert states["deepspeed_version"] == "0.5.10"
    assert states["optimizer"] == "torch.optim.adam.Adam"
    assert states["scheduler"] == "deepspeed.runtime.lr_schedules.WarmupLR"
    assert states["optimizer_defaults"] == {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
    }
    for warmup_steps in (1000, 2000, 3000, 4000):
        reference_lrs = states["lr_sequences"][str(warmup_steps)]
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        package_scheduler = LayoutFormerPPWarmupLR(
            optimizer,
            warmup_num_steps=warmup_steps,
            warmup_max_lr=1e-4,
        )
        package_lrs = [optimizer.param_groups[0]["lr"]]
        package_scheduler.step()
        package_lrs.append(optimizer.param_groups[0]["lr"])
        package_scheduler.step()
        package_lrs.append(optimizer.param_groups[0]["lr"])
        assert package_lrs == pytest.approx(reference_lrs)
        assert reference_lrs[2] == pytest.approx(
            1e-4 * math.log(2.0) / math.log(float(warmup_steps))
        )


def test_s0_checkpoint_selection_payload_and_static_branches_execute() -> None:
    behavior = reference_checkpoint_behavior()
    assert behavior == {
        "eval_loss_updates": [True, True, False],
        "best_eval_loss": 4.0,
        "best_epoch_written": "7",
        "best_miou_written": {"gen_t": {"epoch": 7, "value": 0.5}},
        "non_best_preserves_epoch": True,
        "epoch_checkpoint_expression": "self.model.state_dict()",
        "checkpoint_has_optimizer_state": False,
        "checkpoint_has_scheduler_state": False,
        "basic_device_cases": {
            "cpu": {"device": "cpu", "data_parallel": False},
            "cuda_single": {"device": "cuda:0", "data_parallel": False},
            "cuda_multi": {"device": "cuda:0", "data_parallel": True},
        },
        "trainer_modes": ["basic", "deepspeed"],
        "ddp_rejected": True,
    }


def test_s0_static_branch_seed_and_rico_map_facts() -> None:
    assert source_facts() == {
        "model_before_trainer": True,
        "seed_inside_trainer_setup": True,
        "ddp_training_rejected": True,
        "basic_supported": True,
        "deepspeed_supported": True,
        "optimizer": "torch.optim.Adam",
        "scheduler": "deepspeed.runtime.lr_schedules.WarmupLR",
        "optimizer_call_found": True,
        "scheduler_call_found": True,
        "scheduler_after_optimizer_update": True,
        "basic_device_cpu_fallback": True,
        "basic_device_cuda_zero": True,
        "basic_multigpu_data_parallel": True,
        "basic_multigpu_scales_batch_size": True,
        "deepspeed_initializes_distributed": True,
        "checkpoint_selector": "min_eval_loss_best_epoch",
        "checkpoint_measure_is_eval_loss": True,
        "checkpoint_payload_is_model_state_only": True,
        "multitask_writes_every_epoch_state": True,
        "multitask_best_epoch_is_external_pointer": True,
        "amp_disabled": True,
        "ema_disabled": True,
    }
    assert dict(RICO25_LABEL_TRANSLATION.public_to_sequence)[3] == 5
    assert dict(RICO25_LABEL_TRANSLATION.sequence_to_public)[4] == 4
    assert len(RICO25_LABEL_TRANSLATION.sha256) == 64
