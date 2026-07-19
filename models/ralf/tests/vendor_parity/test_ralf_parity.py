from pathlib import Path
import json
import os
import sys
from typing import cast

import pytest
import torch

from ralf import RalfForConditionalLayoutGeneration


def _reference_dir() -> Path:
    return Path(os.environ.get("RALF_REFERENCE_DIR", ".cache/ralf/references"))


def _cache_dir() -> Path:
    return Path(os.environ.get("RALF_CACHE_DIR", ".cache/ralf/cache"))


def _converted_dir() -> Path:
    return Path(os.environ.get("RALF_CONVERTED_DIR", ".cache/ralf/converted"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.mark.vendor_parity
def test_vendor_reference_metadata_exists() -> None:
    metadata = _reference_dir() / "golden_metadata.json"
    if not metadata.exists():
        pytest.skip(
            "Run generate_reference_outputs.py with the RALF cache to create metadata"
        )
    data = json.loads(metadata.read_text())
    assert data["status"] == "vendor-run"
    assert data["gpu"] == "0"
    assert data["torch_force_no_weights_only_load"] is True


@pytest.mark.vendor_parity
def test_vendor_reference_summary_contains_public_layout() -> None:
    summary = _reference_dir() / "golden_summary.json"
    if not summary.exists():
        pytest.skip(
            "Run generate_reference_outputs.py --run-vendor with the RALF cache"
        )
    data = json.loads(summary.read_text())
    assert data["num_results"] >= 1
    first = data["first_result"]
    assert len(first["labels"]) == len(first["bbox"]) == len(first["mask"])
    assert all(first["mask"])
    for bbox in first["bbox"]:
        assert len(bbox) == 4
        assert all(0.0 <= value <= 1.0 for value in bbox)


def _clone_inputs(inputs: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, dict):
            cloned[key] = {
                inner_key: inner_value.clone()
                for inner_key, inner_value in value.items()
                if torch.is_tensor(inner_value)
            }
        elif torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _to_device(inputs: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device)
                for inner_key, inner_value in value.items()
                if torch.is_tensor(inner_value)
            }
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _synthetic_vendor_inputs(
    model: RalfForConditionalLayoutGeneration,
) -> dict[str, object]:
    config = model.config
    generator = torch.Generator().manual_seed(44)
    batch_size = 1
    height = width = 64
    token_length = 6
    constraint_vocab = model.user_const_encoder.emb.num_embeddings
    return {
        "image": torch.rand((batch_size, 4, height, width), generator=generator),
        "retrieved": {
            "image": torch.rand(
                (batch_size, config.top_k, 4, height, width),
                generator=generator,
            ),
            "saliency": torch.rand(
                (batch_size, config.top_k, 1, height, width),
                generator=generator,
            ),
            "center_x": torch.rand(
                (batch_size, config.top_k, config.max_seq_length),
                generator=generator,
            ),
            "center_y": torch.rand(
                (batch_size, config.top_k, config.max_seq_length),
                generator=generator,
            ),
            "width": torch.rand(
                (batch_size, config.top_k, config.max_seq_length),
                generator=generator,
            ),
            "height": torch.rand(
                (batch_size, config.top_k, config.max_seq_length),
                generator=generator,
            ),
            "label": torch.randint(
                0,
                config.num_labels,
                (batch_size, config.top_k, config.max_seq_length),
                generator=generator,
            ),
            "mask": torch.ones(
                (batch_size, config.top_k, config.max_seq_length),
                dtype=torch.bool,
            ),
        },
        "seq": torch.randint(
            0, config.vocab_size, (batch_size, token_length), generator=generator
        ),
        "tgt_key_padding_mask": torch.zeros(
            (batch_size, token_length), dtype=torch.bool
        ),
        "seq_layout_const": torch.randint(
            0,
            constraint_vocab,
            (batch_size, token_length),
            generator=generator,
        ),
        "seq_layout_const_pad_mask": torch.zeros(
            (batch_size, token_length), dtype=torch.bool
        ),
    }


def _build_vendor_reference_model(
    converted_model: RalfForConditionalLayoutGeneration,
):
    vendor_root = _repo_root() / "vendor" / "ralf"
    if not vendor_root.exists():
        pytest.skip("vendor/ralf is required for local-vs-vendor parity")
    sys.path.insert(0, str(vendor_root))

    from datasets import ClassLabel, Dataset, Features, Sequence
    import image2layout.train.fid.model as fid_model
    import image2layout.train.helpers.layout_tokenizer as vendor_tokenizer
    import image2layout.train.models.common.image as vendor_image
    from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from image2layout.train.models.retrieval_augmented_autoreg import (
        ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
    )

    precomputed_dir = _cache_dir() / "PRECOMPUTED_WEIGHT_DIR"
    if not precomputed_dir.exists():
        pytest.skip("RALF PRECOMPUTED_WEIGHT_DIR cache is required")
    fid_model.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)
    vendor_image.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)
    vendor_tokenizer.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)

    config = converted_model.config
    id2label = cast(dict[int | str, str], config.id2label)
    label_names = [
        label for _idx, label in sorted(id2label.items(), key=lambda item: int(item[0]))
    ]
    features = Features({"label": Sequence(ClassLabel(names=label_names))})
    tokenizer = LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=config.max_seq_length,
        num_bin=config.num_bin,
        var_order=list(config.var_order),
        pad_until_max=False,
        special_tokens=list(config.special_tokens),
        is_loc_vocab_shared=config.is_loc_vocab_shared,
        geo_quantization=config.geo_quantization,
    )
    dataset_name = "pku" if config.dataset_name.startswith("pku") else "cgl"
    return ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
        features=features,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_seq_length=config.max_seq_length,
        db_dataset=Dataset.from_dict({"id": []}),
        d_model=config.d_model,
        decoder_d_model=config.decoder_d_model,
        top_k=config.top_k,
        layout_backbone=config.layout_backbone,
        use_reference_image=config.use_reference_image,
        freeze_layout_encoder=config.freeze_layout_encoder,
        retrieval_backbone=config.retrieval_backbone,
        random_retrieval=False,
        saliency_k="None",
        auxilary_task="uncond",
        use_flag_embedding=config.use_flag_embedding,
        use_multitask=config.use_multitask,
        RELATION_SIZE=config.relation_size,
        global_task_embedding=config.global_task_embedding,
    )


def _local_logits(
    model: RalfForConditionalLayoutGeneration, inputs: dict[str, object]
) -> torch.Tensor:
    encoded = model._encode_into_memory(
        cast(dict[str, torch.Tensor | dict[str, torch.Tensor]], inputs)
    )
    return model.decoder(
        tgt=cast(torch.Tensor, inputs["seq"]),
        tgt_key_padding_mask=cast(torch.Tensor, inputs["tgt_key_padding_mask"]),
        is_causal=True,
        **encoded,
    )


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("name", "job_name", "converted_name"),
    [
        ("cgl", "ralf_uncond_cgl", "ralf-cgl-unconditional-strict"),
        ("pku", "ralf_uncond_pku10", "ralf-pku-unconditional-strict"),
    ],
)
def test_converted_checkpoint_matches_local_weights_and_vendor_logits(
    name: str, job_name: str, converted_name: str
) -> None:
    checkpoint = _cache_dir() / "training_logs" / job_name / "gen_final_model.pt"
    converted = _converted_dir() / converted_name
    if not checkpoint.exists() or not converted.exists():
        pytest.skip("Run strict RALF conversion for CGL and PKU before parity tests")

    report = json.loads((converted / "conversion_report.json").read_text())
    assert report["source_key_count"] == 664
    assert report["target_key_count"] == 664
    assert len(report["matched_keys"]) == 664
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["weight_parity_ready"] is True

    source = torch.load(checkpoint, map_location="cpu")
    state_dict = (
        source.get("state_dict", source) if isinstance(source, dict) else source
    )
    converted_model = RalfForConditionalLayoutGeneration.from_pretrained(
        converted, local_files_only=True
    ).eval()
    converted_state = converted_model.state_dict()
    assert set(converted_state) == set(state_dict)
    for key, value in state_dict.items():
        assert torch.equal(converted_state[key].cpu(), value), f"{name}:{key}"

    if not torch.cuda.is_available():
        pytest.skip("GPU 0 is required for RALF local-vs-vendor logits parity")
    device = torch.device("cuda:0")
    reference_model = _build_vendor_reference_model(converted_model).eval()
    reference_model.load_state_dict(state_dict, strict=True)
    reference_model.to(device)
    converted_model.to(device)
    inputs = _synthetic_vendor_inputs(converted_model)
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        reference_logits = reference_model(_clone_inputs(inputs))["logits"]
        converted_logits = _local_logits(converted_model, _clone_inputs(inputs))
    assert torch.equal(reference_logits, converted_logits)
