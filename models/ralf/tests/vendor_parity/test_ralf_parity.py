from pathlib import Path
import json
import os

import pytest
import torch

from ralf import RalfForConditionalLayoutGeneration


def _reference_dir() -> Path:
    return Path(os.environ.get("RALF_REFERENCE_DIR", ".cache/ralf/references"))


def _cache_dir() -> Path:
    return Path(os.environ.get("RALF_CACHE_DIR", ".cache/ralf/cache"))


def _converted_dir() -> Path:
    return Path(os.environ.get("RALF_CONVERTED_DIR", ".cache/ralf/converted"))


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


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("name", "job_name", "converted_name"),
    [
        ("cgl", "ralf_uncond_cgl", "ralf-cgl-unconditional-strict"),
        ("pku", "ralf_uncond_pku10", "ralf-pku-unconditional-strict"),
    ],
)
def test_converted_checkpoint_matches_vendor_weights_and_logits(
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
        pytest.skip("GPU 0 is required for RALF vendor logits parity")
    device = torch.device("cuda:0")
    reference_model = RalfForConditionalLayoutGeneration(converted_model.config).eval()
    reference_model.load_state_dict(state_dict, strict=True)
    reference_model.to(device)
    converted_model.to(device)
    inputs = _synthetic_vendor_inputs(converted_model)
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        reference_logits = reference_model(vendor_inputs=_clone_inputs(inputs))[
            "logits"
        ]
        converted_logits = converted_model(vendor_inputs=_clone_inputs(inputs))[
            "logits"
        ]
    assert torch.equal(reference_logits, converted_logits)
