from pathlib import Path
import json
import os
import pickle
import random
import sys
from typing import cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.modeling_outputs import LayoutGenerationOutput
from ralf import (
    RalfConfig,
    RalfForConditionalLayoutGeneration,
    RalfLayoutTokenizer,
    RalfPipeline,
    RalfProcessor,
)
from ralf.datasets import _IndexableDataset, build_retrieved_batch
from ralf.modeling_ralf import TASK_BY_CONDITION


def _reference_dir() -> Path:
    return Path(os.environ.get("RALF_REFERENCE_DIR", ".cache/ralf/references"))


def _cache_dir() -> Path:
    return Path(os.environ.get("RALF_CACHE_DIR", ".cache/ralf/cache"))


def _converted_dir() -> Path:
    return Path(os.environ.get("RALF_CONVERTED_DIR", ".cache/ralf/converted"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _vendor_root() -> Path:  # pragma: no cover
    return _repo_root() / "vendor" / "ralf"


CHECKPOINT_CASES = [
    ("cgl", "unconditional", "ralf_uncond_cgl", "ralf-cgl-unconditional-strict"),
    ("cgl", "label", "ralf_c_cgl", "ralf-cgl-label-strict"),
    ("cgl", "label_size", "ralf_cwh_cgl", "ralf-cgl-label-size-strict"),
    ("cgl", "completion", "ralf_partial_cgl", "ralf-cgl-completion-strict"),
    ("cgl", "refinement", "ralf_refinement_cgl", "ralf-cgl-refinement-strict"),
    ("cgl", "relation", "ralf_relation_cgl", "ralf-cgl-relation-strict"),
    ("pku", "unconditional", "ralf_uncond_pku10", "ralf-pku-unconditional-strict"),
    ("pku", "label", "ralf_c_pku10", "ralf-pku-label-strict"),
    ("pku", "label_size", "ralf_cwh_pku10", "ralf-pku-label-size-strict"),
    ("pku", "completion", "ralf_partial_pku10", "ralf-pku-completion-strict"),
    ("pku", "refinement", "ralf_refinement_pku10", "ralf-pku-refinement-strict"),
    ("pku", "relation", "ralf_relation_pku10", "ralf-pku-relation-strict"),
]

RUNTIME_PARITY_TASKS = [
    ("cgl", "label"),
    ("cgl", "label_size"),
    ("cgl", "completion"),
    ("cgl", "refinement"),
    ("cgl", "relation"),
    ("pku", "label"),
    ("pku", "label_size"),
    ("pku", "completion"),
    ("pku", "refinement"),
    ("pku", "relation"),
]


def _ensure_vendor_path() -> None:  # pragma: no cover
    vendor_root = _vendor_root()
    if not (vendor_root / "image2layout").exists():
        skip_or_fail_vendor_parity(
            "vendor/ralf is required for vendor parity",
            missing_paths=[vendor_root / "image2layout"],
            regeneration_hint="initialize the RALF vendor submodule before running vendor parity",
        )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


def _skip_missing_vendor_dependency(
    exc: ModuleNotFoundError,
) -> None:  # pragma: no cover
    skip_or_fail_vendor_parity(
        f"RALF vendor dependency is required: {exc.name}",
        regeneration_hint="install the RALF vendor optional dependencies before running vendor parity",
    )


def _vendor_label_features(config: RalfConfig):  # pragma: no cover
    _ensure_vendor_path()
    try:
        from datasets import ClassLabel, Features, Sequence
    except ModuleNotFoundError as exc:
        _skip_missing_vendor_dependency(exc)
        raise

    id2label = cast(dict[int | str, str], config.id2label)
    label_names = [
        label for _idx, label in sorted(id2label.items(), key=lambda item: int(item[0]))
    ]
    return Features({"label": Sequence(ClassLabel(names=label_names))})


def _build_vendor_tokenizer(config: RalfConfig):  # pragma: no cover
    _ensure_vendor_path()
    try:
        from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
    except ModuleNotFoundError as exc:
        _skip_missing_vendor_dependency(exc)
        raise

    features = _vendor_label_features(config)
    return LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=config.max_seq_length,
        num_bin=config.num_bin,
        var_order=list(config.var_order),
        pad_until_max=False,
        special_tokens=list(config.special_tokens),
        is_loc_vocab_shared=config.is_loc_vocab_shared,
        geo_quantization=config.geo_quantization,
    )


@pytest.mark.vendor_parity
def test_vendor_reference_metadata_exists() -> None:
    metadata = _reference_dir() / "golden_metadata.json"
    if not metadata.exists():
        skip_or_fail_vendor_parity(
            "Run generate_reference_outputs.py with the RALF cache to create metadata",
            missing_paths=[metadata],
            regeneration_hint="run models/ralf/scripts/generate_reference_outputs.py with RALF cache paths",
        )
    data = json.loads(metadata.read_text())
    assert data["status"] == "vendor-run"
    expected_gpu = os.environ.get("RALF_EXPECTED_REFERENCE_GPU")
    if expected_gpu is not None:
        assert data["gpu"] == expected_gpu
    assert data["torch_force_no_weights_only_load"] is True


@pytest.mark.vendor_parity
def test_vendor_reference_summary_contains_public_layout() -> None:
    summary = _reference_dir() / "golden_summary.json"
    if not summary.exists():
        skip_or_fail_vendor_parity(
            "Run generate_reference_outputs.py --run-vendor with the RALF cache",
            missing_paths=[summary],
            regeneration_hint="run models/ralf/scripts/generate_reference_outputs.py --run-vendor with RALF cache paths",
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
def test_local_tokenizer_matches_vendor_token_mask_and_round_trip() -> (
    None
):  # pragma: no cover
    config = RalfConfig(
        id2label={0: "logo", 1: "text", 2: "underlay"},
        max_seq_length=2,
        num_bin=8,
    )
    tokenizer = RalfLayoutTokenizer(config)
    vendor_tokenizer = _build_vendor_tokenizer(config)

    labels = torch.tensor([[0, 1]])
    bbox = torch.tensor([[[0.25, 0.5, 0.125, 0.375], [0.75, 0.25, 0.5, 0.25]]])
    mask = torch.tensor([[True, False]])
    local_encoded = tokenizer.encode_layout(labels=labels, bbox=bbox, mask=mask)
    vendor_inputs = {
        "label": labels.clone(),
        "center_x": bbox[..., 0].clone(),
        "center_y": bbox[..., 1].clone(),
        "width": bbox[..., 2].clone(),
        "height": bbox[..., 3].clone(),
        "mask": mask.clone(),
    }
    vendor_encoded = vendor_tokenizer.encode(vendor_inputs)

    assert torch.equal(tokenizer.token_mask(), vendor_tokenizer.token_mask)
    assert torch.equal(local_encoded["input_ids"], vendor_encoded["seq"])
    assert torch.equal(local_encoded["attention_mask"], vendor_encoded["mask"])

    local_decoded = tokenizer.decode_layout(local_encoded["input_ids"])
    vendor_decoded = vendor_tokenizer.decode(vendor_encoded["seq"][:, 1:])
    vendor_bbox = torch.stack(
        [
            vendor_decoded["center_x"],
            vendor_decoded["center_y"],
            vendor_decoded["width"],
            vendor_decoded["height"],
        ],
        dim=-1,
    )
    assert torch.equal(local_decoded["mask"], vendor_decoded["mask"])
    valid = local_decoded["mask"]
    assert torch.equal(local_decoded["labels"][valid], vendor_decoded["label"][valid])
    assert torch.allclose(local_decoded["bbox"], vendor_bbox)


def _build_vendor_reference_model(
    converted_model: RalfForConditionalLayoutGeneration,
):
    _ensure_vendor_path()

    from datasets import Dataset
    import image2layout.train.fid.model as fid_model
    import image2layout.train.helpers.layout_tokenizer as vendor_tokenizer
    import image2layout.train.models.common.image as vendor_image
    from image2layout.train.models.retrieval_augmented_autoreg import (
        ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
    )

    precomputed_dir = (_cache_dir() / "PRECOMPUTED_WEIGHT_DIR").resolve()
    if not precomputed_dir.exists():
        skip_or_fail_vendor_parity(
            "RALF PRECOMPUTED_WEIGHT_DIR cache is required",
            missing_paths=[precomputed_dir],
            regeneration_hint="populate RALF PRECOMPUTED_WEIGHT_DIR under RALF_CACHE_DIR before vendor parity",
        )
    fid_model.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)
    vendor_image.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)
    vendor_tokenizer.PRECOMPUTED_WEIGHT_DIR = str(precomputed_dir)

    config = converted_model.config
    features = _vendor_label_features(config)
    tokenizer = _build_vendor_tokenizer(config)
    dataset_name = "pku" if config.dataset_name.startswith("pku") else "cgl"
    task_name = str(config.retrieval_metadata.get("task_name", "uncond"))
    previous_cwd = Path.cwd()
    if task_name == "relation":
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.chdir(_cache_dir().parent.resolve())
    try:
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
            auxilary_task=task_name,
            use_flag_embedding=config.use_flag_embedding,
            use_multitask=config.use_multitask,
            RELATION_SIZE=config.relation_size,
            global_task_embedding=config.global_task_embedding,
        )
    finally:
        if task_name == "relation":
            os.chdir(previous_cwd)


def _retrieval_index_path(dataset_name: str) -> Path:
    return (
        _cache_dir()
        / "PRECOMPUTED_WEIGHT_DIR"
        / "retrieval_indexes"
        / f"{dataset_name}_test_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt"
    )


def _load_relationship_table() -> dict[str, list[object]]:  # pragma: no cover
    _ensure_vendor_path()
    relationship_path = (
        _cache_dir() / "pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt"
    )
    if not relationship_path.exists():
        skip_or_fail_vendor_parity(
            "RALF relationship table is required for relation parity",
            missing_paths=[relationship_path],
            regeneration_hint="populate pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt under RALF_CACHE_DIR",
        )
    return cast(
        dict[str, list[object]],
        torch.load(relationship_path, map_location="cpu", weights_only=False),
    )


def _runtime_fixture(
    dataset_name: str,
) -> tuple[str | int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset_name == "cgl":
        from datasets import load_dataset
        from ralf.datasets import normalize_org_sample

        dataset_path = _cache_dir() / "dataset" / "cgl"
        if not dataset_path.exists():
            skip_or_fail_vendor_parity(
                "CGL cache dataset is required for runtime parity",
                missing_paths=[dataset_path],
                regeneration_hint="unpack the RALF cache dataset/cgl directory",
            )
        sample = load_dataset(str(dataset_path), split="test")[0]
        normalized = normalize_org_sample(sample, "cgl")
        return (
            cast(str, sample["id"]),
            cast(torch.Tensor, normalized["labels"]).unsqueeze(0),
            cast(torch.Tensor, normalized["bbox"]).unsqueeze(0),
            cast(torch.Tensor, normalized["mask"]).unsqueeze(0),
        )

    generated = (
        _cache_dir()
        / "training_logs"
        / "ralf_uncond_pku10"
        / "generated_samples_uncond_name_top_k_temperature_1.0_top_k_5_final_dynamictopk_16"
        / "test_0.pkl"
    )
    if not generated.exists():
        skip_or_fail_vendor_parity(
            "PKU generated sample cache is required for runtime parity",
            missing_paths=[generated],
            regeneration_hint="unpack the RALF cache training_logs/ralf_uncond_pku10 generated samples",
        )
    with generated.open("rb") as f:
        data = pickle.load(f)
    first = data["results"][0]
    labels = torch.tensor([first["label"]], dtype=torch.long)
    bbox = torch.tensor(
        [
            [
                [x, y, w, h]
                for x, y, w, h in zip(
                    first["center_x"],
                    first["center_y"],
                    first["width"],
                    first["height"],
                    strict=True,
                )
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(labels, dtype=torch.bool)
    return int(first["id"]), labels, bbox, mask


def _vendor_condition_inputs(
    *,
    config: RalfConfig,
    task_name: str,
    sample_id: str | int,
    labels: torch.Tensor,
    bbox: torch.Tensor,
    mask: torch.Tensor,
):
    _ensure_vendor_path()
    try:
        from image2layout.train.helpers.task import get_condition
    except ModuleNotFoundError as exc:
        _skip_missing_vendor_dependency(exc)
        raise

    tokenizer = _build_vendor_tokenizer(config)
    batch = {
        "id": [sample_id],
        "image": torch.zeros(1, 3, 64, 64),
        "saliency": torch.zeros(1, 1, 64, 64),
        "label": labels.clone(),
        "center_x": bbox[..., 0].clone(),
        "center_y": bbox[..., 1].clone(),
        "width": bbox[..., 2].clone(),
        "height": bbox[..., 3].clone(),
        "mask": mask.clone(),
    }
    condition, _ = get_condition(batch, task_name, tokenizer)
    return condition


def _local_condition_tokens(
    *,
    config: RalfConfig,
    task_name: str,
    condition: object,
    relationship_table: dict[str, list[object]],
    seed: int,
) -> dict[str, torch.Tensor]:
    image = cast(torch.Tensor, getattr(condition, "image"))
    model = RalfForConditionalLayoutGeneration(config)
    random.seed(seed)
    torch.manual_seed(seed)
    prepared = model._prepare_conditional_inputs(
        pixel_values=image[:, :3],
        saliency=image[:, 3:4],
        retrieved=None,
        batch_size=image.size(0),
        condition_type=task_name,
        constraint_input_ids=cast(torch.Tensor | None, getattr(condition, "seq")),
        constraint_mask=cast(torch.Tensor | None, getattr(condition, "mask")),
        relationship_table=relationship_table if task_name == "relation" else None,
        sample_ids=getattr(condition, "id", None),
    )
    return {
        "seq": cast(torch.Tensor, prepared["seq_layout_const"]),
        "pad_mask": cast(torch.Tensor, prepared["seq_layout_const_pad_mask"]),
    }


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
def test_local_pipeline_matches_vendor_golden_cgl_e2e() -> None:  # pragma: no cover
    summary_path = _reference_dir() / "golden_summary.json"
    converted = _converted_dir() / "ralf-cgl-unconditional-strict"
    retrieval_index_path = (
        _cache_dir()
        / "PRECOMPUTED_WEIGHT_DIR"
        / "retrieval_indexes"
        / "cgl_test_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt"
    )
    dataset_path = _cache_dir() / "dataset" / "cgl"
    if (
        not summary_path.exists()
        or not converted.exists()
        or not retrieval_index_path.exists()
        or not dataset_path.exists()
    ):
        skip_or_fail_vendor_parity(
            "Run CGL reference generation and strict conversion before e2e parity",
            missing_paths=[summary_path, converted, retrieval_index_path, dataset_path],
            regeneration_hint=(
                "run models/ralf/scripts/generate_reference_outputs.py --run-vendor "
                "and strict RALF conversion for CGL"
            ),
        )
    if not torch.cuda.is_available():
        skip_or_fail_vendor_parity(
            "GPU 0 is required for RALF e2e parity",
            missing_paths=["CUDA device 0"],
            regeneration_hint="rerun on a host with GPU 0 and generated RALF assets",
        )

    from datasets import load_dataset

    summary = json.loads(summary_path.read_text())
    first = summary["first_result"]
    test_dataset = load_dataset(str(dataset_path), split="test")
    train_dataset = load_dataset(str(dataset_path), split="train")
    sample = test_dataset[0]
    assert sample["id"] == first["id"]
    retrieval_indexes = torch.load(
        retrieval_index_path,
        map_location="cpu",
        weights_only=False,
    )
    indexes = torch.tensor([retrieval_indexes[sample["id"]][:16]])
    retrieved = build_retrieved_batch(
        cast(_IndexableDataset, train_dataset),
        indexes,
        max_seq_length=10,
        dataset_name="cgl",
    )
    pipe = RalfPipeline.from_pretrained(converted, local_files_only=True)
    pipe.model.to(torch.device("cuda:0"))

    output = cast(
        LayoutGenerationOutput,
        pipe(
            images=[sample["image"]],
            saliency=[sample["saliency"]],
            retrieval={
                "items": {
                    "bbox": retrieved.bbox,
                    "labels": retrieved.labels,
                    "mask": retrieved.mask,
                },
                "ids": indexes,
            },
            seed=int(
                json.loads((_reference_dir() / "golden_metadata.json").read_text())[
                    "seed"
                ]
            ),
            top_k=int(summary["test_cfg"]["sampling"]["top_k"]),
        ),
    )
    valid = output.mask[0]
    assert int(valid.sum()) == len(first["mask"])
    assert all(first["mask"])
    assert output.labels[0][valid].tolist() == first["labels"]
    assert torch.equal(
        output.bbox[0][valid].cpu(),
        torch.tensor(first["bbox"], dtype=torch.float32),
    )


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("name", "condition_type", "job_name", "converted_name"),
    CHECKPOINT_CASES,
)
def test_converted_checkpoint_matches_local_weights_and_vendor_logits(
    name: str, condition_type: str, job_name: str, converted_name: str
) -> None:
    checkpoint = _cache_dir() / "training_logs" / job_name / "gen_final_model.pt"
    converted = _converted_dir() / converted_name
    if not checkpoint.exists() or not converted.exists():
        skip_or_fail_vendor_parity(
            "Run strict RALF conversion for all RALF checkpoints before parity tests",
            missing_paths=[checkpoint, converted],
            regeneration_hint="run the strict RALF conversion commands for all CGL and PKU tasks before vendor parity",
        )

    report = json.loads((converted / "conversion_report.json").read_text())
    assert report["task"] == condition_type
    assert report["source_key_count"] == 664
    assert report["target_key_count"] == 664
    assert len(report["matched_keys"]) == 664
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["skipped_shape_mismatch_keys"] == {}
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
        skip_or_fail_vendor_parity(
            "GPU 0 is required for RALF local-vs-vendor logits parity",
            missing_paths=["CUDA device 0"],
            regeneration_hint="rerun on a host with GPU 0 and generated RALF assets",
        )
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


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset_name", "condition_type"), RUNTIME_PARITY_TASKS)
def test_condition_runtime_path_matches_vendor_tokens_and_retrieval_indexes(
    dataset_name: str,
    condition_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted = (
        _converted_dir()
        / f"ralf-{dataset_name}-{condition_type.replace('_', '-')}-strict"
    )
    if condition_type == "label_size":
        converted = _converted_dir() / f"ralf-{dataset_name}-label-size-strict"
    if condition_type == "completion":
        converted = _converted_dir() / f"ralf-{dataset_name}-completion-strict"
    converted = converted.resolve()
    if not converted.exists():
        skip_or_fail_vendor_parity(
            "Run strict RALF conversion before runtime parity",
            missing_paths=[converted],
            regeneration_hint="run the strict conversion commands for all conditional checkpoints",
        )
    retrieval_path = _retrieval_index_path(dataset_name).resolve()
    if not retrieval_path.exists():
        skip_or_fail_vendor_parity(
            "RALF retrieval index table is required for runtime parity",
            missing_paths=[retrieval_path],
            regeneration_hint="populate PRECOMPUTED_WEIGHT_DIR/retrieval_indexes under RALF_CACHE_DIR",
        )

    config = RalfConfig.from_pretrained(converted, local_files_only=True)
    task_name = TASK_BY_CONDITION[condition_type]
    sample_id, labels, bbox, mask = _runtime_fixture(dataset_name)

    retrieval_table = torch.load(retrieval_path, map_location="cpu", weights_only=False)
    retrieval_key = int(sample_id) if dataset_name == "pku" else str(sample_id)
    assert retrieval_key in retrieval_table
    retrieved_indexes = torch.tensor([retrieval_table[retrieval_key][: config.top_k]])
    assert retrieved_indexes.shape == (1, config.top_k)

    relationship_table = _load_relationship_table()
    monkeypatch.chdir(_cache_dir().resolve().parent)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    torch.manual_seed(1234)
    condition = _vendor_condition_inputs(
        config=config,
        task_name=task_name,
        sample_id=sample_id,
        labels=labels,
        bbox=bbox,
        mask=mask,
    )

    try:
        import image2layout.train.models.layoutformerpp.task_preprocessor as task_preprocessor
    except ModuleNotFoundError as exc:
        _skip_missing_vendor_dependency(exc)
        raise

    random.seed(777)
    torch.manual_seed(777)
    vendor_preprocessor = task_preprocessor.PREPROCESSOR[task_name](
        tokenizer=_build_vendor_tokenizer(config),
        global_task_embedding=config.global_task_embedding,
    )
    vendor_tokens = vendor_preprocessor(condition)

    random.seed(777)
    torch.manual_seed(777)
    local_tokens = _local_condition_tokens(
        config=config,
        task_name=task_name,
        condition=condition,
        relationship_table=relationship_table,
        seed=777,
    )

    assert torch.equal(local_tokens["seq"].cpu(), vendor_tokens["seq"].cpu())
    assert torch.equal(local_tokens["pad_mask"].cpu(), vendor_tokens["pad_mask"].cpu())

    processor = RalfProcessor.from_pretrained(converted, local_files_only=True)
    output = processor.post_process_layouts(
        cast(torch.Tensor, condition.seq).clone().cpu(),
        output_type="dataclass",
        intermediates={
            "runtime_parity": {
                "dataset": dataset_name,
                "condition_type": condition_type,
                "task_name": task_name,
                "sample_id": str(sample_id),
                "retrieved_indexes": retrieved_indexes,
                "generation_sequence_parity": "not-deterministic-top-k-sampling; condition path exact, decoded schema/range asserted",
            }
        },
    )
    assert isinstance(output, LayoutGenerationOutput)
    output_bbox = cast(torch.Tensor, output.bbox)
    output_labels = cast(torch.Tensor, output.labels)
    output_mask = cast(torch.Tensor, output.mask)
    assert output_bbox.ndim == 3
    assert output_labels.ndim == 2
    assert output_mask.ndim == 2
    assert output_bbox.shape[:2] == output_labels.shape == output_mask.shape
    assert bool(torch.all((0.0 <= output_bbox) & (output_bbox <= 1.0)).item())
