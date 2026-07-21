import torch
from typing import cast

from laygen.modeling_outputs import LayoutGenerationOutput

from layout_action import (
    LayoutActionConfig,
    LayoutActionForCausalLM,
    LayoutActionPipeline,
    LayoutActionProcessor,
    LayoutActionTokenizer,
)


def tiny_pipeline() -> LayoutActionPipeline:
    config = LayoutActionConfig(
        dataset_name="publaynet",
        max_elements=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    return LayoutActionPipeline(
        model=LayoutActionForCausalLM(config),
        processor=LayoutActionProcessor(LayoutActionTokenizer(config)),
        config=config,
    )


def test_pipeline_returns_layout_generation_output() -> None:
    pipe = tiny_pipeline()

    output = pipe(
        batch_size=1,
        condition_type="unconditional",
        sampling="greedy",
        num_inference_steps=2,
    )

    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.shape == (1, 1, 4)
    assert cast(torch.Tensor, output.sequences).shape == (1, 3)


def test_pipeline_label_condition_forces_label() -> None:
    pipe = tiny_pipeline()

    output = pipe(
        condition_type="label",
        labels=torch.tensor([[2]]),
        sampling="greedy",
        num_inference_steps=13,
        output_type="dict",
    )

    sequences = cast(torch.Tensor, output["sequences"])
    assert sequences[0, 1].item() == pipe.config.label_token_id(2)


def test_pipeline_save_pretrained_round_trip(tmp_path) -> None:
    pipe = tiny_pipeline()

    pipe.save_pretrained(tmp_path)
    restored = LayoutActionPipeline.from_pretrained(tmp_path, local_files_only=True)

    assert restored.config.model_type == "layout-action"


def test_pipeline_rejects_missing_component(tmp_path) -> None:
    pipe = tiny_pipeline()
    pipe.config.save_pretrained(tmp_path)

    try:
        LayoutActionPipeline.from_pretrained(tmp_path, local_files_only=True)
    except (FileNotFoundError, OSError):
        pass
    else:
        raise AssertionError("missing model component must raise")
