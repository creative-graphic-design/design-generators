from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from posterllava import PosterLlavaConfig, PosterLlavaPipeline, PosterLlavaProcessor
import posterllava.pipeline_posterllava as pipeline_module


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [ord(char) % 19 + 1 for char in text]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}

    def batch_decode(self, ids, skip_special_tokens: bool = True):
        _ = ids, skip_special_tokens
        return ["[{'label': 'text', 'box': [0.0, 0.0, 1.0, 1.0]}]"]

    def save_pretrained(self, save_directory: str | Path) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "tokenizer_config.json").write_text("{}\n")


class FakeImageProcessor:
    def preprocess(self, images, return_tensors: str = "pt"):
        _ = images, return_tensors
        return {"pixel_values": torch.zeros(1, 3, 4, 4)}

    def save_pretrained(self, save_directory: str | Path) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "preprocessor_config.json").write_text("{}\n")


class FakeModel:
    def generate(self, input_ids, **kwargs):
        _ = kwargs
        suffix = torch.tensor([[7, 8, 9]], dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
    ) -> None:
        _ = is_main_process
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "config.json").write_text("{}\n")


def test_pipeline_rejects_unsupported_conditions() -> None:
    pipe = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor.from_config(),
    )

    try:
        pipe(condition_type="text")
    except NotImplementedError as exc:
        assert "content_image" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


def test_pipeline_generates_schema_with_fake_components() -> None:
    tokenizer = FakeTokenizer()
    image_processor = FakeImageProcessor()
    pipe = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner", max_new_tokens=8),
        PosterLlavaProcessor(
            tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
            image_processor=image_processor,
        ),
        model=cast(PreTrainedModel, FakeModel()),
        tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
        image_processor=image_processor,
    )
    image = Image.new("RGB", (4, 4))

    output = cast(
        LayoutGenerationOutput,
        pipe(images=image, num_elements=1, do_sample=False),
    )

    assert_layout_output_schema(output, batch_size=1)
    assert output.id2label == {0: "text"}

    as_dict = pipe(
        images=image,
        num_elements=torch.tensor([1]),
        do_sample=False,
        output_type="dict",
        top_p=None,
        top_k=5,
        num_beams=None,
        return_intermediates=True,
    )
    assert as_dict["id2label"] == {0: "text"}


def test_pipeline_save_pretrained_round_trip_without_model(tmp_path) -> None:
    pipe = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor.from_config(),
    )

    pipe.save_pretrained(tmp_path)
    loaded = PosterLlavaPipeline.from_pretrained(tmp_path)

    assert loaded.config.dataset_name == "ad_banner"
    assert loaded.processor.id2label[0] == "header"


def test_pipeline_content_image_and_missing_component_errors() -> None:
    image = Image.new("RGB", (4, 4))
    pipe = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor.from_config(),
    )

    try:
        pipe(content={"image": image, "num_elements": 1, "text": "Sale"})
    except ValueError as exc:
        assert "model is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    pipe_with_model = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor.from_config(),
        model=cast(PreTrainedModel, FakeModel()),
    )
    try:
        pipe_with_model(images=image, num_elements=1)
    except ValueError as exc:
        assert "tokenizer is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    pipe_with_tokenizer = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor(
            tokenizer=cast(PreTrainedTokenizerBase, FakeTokenizer()),
        ),
        model=cast(PreTrainedModel, FakeModel()),
        tokenizer=cast(PreTrainedTokenizerBase, FakeTokenizer()),
    )
    try:
        pipe_with_tokenizer(images=image, num_elements=1)
    except ValueError as exc:
        assert "image_processor is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        pipe(images=image)
    except ValueError as exc:
        assert "num_elements is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        pipe(images=[image, image], num_elements=1, prompt=["one"])
    except ValueError as exc:
        assert "same batch size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pipeline_component_loaders_are_wired(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((str(path), kwargs))
            return object()

    monkeypatch.setattr(
        pipeline_module.AutoModelForCausalLM, "from_pretrained", Loader.from_pretrained
    )
    monkeypatch.setattr(
        pipeline_module.AutoTokenizer, "from_pretrained", Loader.from_pretrained
    )
    monkeypatch.setattr(
        pipeline_module.AutoImageProcessor, "from_pretrained", Loader.from_pretrained
    )

    pipeline_module._load_model_component(
        "repo", local_files_only=True, subfolder="model"
    )
    pipeline_module._load_tokenizer_component("repo", subfolder="tokenizer")
    pipeline_module._load_image_processor_component("repo")

    assert calls[0] == ("repo", {"local_files_only": True, "subfolder": "model"})
    assert calls[1] == ("repo", {"local_files_only": False, "subfolder": "tokenizer"})
    assert calls[2] == ("repo", {"local_files_only": False})


def test_pipeline_preprocess_rejects_bad_image_processor() -> None:
    pipe = PosterLlavaPipeline(
        PosterLlavaConfig(dataset_name="ad_banner"),
        PosterLlavaProcessor.from_config(),
        image_processor=object(),
    )

    try:
        pipe._preprocess_images([Image.new("RGB", (2, 2))])
    except ValueError as exc:
        assert "preprocess" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pipeline_from_components_builds_default_processor() -> None:
    config = PosterLlavaConfig(dataset_name="ad_banner")

    pipe = PosterLlavaPipeline._from_pretrained_components(
        config=config,
        components={},
    )

    assert pipe.processor.id2label[0] == "header"
