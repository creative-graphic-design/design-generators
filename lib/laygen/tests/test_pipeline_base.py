from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec


LOADER_CALLS: list[tuple[str | Path, str | None, bool]] = []


class ToyConfig(PretrainedConfig):
    model_type = "toy-layout-pipeline"

    def __init__(self, label: str = "text", **kwargs: object) -> None:
        _ = kwargs
        self.label = label
        super().__init__()


class ToyModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.saved_is_main_process: bool | None = None
        self.moved_device: torch.device | None = None
        self.moved_dtype: torch.dtype | None = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
    ) -> None:
        self.saved_is_main_process = is_main_process
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "component.txt").write_text(self.name, encoding="utf-8")

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.moved_device = device
        self.moved_dtype = dtype


class ToyProcessor:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(self, save_directory: str | Path) -> None:
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "processor.txt").write_text(self.name, encoding="utf-8")


def load_toy_model(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    _ = subfolder, local_files_only
    return ToyModel(Path(pretrained_model_name_or_path, "component.txt").read_text())


def load_toy_processor(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    _ = subfolder, local_files_only
    return ToyProcessor(
        Path(pretrained_model_name_or_path, "processor.txt").read_text()
    )


def record_loader(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    LOADER_CALLS.append((pretrained_model_name_or_path, subfolder, local_files_only))
    return ToyModel(str(pretrained_model_name_or_path))


class ToyPipeline(LayoutGenerationPipeline):
    config_class: ClassVar[type[PretrainedConfig]] = ToyConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            loader=load_toy_model,
            subfolder="model",
            marker_file="component.txt",
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=load_toy_processor,
            marker_file="processor.txt",
            save_with_is_main_process=False,
        ),
        "optional": PipelineComponentSpec(
            attribute_name="optional",
            loader=load_toy_model,
            subfolder="optional",
            marker_file="component.txt",
            required=False,
        ),
    }

    def __init__(
        self,
        config: ToyConfig,
        model: ToyModel,
        processor: ToyProcessor,
        optional: ToyModel | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.model = model
        self.processor = processor
        self.optional = optional

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "ToyPipeline":
        return cls(
            config=cast(ToyConfig, config),
            model=cast(ToyModel, components["model"]),
            processor=cast(ToyProcessor, components["processor"]),
            optional=cast(ToyModel | None, components["optional"]),
        )

    def __call__(self) -> LayoutGenerationOutput:
        return LayoutGenerationOutput(
            bbox=torch.zeros(1, 1, 4),
            labels=torch.zeros(1, 1, dtype=torch.long),
            mask=torch.ones(1, 1, dtype=torch.bool),
            id2label={0: self.config.label},
        )


def test_pipeline_base_saves_and_loads_subfolder_components(tmp_path: Path) -> None:
    pipeline = ToyPipeline(
        config=ToyConfig(label="button"),
        model=ToyModel("stage"),
        processor=ToyProcessor("processor"),
    )

    pipeline.save_pretrained(tmp_path, is_main_process=False)
    loaded = ToyPipeline.from_pretrained(tmp_path, local_files_only=True)

    assert pipeline.model.saved_is_main_process is False
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model" / "component.txt").read_text() == "stage"
    assert (tmp_path / "processor.txt").read_text() == "processor"
    assert loaded.config.label == "button"
    assert loaded.model.name == "stage"
    assert loaded.processor.name == "processor"
    assert loaded.optional is None
    assert loaded().id2label == {0: "button"}


def test_pipeline_base_accepts_preloaded_components(tmp_path: Path) -> None:
    ToyConfig(label="text").save_pretrained(tmp_path)
    ToyProcessor("disk").save_pretrained(tmp_path)
    provided = ToyModel("provided")

    loaded = ToyPipeline.from_pretrained(
        tmp_path,
        components={"model": provided},
        local_files_only=True,
    )

    assert loaded.model is provided
    assert loaded.processor.name == "disk"


def test_pipeline_base_requires_missing_required_component(tmp_path: Path) -> None:
    ToyConfig().save_pretrained(tmp_path)

    with pytest.raises(FileNotFoundError, match="Required pipeline component"):
        ToyPipeline.from_pretrained(tmp_path, local_files_only=True)


def test_pipeline_base_delegates_hub_id_loading_without_marker_check() -> None:
    class HubToyPipeline(ToyPipeline):
        component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
            "model": PipelineComponentSpec(
                attribute_name="model",
                loader=record_loader,
                subfolder="model",
                marker_file="component.txt",
            ),
            "processor": PipelineComponentSpec(
                attribute_name="processor",
                loader=record_loader,
                config_subfolder_attribute="processor_subfolder",
                marker_file="processor.txt",
                save_with_is_main_process=False,
            ),
            "optional": PipelineComponentSpec(
                attribute_name="optional",
                required=False,
            ),
        }

    LOADER_CALLS.clear()
    config = ToyConfig(label="hub")
    config.processor_subfolder = "processor"  # type: ignore[attr-defined]

    loaded = HubToyPipeline.from_pretrained(
        "creative-graphic-design/toy-layout",
        config=config,
        local_files_only=True,
    )

    assert loaded.model.name == "creative-graphic-design/toy-layout"
    assert LOADER_CALLS == [
        ("creative-graphic-design/toy-layout", "model", True),
        ("creative-graphic-design/toy-layout", "processor", True),
    ]


def test_pipeline_base_device_dtype_and_generator_seed_precedence() -> None:
    pipeline = ToyPipeline(
        config=ToyConfig(),
        model=ToyModel("stage"),
        processor=ToyProcessor("processor"),
    )

    moved = pipeline.to("cpu", dtype=torch.float32)
    explicit_generator = torch.Generator().manual_seed(7)
    with patch("laygen.pipelines.base.set_seed") as mocked_seed:
        resolved = pipeline.prepare_generator(
            generator=explicit_generator,
            seed=123,
        )

    assert moved is pipeline
    assert pipeline.model.moved_device == torch.device("cpu")
    assert pipeline.model.moved_dtype is torch.float32
    assert resolved is explicit_generator
    mocked_seed.assert_not_called()

    with patch("laygen.pipelines.base.set_seed") as mocked_seed:
        seeded = pipeline.prepare_generator(seed=123)

    mocked_seed.assert_called_once_with(123)
    assert seeded is not None
    assert seeded.initial_seed() == 123


def test_pipeline_base_is_abstract() -> None:
    assert LayoutGenerationPipeline.__call__.__doc__

    with pytest.raises(TypeError):
        LayoutGenerationPipeline(PretrainedConfig())
