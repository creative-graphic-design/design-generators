"""Shared base class for Transformers-side layout-generation pipelines.

`transformers.Pipeline` is optimized for registered single-model tasks using a
`preprocess` -> `_forward` -> `postprocess` contract. Layout-generation
packages in this workspace often compose processors, tokenizers, and multiple
standard Transformers models, while still loading from a single checkpoint root
with subfolders. `LayoutGenerationPipeline` is the Transformers-side analogue of
Diffusers' pipeline role: it owns root config metadata, component subfolder
loading and saving, device and dtype movement, and seed/generator precedence.
Subclasses keep the model-specific orchestration in `__call__`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, Self, cast, runtime_checkable

import torch
from transformers import PretrainedConfig, set_seed

from laygen.modeling_outputs import LayoutGenerationOutput


class PipelineComponentLoader(Protocol):
    """Callable that loads one pipeline component from a checkpoint path."""

    def __call__(
        self,
        pretrained_model_name_or_path: str | Path,
        *,
        local_files_only: bool = False,
        subfolder: str | None = None,
    ) -> object:
        """Load a component.

        Args:
            pretrained_model_name_or_path: Root checkpoint path or Hub repo id.
            local_files_only: Whether to avoid network access.
            subfolder: Optional component subfolder for Hub-backed loading.

        Returns:
            Loaded component object.
        """


@runtime_checkable
class SavePretrainedWithMainProcess(Protocol):
    """Component protocol for model-like `save_pretrained` methods."""

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
    ) -> object:
        """Save a component and accept the common main-process flag.

        Args:
            save_directory: Directory to write.
            is_main_process: Whether this process should perform main writes.

        Returns:
            Component-specific save result.
        """


@runtime_checkable
class SavePretrainedPlain(Protocol):
    """Component protocol for processor-like `save_pretrained` methods."""

    def save_pretrained(self, save_directory: str | Path) -> object:
        """Save a component.

        Args:
            save_directory: Directory to write.

        Returns:
            Component-specific save result.
        """


@runtime_checkable
class TorchMovable(Protocol):
    """Component protocol for objects that can move device and dtype."""

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> object:
        """Move a component.

        Args:
            device: Target torch device.
            dtype: Target torch dtype.

        Returns:
            Component-specific move result.
        """


@dataclass(frozen=True)
class PipelineComponentSpec:
    """Declarative loading and saving rule for one pipeline component.

    Args:
        attribute_name: Attribute on the pipeline instance.
        loader: Loader callable used by `from_pretrained`.
        subfolder: Fixed subfolder under the checkpoint root. If omitted, the
            root itself is used.
        config_subfolder_attribute: Config attribute that stores the subfolder
            name. This takes precedence over `subfolder`.
        required: Whether missing marker files or missing instance attributes
            are errors.
        marker_file: File used to detect whether an optional component exists.
            Set to `None` to always call the loader.
        save_with_is_main_process: Whether to pass `is_main_process` to the
            component's `save_pretrained` method.
    """

    attribute_name: str
    loader: PipelineComponentLoader | None = None
    subfolder: str | None = None
    config_subfolder_attribute: str | None = None
    required: bool = True
    marker_file: str | None = "config.json"
    save_with_is_main_process: bool = True

    def component_path(
        self,
        root: Path,
        config: PretrainedConfig,
    ) -> Path:
        """Resolve the component path under a checkpoint root.

        Args:
            root: Checkpoint root directory.
            config: Root pipeline config.

        Returns:
            Resolved root or subfolder path.

        Raises:
            TypeError: If the configured subfolder attribute is not a string.
        """
        if self.config_subfolder_attribute is not None:
            value = getattr(config, self.config_subfolder_attribute)
            if not isinstance(value, str):
                raise TypeError(
                    f"{self.config_subfolder_attribute} must be a string subfolder"
                )
            return root / value
        if self.subfolder is not None:
            return root / self.subfolder
        return root

    def component_subfolder(self, config: PretrainedConfig) -> str | None:
        """Resolve the component subfolder for Hub-backed loading.

        Args:
            config: Root pipeline config.

        Returns:
            Component subfolder or `None` when the component lives at the root.

        Raises:
            TypeError: If the configured subfolder attribute is not a string.
        """
        if self.config_subfolder_attribute is not None:
            value = getattr(config, self.config_subfolder_attribute)
            if not isinstance(value, str):
                raise TypeError(
                    f"{self.config_subfolder_attribute} must be a string subfolder"
                )
            return value
        return self.subfolder


class LayoutGenerationPipeline(ABC):
    """Base class for Transformers-side layout-generation pipelines.

    Subclasses declare checkpoint components with `component_specs`, implement
    `_from_pretrained_components`, and put generation orchestration in
    `__call__`. The public `__call__` contract is to return
    `laygen.modeling_outputs.LayoutGenerationOutput` for layout-generation
    outputs.

    Args:
        config: Root pipeline config, usually a `PretrainedConfig` subclass.

    Examples:
        >>> from transformers import PretrainedConfig
        >>> class ToyPipeline(LayoutGenerationPipeline):
        ...     config_class = PretrainedConfig
        ...     @classmethod
        ...     def _from_pretrained_components(cls, *, config, components):
        ...         return cls(config)
        ...     def __call__(self):
        ...         import torch
        ...         return LayoutGenerationOutput(
        ...             bbox=torch.zeros(1, 1, 4),
        ...             labels=torch.zeros(1, 1, dtype=torch.long),
        ...             mask=torch.ones(1, 1, dtype=torch.bool),
        ...             id2label={0: "text"},
        ...         )
        >>> isinstance(ToyPipeline(PretrainedConfig()).config, PretrainedConfig)
        True
    """

    config_class: ClassVar[type[PretrainedConfig]] = PretrainedConfig
    component_specs: ClassVar[Mapping[str, PipelineComponentSpec]] = {}

    config: PretrainedConfig
    device: torch.device | None
    dtype: torch.dtype | None

    def __init__(self, config: PretrainedConfig) -> None:
        """Initialize root config and runtime placement metadata.

        Args:
            config: Root pipeline config.
        """
        self.config = config
        self.device = None
        self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        local_files_only: bool = False,
        config: PretrainedConfig | None = None,
        components: Mapping[str, object] | None = None,
    ) -> Self:
        """Load a pipeline from a checkpoint root and declared subfolders.

        Args:
            pretrained_model_name_or_path: Checkpoint root.
            local_files_only: Whether to avoid network access.
            config: Optional preloaded root config.
            components: Optional preloaded components keyed by spec name.

        Returns:
            Loaded pipeline instance.

        Raises:
            FileNotFoundError: If a required component marker file is missing.
            TypeError: If `config` is not compatible with `config_class`.
        """
        root = Path(pretrained_model_name_or_path)
        source = pretrained_model_name_or_path
        pipeline_config = cls._load_pipeline_config(
            source,
            local_files_only=local_files_only,
            config=config,
        )
        loaded_components = cls._load_pipeline_components(
            root,
            source,
            pipeline_config,
            local_files_only=local_files_only,
            components=components or {},
        )
        return cls._from_pretrained_components(
            config=pipeline_config,
            components=loaded_components,
        )

    @classmethod
    def _load_pipeline_config(
        cls,
        source: str | Path,
        *,
        local_files_only: bool,
        config: PretrainedConfig | None,
    ) -> PretrainedConfig:
        if config is None:
            return cls.config_class.from_pretrained(
                source,
                local_files_only=local_files_only,
            )
        if isinstance(config, cls.config_class):
            return config
        if isinstance(config, PretrainedConfig):
            return cls.config_class.from_dict(config.to_dict())
        raise TypeError(f"config must be a {cls.config_class.__name__}")

    @classmethod
    def _load_pipeline_components(
        cls,
        root: Path,
        source: str | Path,
        config: PretrainedConfig,
        *,
        local_files_only: bool,
        components: Mapping[str, object],
    ) -> dict[str, object | None]:
        loaded: dict[str, object | None] = {}
        for name, spec in cls.component_specs.items():
            if name in components:
                loaded[name] = components[name]
                continue
            if spec.loader is None:
                loaded[name] = None
                continue
            if root.is_dir():
                component_path = spec.component_path(root, config)
                marker = (
                    component_path / spec.marker_file
                    if spec.marker_file is not None
                    else None
                )
                if marker is not None and not marker.exists():
                    if spec.required:
                        raise FileNotFoundError(
                            f"Required pipeline component '{name}' is missing: {marker}"
                        )
                    loaded[name] = None
                    continue
                loaded[name] = spec.loader(
                    component_path,
                    local_files_only=local_files_only,
                )
                continue
            loaded[name] = spec.loader(
                source,
                local_files_only=local_files_only,
                subfolder=spec.component_subfolder(config),
            )
        return loaded

    @classmethod
    @abstractmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> Self:
        """Build a pipeline from a root config and loaded components.

        Args:
            config: Loaded root config.
            components: Components keyed by `component_specs` names.

        Returns:
            Loaded pipeline instance.
        """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
    ) -> None:
        """Save root config and declared components.

        Args:
            save_directory: Checkpoint root directory.
            is_main_process: Whether model-like components should perform main
                process writes.

        Raises:
            TypeError: If a component does not implement `save_pretrained`.
            ValueError: If a required component attribute is missing.
        """
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(root)
        for name, spec in self.component_specs.items():
            component = getattr(self, spec.attribute_name, None)
            if component is None:
                if spec.required:
                    raise ValueError(f"Required pipeline component '{name}' is None")
                continue
            component_path = spec.component_path(root, self.config)
            component_path.mkdir(parents=True, exist_ok=True)
            if spec.save_with_is_main_process:
                if not isinstance(component, SavePretrainedWithMainProcess):
                    raise TypeError(
                        f"Pipeline component '{name}' does not support save_pretrained"
                    )
                component.save_pretrained(
                    component_path,
                    is_main_process=is_main_process,
                )
            else:
                if not isinstance(component, SavePretrainedPlain):
                    raise TypeError(
                        f"Pipeline component '{name}' does not support save_pretrained"
                    )
                component.save_pretrained(component_path)

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move movable components to a device and/or dtype.

        Args:
            device: Target torch device.
            dtype: Target torch dtype.

        Returns:
            This pipeline instance.
        """
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if self.device is None and self.dtype is None:
            return self
        for component in self._pipeline_component_values():
            if isinstance(component, TorchMovable):
                component.to(device=self.device, dtype=self.dtype)
        return self

    def prepare_generator(
        self,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Generator | None:
        """Apply generator-over-seed precedence for generation calls.

        Args:
            generator: Explicit torch generator. When provided, `seed` is
                ignored.
            seed: Integer seed used only when `generator` is absent.
            device: Optional device for a newly created generator. If omitted,
                the pipeline's current device is used.

        Returns:
            The explicit generator, a seeded generator when a device is known,
            or `None` after setting global Transformers/PyTorch seed state.
        """
        if generator is not None:
            return generator
        if seed is None:
            return None
        set_seed(seed)
        generator_device = torch.device(device) if device is not None else self.device
        if generator_device is None:
            return None
        return torch.Generator(device=generator_device).manual_seed(seed)

    def _pipeline_component_values(self) -> tuple[object, ...]:
        values: list[object] = []
        for spec in self.component_specs.values():
            component = getattr(self, spec.attribute_name, None)
            if component is not None:
                values.append(cast(object, component))
        return tuple(values)

    @abstractmethod
    def __call__(self) -> LayoutGenerationOutput:
        """Generate a layout.

        Returns:
            Layout generation output in the canonical Transformers-style schema.
        """


__all__ = [
    "LayoutGenerationPipeline",
    "PipelineComponentLoader",
    "PipelineComponentSpec",
]
