"""Configuration helpers for converted LayouSyn checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

from diffusers import ConfigMixin
from diffusers.configuration_utils import register_to_config

LayoutType = Literal["xyxy", "cxcywh"]


class LayouSynModelShape(TypedDict):
    """Resolved DiT architecture shape."""

    hidden_size: int
    depth: int
    num_heads: int


_MODEL_SHAPES: Final[dict[str, LayouSynModelShape]] = {
    "DiT-XS": {"hidden_size": 192, "depth": 6, "num_heads": 6},
    "DiT-S": {"hidden_size": 256, "depth": 8, "num_heads": 8},
    "DiT-B": {"hidden_size": 384, "depth": 12, "num_heads": 12},
    "DiT-L": {"hidden_size": 576, "depth": 18, "num_heads": 18},
    "DiT-XL": {"hidden_size": 768, "depth": 24, "num_heads": 24},
}
_DIT_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^DiT-D(?P<depth>\d+)-H(?P<hidden_size>\d+)-N(?P<num_heads>\d+)$"
)


def resolve_model_shape(
    model_name: str,
    *,
    hidden_size: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
) -> LayouSynModelShape:
    """Resolve a reference DiT name to concrete architecture dimensions.

    Args:
        model_name: Reference model key such as ``DiT-S`` or
            ``DiT-D28-H1152-N16``.
        hidden_size: Optional explicit override.
        depth: Optional explicit override.
        num_heads: Optional explicit override.

    Returns:
        Resolved architecture dimensions.

    Raises:
        ValueError: If the model name is unsupported.
    """
    if model_name in _MODEL_SHAPES:
        shape = dict(_MODEL_SHAPES[model_name])
    else:
        match = _DIT_REGEX.match(model_name)
        if match is None:
            raise ValueError(f"Unsupported LayouSyn model_name: {model_name}")
        shape = {
            "hidden_size": int(match.group("hidden_size")),
            "depth": int(match.group("depth")),
            "num_heads": int(match.group("num_heads")),
        }
    if hidden_size is not None:
        shape["hidden_size"] = hidden_size
    if depth is not None:
        shape["depth"] = depth
    if num_heads is not None:
        shape["num_heads"] = num_heads
    return cast(LayouSynModelShape, shape)


class LayouSynConfig(ConfigMixin):
    """Serializable LayouSyn configuration.

    Args:
        model_name: Reference DiT architecture key.
        in_channels: Layout coordinate channels.
        concept_in_channels: Concept embedding width.
        y_in_channels: Caption embedding width.
        max_in_len: Maximum number of object slots.
        max_y_len: Maximum number of caption tokens.
        layout_type: Reference layout coordinate type.
        t5_size: Reference T5 size suffix.
        scale: Default classifier-free guidance scale.
        noise_schedule: Reference diffusion beta schedule.
        diffusion_steps: Number of diffusion training timesteps.
        hidden_size: Optional resolved hidden width override.
        depth: Optional resolved transformer depth override.
        num_heads: Optional resolved attention head override.
        license: Upstream checkpoint license identifier.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        *,
        model_name: str = "DiT-S",
        in_channels: int = 4,
        concept_in_channels: int = 768,
        y_in_channels: int | None = 768,
        max_in_len: int = 60,
        max_y_len: int | None = 120,
        layout_type: LayoutType = "xyxy",
        t5_size: str | None = "base",
        scale: float = 2.0,
        noise_schedule: str = "linear",
        diffusion_steps: int = 100,
        hidden_size: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        license: str = "cc-by-nc-4.0",
    ) -> None:
        """Initialize configuration fields."""
        shape = resolve_model_shape(
            model_name,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
        )
        self.model_name = model_name
        self.in_channels = in_channels
        self.concept_in_channels = concept_in_channels
        self.y_in_channels = y_in_channels
        self.max_in_len = max_in_len
        self.max_y_len = max_y_len
        self.layout_type = layout_type
        self.t5_size = t5_size
        self.scale = scale
        self.noise_schedule = noise_schedule
        self.diffusion_steps = diffusion_steps
        self.hidden_size = shape["hidden_size"]
        self.depth = shape["depth"]
        self.num_heads = shape["num_heads"]
        self.license = license

    @classmethod
    def from_reference_json(cls, path: str | Path) -> "LayouSynConfig":
        """Load a reference JSON config.

        Args:
            path: Path to a Lay-Your-Scene JSON config.

        Returns:
            Converted configuration object.
        """
        data = json.loads(Path(path).read_text())
        layout_type = data.get("layout_type", "xyxy")
        if not isinstance(layout_type, str):
            layout_type = str(layout_type)
        return cls(
            model_name=data.get("model", "DiT-S"),
            in_channels=data.get("in_channel", 4),
            concept_in_channels=data.get("concept_in_channel", 768),
            y_in_channels=data.get("y_in_channel"),
            max_in_len=data.get("max_in_len", 60),
            max_y_len=data.get("max_y_len"),
            layout_type="cxcywh" if "cxcywh" in layout_type.lower() else "xyxy",
            t5_size=data.get("t5_size"),
            scale=data.get("scale", 1.0),
            noise_schedule=data.get("noise_schedule", "linear"),
            diffusion_steps=data.get("diffusion_steps", 1000),
        )

    def to_reference_dict(self) -> dict[str, object]:
        """Return the config keys expected by the original repository."""
        return {
            "model": self.model_name,
            "in_channel": self.in_channels,
            "concept_in_channel": self.concept_in_channels,
            "y_in_channel": self.y_in_channels,
            "max_in_len": self.max_in_len,
            "max_y_len": self.max_y_len,
            "scale": self.scale,
            "noise_schedule": self.noise_schedule,
            "layout_type": self.layout_type,
            "diffusion_steps": self.diffusion_steps,
            "t5_size": self.t5_size,
        }
