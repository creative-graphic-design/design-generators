"""Runtime adapter interfaces for converted PosterLlama components."""

from __future__ import annotations

from collections.abc import Sequence
import json
from os import PathLike
from pathlib import Path

import torch
from jaxtyping import Float


class PosterLlamaRuntime(torch.nn.Module):
    """Minimal runtime interface used by ``PosterLlamaPipeline``.

    The full MiniGPT/DINO/CodeLLaMA stack is created by local conversion tools.
    This lightweight adapter keeps the public package importable in ordinary CI
    and gives tests a serializable runtime shape.

    Args:
        generated_text: Optional deterministic text emitted by this runtime.

    Examples:
        >>> runtime = PosterLlamaRuntime('<svg width="1" height="1"></svg>')
        >>> runtime.generate_texts(["prompt"])
        ['<svg width="1" height="1"></svg>']
    """

    def __init__(self, generated_text: str | None = None) -> None:
        """Initialize deterministic runtime metadata."""
        super().__init__()
        self.generated_text = generated_text

    def generate_texts(
        self,
        prompts: Sequence[str],
        *,
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        generator: torch.Generator | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int | None = None,
        num_beams: int = 1,
    ) -> list[str]:
        """Generate markup text for prompts.

        Args:
            prompts: Prompt strings.
            pixel_values: Optional image tensors.
            generator: Optional PyTorch generator.
            max_new_tokens: Generation length budget.
            do_sample: Sampling flag.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            top_k: Top-k sampling value.
            num_beams: Beam count.

        Returns:
            Generated markup strings.

        Raises:
            RuntimeError: If no converted runtime text generator is available.
        """
        _ = (
            pixel_values,
            generator,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            num_beams,
        )
        if self.generated_text is None:
            raise RuntimeError(
                "PosterLlama runtime assets are missing. Run the local conversion "
                "script with the raw checkpoint and backbone paths before inference."
            )
        return [self.generated_text for _ in prompts]

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        *,
        is_main_process: bool = True,
    ) -> None:
        """Save runtime metadata.

        Args:
            save_directory: Directory to write.
            is_main_process: Whether this process should write files.
        """
        if not is_main_process:
            return
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "runtime_config.json").write_text(
            json.dumps({"generated_text": self.generated_text}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        local_files_only: bool = False,
        subfolder: str | None = None,
    ) -> "PosterLlamaRuntime":
        """Load runtime metadata.

        Args:
            pretrained_model_name_or_path: Runtime directory.
            local_files_only: Accepted for loader compatibility.
            subfolder: Optional subfolder.

        Returns:
            PosterLlamaRuntime instance.
        """
        _ = local_files_only
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        data = json.loads((root / "runtime_config.json").read_text(encoding="utf-8"))
        return cls(generated_text=data.get("generated_text"))
