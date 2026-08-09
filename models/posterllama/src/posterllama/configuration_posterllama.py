"""Configuration for PosterLlama inference recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from transformers import PretrainedConfig

from posgen.common.labels import id2label_for_dataset

PosterLlamaDatasetName = Literal["cgl", "cgl_v2", "pku_posterlayout"]
PosterLlamaVisionModelName = Literal["dino_v2", "eva_clip_g"]
PosterLlamaLicenseStatus = Literal["unverified"]
PosterLlamaConfigValue = (
    str | int | float | bool | None | list[str] | list[int] | dict[str, str]
)


class PosterLlamaConfig(PretrainedConfig):
    """Configuration for local PosterLlama recipe artifacts.

    Args:
        checkpoint_repo_id: Source Hub repository containing the raw checkpoint.
        base_llm_repo_id: Preferred CodeLLaMA/LLaMA backbone repository id.
        alternate_base_llm_repo_ids: Alternate backbone ids recorded for audit.
        vision_encoder_repo_id: Vision encoder repository id.
        vision_model_name: Original vision tower selector.
        lora_r: LoRA rank used by the released recipe.
        lora_alpha: LoRA alpha used by the released recipe.
        lora_dropout: LoRA dropout used by the released recipe.
        lora_target_modules: LLM projection module names targeted by LoRA.
        prompt_template: Wrapper applied around generated layout prompts.
        image_placeholder: Placeholder token used for image feature insertion.
        image_end_token: End marker for image features.
        max_txt_len: Original maximum text length.
        max_context_len: Original context length budget.
        default_max_new_tokens: Default generation budget.
        default_do_sample: Default sampled-generation flag.
        default_temperature: Default generation temperature.
        default_top_p: Default nucleus sampling value.
        default_top_k: Default top-k sampling value.
        default_num_beams: Default beam count.
        dataset_name: Poster dataset key.
        id2label: Dataset-local label vocabulary.
        canvas_size: Optional default canvas size as ``(width, height)``.
        checkpoint_license_status: Redistribution status for converted weights.
        processor_subfolder: Pipeline processor subfolder.
        runtime_subfolder: Optional converted runtime subfolder.
        kwargs: Extra ``PretrainedConfig`` keyword arguments.

    Examples:
        >>> cfg = PosterLlamaConfig(canvas_size=(360, 504))
        >>> cfg.id2label[1]
        'text'
    """

    model_type = "posterllama"

    def __init__(
        self,
        checkpoint_repo_id: str = "poong/PosterLlama",
        base_llm_repo_id: str = "codellama/CodeLlama-7b-hf",
        alternate_base_llm_repo_ids: Sequence[str] = ("meta-llama/Llama-2-7b-chat-hf",),
        vision_encoder_repo_id: str = "facebook/dinov2-base",
        vision_model_name: PosterLlamaVisionModelName = "dino_v2",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        prompt_template: str = "{}",
        image_placeholder: str = "<ImageHere>",
        image_end_token: str = "</Img>",
        max_txt_len: int = 400,
        max_context_len: int = 3800,
        default_max_new_tokens: int = 1024,
        default_do_sample: bool = True,
        default_temperature: float = 0.6,
        default_top_p: float = 0.9,
        default_top_k: int = 40,
        default_num_beams: int = 4,
        dataset_name: PosterLlamaDatasetName = "cgl",
        id2label: Mapping[int | str, str] | None = None,
        canvas_size: tuple[int, int] | list[int] | None = None,
        checkpoint_license_status: PosterLlamaLicenseStatus = "unverified",
        processor_subfolder: str = "processor",
        runtime_subfolder: str = "runtime",
        **kwargs: PosterLlamaConfigValue,
    ) -> None:
        """Initialize configuration values."""
        labels = (
            id2label_for_dataset(dataset_name)
            if id2label is None
            else {int(key): str(value) for key, value in id2label.items()}
        )
        kwargs.pop("model_type", None)
        kwargs.pop("id2label", None)
        kwargs.pop("label2id", None)
        super().__init__(
            id2label=labels,
            label2id={label: idx for idx, label in labels.items()},
        )
        self.checkpoint_repo_id = checkpoint_repo_id
        self.base_llm_repo_id = base_llm_repo_id
        self.alternate_base_llm_repo_ids = list(alternate_base_llm_repo_ids)
        self.vision_encoder_repo_id = vision_encoder_repo_id
        self.vision_model_name = vision_model_name
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = list(lora_target_modules)
        self.prompt_template = prompt_template
        self.image_placeholder = image_placeholder
        self.image_end_token = image_end_token
        self.max_txt_len = int(max_txt_len)
        self.max_context_len = int(max_context_len)
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.default_do_sample = bool(default_do_sample)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_top_k = int(default_top_k)
        self.default_num_beams = int(default_num_beams)
        self.dataset_name = dataset_name
        self.canvas_size = tuple(canvas_size) if canvas_size is not None else None
        self.checkpoint_license_status = checkpoint_license_status
        self.processor_subfolder = processor_subfolder
        self.runtime_subfolder = runtime_subfolder
        for key, value in kwargs.items():
            setattr(self, key, value)
