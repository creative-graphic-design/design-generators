from __future__ import annotations

from typing import Literal

from diffusers.configuration_utils import ConfigMixin, register_to_config

from layout_generation_common.labels import id2label_for_dataset, normalize_dataset_name


class LayoutCorrectorConfig(ConfigMixin):
    config_name = "corrector_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str,
        vocab_size: int,
        id2label: dict[int | str, str] | None = None,
        max_seq_length: int = 25,
        num_attributes_per_element: int = 5,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: str | None = "adalayernorm",
        num_timesteps: int = 100,
        recon_type: Literal["x_0", "x_t-1"] = "x_t-1",
        target: Literal["mask", "recon_acc"] = "recon_acc",
        attr_loss_weights: tuple[float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        use_padding_as_vocab: bool = True,
        pos_emb: Literal[
            "default", "none", "pos_enc", "shuffle_pos_enc", "shuffle"
        ] = "none",
        transformer_type: Literal["aggregated"] = "aggregated",
        corrector_steps: int = 1,
        corrector_t_list: tuple[int, ...] = (10, 20, 30),
        corrector_mask_mode: Literal["thresh", "topk"] = "thresh",
        corrector_mask_threshold: float = 0.7,
        corrector_temperature: float = 1.0,
        use_gumbel_noise: bool = True,
        gumbel_temperature: float = 1.0,
        time_adaptive_temperature: bool = False,
    ) -> None:
        dataset_name = normalize_dataset_name(dataset_name)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if num_attributes_per_element != 5:
            raise ValueError("Layout-Corrector supports 5 attributes per element")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if recon_type not in {"x_0", "x_t-1"}:
            raise ValueError(f"Unsupported recon_type: {recon_type}")
        if target not in {"mask", "recon_acc"}:
            raise ValueError(f"Unsupported target: {target}")
        if transformer_type != "aggregated":
            raise ValueError("Only transformer_type='aggregated' is supported")
        if pos_emb not in {"default", "none", "pos_enc", "shuffle_pos_enc", "shuffle"}:
            raise ValueError(f"Unsupported pos_emb: {pos_emb}")
        if len(attr_loss_weights) != num_attributes_per_element:
            raise ValueError("attr_loss_weights must match num_attributes_per_element")
        if corrector_steps <= 0:
            raise ValueError("corrector_steps must be positive")
        if corrector_mask_mode not in {"thresh", "topk"}:
            raise ValueError(f"Unsupported corrector_mask_mode: {corrector_mask_mode}")
        if not 0.0 <= corrector_mask_threshold <= 1.0:
            raise ValueError("corrector_mask_threshold must be in [0, 1]")

        self.dataset_name = dataset_name
        raw_id2label = id2label or id2label_for_dataset(dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_attributes_per_element = num_attributes_per_element
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.timestep_type = timestep_type
        self.num_timesteps = num_timesteps
        self.recon_type = recon_type
        self.target = target
        self.attr_loss_weights = tuple(float(v) for v in attr_loss_weights)
        self.use_padding_as_vocab = use_padding_as_vocab
        self.pos_emb = pos_emb
        self.transformer_type = transformer_type
        self.corrector_steps = corrector_steps
        self.corrector_t_list = tuple(int(v) for v in corrector_t_list)
        self.corrector_mask_mode = corrector_mask_mode
        self.corrector_mask_threshold = corrector_mask_threshold
        self.corrector_temperature = corrector_temperature
        self.use_gumbel_noise = use_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.time_adaptive_temperature = time_adaptive_temperature

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * self.num_attributes_per_element
