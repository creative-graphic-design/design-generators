"""Configuration for converted LayoutDiffusion checkpoints."""

from __future__ import annotations

from diffusers.configuration_utils import ConfigMixin, register_to_config

from laygen.common.labels import DatasetName, normalize_dataset_name

from .labels import default_id2label


class LayoutDiffusionConfig(ConfigMixin):
    """Serializable LayoutDiffusion model, tokenizer, and scheduler settings.

    Args:
        dataset_name: Dataset name or alias.
        id2label: Optional persisted dataset-local label mapping.
        vocab: Optional token-to-id vocabulary loaded from ``vocab.json``.
        seq_length: Full vendor token sequence length.
        max_num_elements: Maximum number of layout elements.
        num_coordinate_bins: Number of coordinate tokens.
        diffusion_steps: Number of training diffusion timesteps.
        noise_schedule: Vendor diffusion schedule name.
        num_channels: OpenAI timestep embedding dimension.
        bert_config_name: Name of the BERT config used by the vendor.
        hidden_size: Transformer hidden size.
        num_hidden_layers: BERT encoder layer count.
        num_attention_heads: Attention head count.
        intermediate_size: Feed-forward hidden size.
        dropout: Dropout probability.
        training_mode: Vendor training mode.
        vocab_size: Full vocabulary size including mask.
        refine_start_step: Dataset-specific refinement start step.
        type_start_step: Vendor type-conditioned start step.
        element_count_prior: Optional 20-entry unconditional element count prior.
        pow_num: Gaussian transition exponent.
        mul_num: Gaussian transition multiplier.

    Examples:
        >>> cfg = LayoutDiffusionConfig(dataset_name="publaynet")
        >>> cfg.mask_token_id == cfg.vocab_size - 1
        True
    """

    config_name = "layoutdiffusion_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: DatasetName | str = DatasetName.rico25,
        id2label: dict[int | str, str] | None = None,
        vocab: dict[str, int] | None = None,
        seq_length: int = 121,
        max_num_elements: int = 20,
        num_coordinate_bins: int = 128,
        diffusion_steps: int = 200,
        noise_schedule: str = "gaussian_refine_pow2.5",
        num_channels: int = 128,
        bert_config_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        training_mode: str = "discrete",
        vocab_size: int | None = None,
        refine_start_step: int | None = None,
        type_start_step: int = 160,
        element_count_prior: list[float] | None = None,
        pow_num: float = 2.5,
        mul_num: float = 12.4,
    ) -> None:
        """Initialize LayoutDiffusion configuration."""
        self.dataset_name = str(normalize_dataset_name(dataset_name))
        raw_id2label = id2label or default_id2label(self.dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.vocab = vocab or self.default_vocab()
        self.seq_length = seq_length
        self.max_num_elements = max_num_elements
        self.num_coordinate_bins = num_coordinate_bins
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.num_channels = num_channels
        self.bert_config_name = bert_config_name
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.training_mode = training_mode
        self.vocab_size = vocab_size or len(self.vocab)
        self.refine_start_step = refine_start_step or (
            60 if self.dataset_name == str(DatasetName.publaynet) else 50
        )
        self.type_start_step = type_start_step
        self.element_count_prior = element_count_prior or self.default_element_prior()
        self.pow_num = pow_num
        self.mul_num = mul_num

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return LayoutDiffusion special-token ids."""
        return {
            "START": self.vocab["START"],
            "END": self.vocab["END"],
            "UNK": self.vocab["UNK"],
            "PAD": self.vocab["PAD"],
            "|": self.vocab["|"],
        }

    @property
    def pad_token_id(self) -> int:
        """Return the padding token id."""
        return self.vocab["PAD"]

    @property
    def mask_token_id(self) -> int:
        """Return the mask token id."""
        return self.vocab_size - 1

    @property
    def label_token_offset(self) -> int:
        """Return the first label-token id."""
        return 5

    @property
    def num_labels(self) -> int:
        """Return the dataset label count."""
        return len(self.id2label)

    @property
    def coordinate_token_offset(self) -> int:
        """Return the first coordinate-token id."""
        return self.label_token_offset + self.num_labels

    @property
    def max_token_length(self) -> int:
        """Return the full LayoutDiffusion token sequence length."""
        return self.seq_length

    @property
    def label2id(self) -> dict[str, int]:
        """Return inverse public label mapping."""
        return {v: k for k, v in self.id2label.items()}

    @property
    def type_classes(self) -> int:
        """Return the number of vendor type classes."""
        return self.vocab_size - 1 - self.num_coordinate_bins - 5

    def default_vocab(self) -> dict[str, int]:
        """Build the default LayoutDiffusion vocabulary for the dataset."""
        vocab = {"START": 0, "END": 1, "UNK": 2, "PAD": 3, "|": 4}
        for label in default_id2label(self.dataset_name).values():
            vocab[label] = len(vocab)
        for coord in range(self.num_coordinate_bins):
            vocab[str(coord)] = len(vocab)
        vocab["MASK"] = len(vocab)
        return vocab

    def default_element_prior(self) -> list[float]:
        """Return the vendor unconditional element-count prior."""
        if self.dataset_name == str(DatasetName.publaynet):
            return [
                0.00321776,
                0.03342678,
                0.04233181,
                0.04218409,
                0.05404355,
                0.07231605,
                0.08247029,
                0.0905211,
                0.0949399,
                0.0959322,
                0.08953522,
                0.07810608,
                0.0619627,
                0.04775897,
                0.03585776,
                0.0261788,
                0.018812,
                0.01404317,
                0.00972071,
                0.00664104,
            ]
        return [
            0.04849498,
            0.03704171,
            0.0534486,
            0.06045308,
            0.06354515,
            0.07585032,
            0.08045687,
            0.0644917,
            0.05676153,
            0.05742412,
            0.05471067,
            0.04944153,
            0.04552912,
            0.04190068,
            0.04426705,
            0.0387455,
            0.03533792,
            0.03167792,
            0.02997413,
            0.0304474,
        ]
