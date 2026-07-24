"""Processor for PosterLLaVA prompts and JSON layout decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, PreTrainedTokenizerBase, ProcessorMixin

from laygen.common.bbox import BoxFormat, prepare_layout_tensors
from laygen.modeling_outputs import LayoutGenerationOutput
from posgen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)

from .configuration_posterllava import (
    DEFAULT_PROMPT_TEMPLATE,
    ConversationMode,
    OutputType,
    normalize_conversation_mode,
    normalize_output_type,
)
from .generation_posterllava import IMAGE_TOKEN, tokenizer_image_token

PROCESSOR_CONFIG_NAME: Final[str] = "processor_config.json"
DEFAULT_CANVAS_SIZE: Final[tuple[int, int]] = (1, 1)
DEFAULT_DOMAIN_NAME: Final[str] = "social media promotion poster with qbposter style"


class PosterLlavaJsonElement(TypedDict):
    """One generated PosterLLaVA JSON element."""

    label: str
    box: list[float]


class ParsedPosterLlavaElement(TypedDict):
    """One parsed element with batch-local numeric label id."""

    label: int
    label_text: str
    bbox_ltrb: list[float]


class PromptBundle(TypedDict):
    """Prompt text and normalized element metadata."""

    prompt: str
    initial_elements: list[PosterLlavaJsonElement]
    num_elements: int


class PosterLlavaProcessor(ProcessorMixin):
    """Build PosterLLaVA prompts and decode generated JSON layouts.

    Args:
        tokenizer: Optional LLaVA tokenizer used to insert the image sentinel.
        image_processor: Optional image processor component.
        dataset_name: Poster/content dataset used for known label metadata.
        canvas_size: Canvas size used when public input boxes are pixel based.
        id2label: Persisted known label map.
        prompt_template: JSON instruction body template.
        default_domain_name: Domain phrase inserted into prompts.

    Examples:
        >>> processor = PosterLlavaProcessor.from_config()
        >>> processor.parse_output("[{'label': 'text', 'box': [0, 0, 1, 1]}]")[0]["label"]
        'text'
    """

    attributes = ["tokenizer", "image_processor"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "AutoImageProcessor"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        image_processor: object | None = None,
        dataset_name: DatasetName | str = DatasetName.ad_banner,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        default_domain_name: str = DEFAULT_DOMAIN_NAME,
    ) -> None:
        """Initialize tokenizer handles and layout metadata."""
        dataset = normalize_dataset_name(dataset_name)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_name = str(dataset)
        self.canvas_size = canvas_size
        self.id2label = {
            int(key): str(value)
            for key, value in (id2label or id2label_for_dataset(dataset)).items()
        }
        self.prompt_template = prompt_template
        self.default_domain_name = default_domain_name

    @classmethod
    def from_config(
        cls,
        *,
        dataset_name: DatasetName | str = DatasetName.ad_banner,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        default_domain_name: str = DEFAULT_DOMAIN_NAME,
    ) -> "PosterLlavaProcessor":
        """Construct a metadata-only processor for tests and local smoke checks.

        Args:
            dataset_name: Poster/content dataset key.
            canvas_size: Canvas size used for pixel input normalization.
            id2label: Optional known label map.
            prompt_template: Prompt body template.
            default_domain_name: Default domain phrase.

        Returns:
            Metadata-only processor.
        """
        return cls(
            tokenizer=None,
            image_processor=None,
            dataset_name=dataset_name,
            canvas_size=canvas_size,
            id2label=id2label,
            prompt_template=prompt_template,
            default_domain_name=default_domain_name,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        *,
        subfolder: str | None = None,
        **kwargs: object,
    ) -> "PosterLlavaProcessor":
        """Load processor metadata from a checkpoint directory.

        Args:
            pretrained_model_name_or_path: Root checkpoint path.
            cache_dir: Accepted for Transformers processor compatibility.
            force_download: Accepted for Transformers processor compatibility.
            local_files_only: Accepted for compatibility with pipeline loaders.
            token: Accepted for Transformers processor compatibility.
            revision: Accepted for Transformers processor compatibility.
            subfolder: Optional processor subfolder.
            kwargs: Metadata overrides.

        Returns:
            Loaded processor.

        Raises:
            FileNotFoundError: If ``processor_config.json`` is absent.
        """
        _ = cache_dir, force_download, local_files_only, token, revision
        root = Path(pretrained_model_name_or_path)
        path = root / subfolder if subfolder is not None else root
        config_path = path / PROCESSOR_CONFIG_NAME
        data = json.loads(config_path.read_text())
        data.update(kwargs)
        canvas_size = cast(list[int], data["canvas_size"])
        if len(canvas_size) != 2:
            raise ValueError("canvas_size must contain width and height")
        return cls.from_config(
            dataset_name=cast(str, data["dataset_name"]),
            canvas_size=(canvas_size[0], canvas_size[1]),
            id2label=cast(dict[str, str], data["id2label"]),
            prompt_template=cast(str, data["prompt_template"]),
            default_domain_name=cast(str, data["default_domain_name"]),
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata and optional component processors.

        Args:
            save_directory: Directory to write.
            push_to_hub: Accepted for Transformers processor compatibility.
            kwargs: Additional save options accepted for compatibility.
        """
        _ = push_to_hub, kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "processor_class": self.__class__.__name__,
            "dataset_name": self.dataset_name,
            "canvas_size": list(self.canvas_size),
            "id2label": {str(key): value for key, value in self.id2label.items()},
            "prompt_template": self.prompt_template,
            "default_domain_name": self.default_domain_name,
        }
        (root / PROCESSOR_CONFIG_NAME).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )

    def build_initial_json(
        self,
        *,
        labels: Sequence[str | int] | Int[torch.Tensor, "elements"] | None = None,
        bbox: Float[torch.Tensor, "elements 4"]
        | Sequence[Sequence[float]]
        | None = None,
        mask: Bool[torch.Tensor, "elements"] | Sequence[bool] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> list[PosterLlavaJsonElement]:
        """Build optional initial layout JSON from public layout inputs.

        Args:
            labels: Known labels as strings or integer ids.
            bbox: Optional boxes aligned with labels.
            mask: Optional valid-element mask.
            box_format: Public box format for ``bbox``.
            normalized: Whether ``bbox`` is already normalized.
            canvas_size: Pixel canvas size when ``normalized=False``.

        Returns:
            Initial JSON elements used in the prompt.

        Raises:
            ValueError: If labels and boxes are inconsistently shaped.
        """
        if labels is None:
            return []
        label_items = (
            [int(item) for item in labels.tolist()]
            if isinstance(labels, torch.Tensor)
            else list(labels)
        )
        if bbox is None:
            return [
                {
                    "label": self.id2label.get(label, str(label))
                    if isinstance(label, int)
                    else str(label),
                    "box": [],
                }
                for label in label_items
            ]
        bbox_t, _, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=torch.arange(len(label_items)),
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size or self.canvas_size,
        )
        if bbox_t.shape[1] != len(label_items):
            raise ValueError("labels and bbox must contain the same element count")
        ltrb = self._xywh_to_ltrb(bbox_t[0])
        elements: list[PosterLlavaJsonElement] = []
        for idx, label in enumerate(label_items):
            if not bool(mask_t[0, idx]):
                continue
            label_text = (
                self.id2label.get(label, str(label))
                if isinstance(label, int)
                else str(label)
            )
            elements.append({"label": label_text, "box": ltrb[idx].tolist()})
        return elements

    def build_prompt(
        self,
        *,
        num_elements: int,
        canvas_size: tuple[int, int] | None = None,
        elements: Sequence[Mapping[str, object]] = (),
        domain_name: str | None = None,
        conv_mode: ConversationMode | str = ConversationMode.llava_v0,
        prompt: str | None = None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> str:
        """Build the LLaVA conversation prompt.

        Args:
            num_elements: Number of requested layout elements.
            canvas_size: Optional canvas metadata. Stored in prompt text only
                when a custom prompt uses it.
            elements: Optional initial layout JSON.
            domain_name: Domain phrase for the default template.
            conv_mode: LLaVA conversation template.
            prompt: Optional user-supplied prompt body override.
            texts: Optional text payload inserted into the default body.

        Returns:
            Full conversation prompt with the ``<image>`` marker.

        Raises:
            ValueError: If ``num_elements`` is not positive.
        """
        if num_elements <= 0:
            raise ValueError("num_elements must be positive")
        element_list = list(elements)
        initial = ""
        if element_list:
            initial = " Initial layout JSON: " + json.dumps(element_list)
        resolution = list(canvas_size or self.canvas_size)
        text_payload = self._format_texts(texts)
        body = prompt or self.prompt_template.format(
            num_elements=num_elements,
            domain_name=domain_name or self.default_domain_name,
            initial_layout=initial,
            initial_json=json.dumps(element_list),
            canvas_size=resolution,
            resolution=resolution,
            texts=text_payload,
        )
        if text_payload and "{texts}" not in self.prompt_template and prompt is None:
            body = f"{body}\nText payload: {text_payload}"
        return self._wrap_conversation(body, conv_mode=conv_mode)

    def __call__(
        self,
        prompt: str | Sequence[str],
        *,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Tokenize prompts with LLaVA image-token insertion.

        Args:
            prompt: Prompt string or prompt sequence.
            return_tensors: Only ``"pt"`` is supported.

        Returns:
            Batch encoding with ``input_ids`` and ``prompt_text``.

        Raises:
            ValueError: If tokenizer is absent.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to encode PosterLLaVA prompts")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        encoded = [
            tokenizer_image_token(item, self.tokenizer, return_tensors=return_tensors)
            for item in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            encoded,
            batch_first=True,
            padding_value=getattr(self.tokenizer, "pad_token_id", 0) or 0,
        )
        attention_mask = input_ids.ne(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_text": prompts,
            }
        )

    def parse_output(self, text: str) -> list[PosterLlavaJsonElement]:
        """Parse the first generated JSON-like array span.

        Args:
            text: Decoded LLaVA generation text.

        Returns:
            Parsed element dictionaries.

        Raises:
            ValueError: If no JSON array span can be parsed.
        """
        span = self._extract_json_array(text)
        raw_items = json.loads(span.replace("'", '"'))
        if not isinstance(raw_items, list):
            raise ValueError("PosterLLaVA output JSON must be a list")
        elements: list[PosterLlavaJsonElement] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise ValueError("PosterLLaVA output elements must be objects")
            label = item.get("label")
            box = item.get("box")
            if not isinstance(label, str):
                raise ValueError("PosterLLaVA output element label must be a string")
            if not isinstance(box, Sequence) or len(box) != 4:
                raise ValueError("PosterLLaVA output element box must have four values")
            elements.append(
                {
                    "label": label,
                    "box": [float(value) for value in box],
                }
            )
        return elements

    def decode_layout(
        self,
        text: str | Sequence[str],
        *,
        output_type: OutputType | Literal["dataclass", "dict"] = OutputType.dataclass,
        return_intermediates: bool = False,
        sequences: object | None = None,
        prompts: Sequence[str] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode generated text into the shared layout output schema.

        Args:
            text: Generated text or batch of generated texts.
            output_type: Output container mode.
            return_intermediates: Whether to include raw text and parser data.
            sequences: Optional generated token ids.
            prompts: Optional prompt texts.

        Returns:
            Layout output dataclass or dictionary.
        """
        texts = [text] if isinstance(text, str) else list(text)
        parsed_batches = [self.parse_output(item) for item in texts]
        label_names = self._batch_label_names(parsed_batches)
        id2label = dict(enumerate(label_names))
        label2id = {label: idx for idx, label in id2label.items()}
        max_len = max((len(item) for item in parsed_batches), default=0) or 1
        bbox_rows: list[Float[torch.Tensor, "elements 4"]] = []
        label_rows: list[Int[torch.Tensor, "elements"]] = []
        mask_rows: list[Bool[torch.Tensor, "elements"]] = []
        parsed_numeric: list[list[ParsedPosterLlavaElement]] = []
        for parsed in parsed_batches:
            numeric: list[ParsedPosterLlavaElement] = [
                {
                    "label": label2id[item["label"]],
                    "label_text": item["label"],
                    "bbox_ltrb": item["box"],
                }
                for item in parsed
            ]
            parsed_numeric.append(numeric)
            boxes = torch.tensor(
                [item["bbox_ltrb"] for item in numeric], dtype=torch.float32
            )
            labels = torch.tensor([item["label"] for item in numeric], dtype=torch.long)
            mask = torch.ones(len(numeric), dtype=torch.bool)
            if len(numeric) == 0:
                boxes = torch.zeros(max_len, 4, dtype=torch.float32)
                labels = torch.zeros(max_len, dtype=torch.long)
                mask = torch.zeros(max_len, dtype=torch.bool)
            elif len(numeric) < max_len:
                pad = max_len - len(numeric)
                boxes = torch.nn.functional.pad(boxes, (0, 0, 0, pad))
                labels = torch.nn.functional.pad(labels, (0, pad))
                mask = torch.nn.functional.pad(mask, (0, pad))
            bbox_rows.append(boxes)
            label_rows.append(labels)
            mask_rows.append(mask)
        raw_ltrb = torch.stack(bbox_rows)
        bbox = self._ltrb_to_xywh(raw_ltrb).clamp(0.0, 1.0)
        intermediates: dict[str, object] | None = None
        if return_intermediates:
            intermediates = {
                "generated_text": texts,
                "parsed_json": parsed_batches,
                "parsed_elements": parsed_numeric,
                "id2label_per_example": [
                    dict(enumerate(self._batch_label_names([batch])))
                    for batch in parsed_batches
                ],
            }
            if prompts is not None:
                intermediates["prompts"] = list(prompts)
        output = LayoutGenerationOutput(
            bbox=bbox.float(),
            labels=torch.stack(label_rows).long(),
            mask=torch.stack(mask_rows).bool(),
            id2label=id2label,
            sequences=sequences,
            intermediates=intermediates,
        )
        mode = normalize_output_type(output_type)
        if mode is OutputType.dict:
            return dict(output)
        return output

    def _extract_json_array(self, text: str) -> str:
        start = text.find("[")
        if start < 0:
            raise ValueError("PosterLLaVA output does not contain a JSON array")
        depth = 0
        in_string: str | None = None
        escaped = False
        for idx, char in enumerate(text[start:], start=start):
            if in_string is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue
            if char in {"'", '"'}:
                in_string = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        raise ValueError("PosterLLaVA output contains an unterminated JSON array")

    def _wrap_conversation(
        self,
        body: str,
        *,
        conv_mode: ConversationMode | str,
    ) -> str:
        mode = normalize_conversation_mode(conv_mode)
        image_body = f"{IMAGE_TOKEN}\n{body}"
        if mode is ConversationMode.llava_v0:
            return (
                "A chat between a curious human and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the human's questions.###Human: "
                f"{image_body}###Assistant:"
            )
        return (
            "A chat between a curious human and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions. USER: "
            f"{image_body} ASSISTANT:"
        )

    def _format_texts(
        self,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None,
    ) -> str:
        if texts is None:
            return ""
        if isinstance(texts, str):
            return texts
        values: list[str] = []
        for item in texts:
            if isinstance(item, str):
                values.append(item)
            else:
                values.append(", ".join(str(value) for value in item))
        return "; ".join(values)

    def _batch_label_names(
        self,
        batches: Sequence[Sequence[PosterLlavaJsonElement]],
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for batch in batches:
            for item in batch:
                label = item["label"]
                if label not in seen:
                    seen.add(label)
                    names.append(label)
        return names or ["unknown"]

    def _ltrb_to_xywh(
        self,
        bbox: Float[torch.Tensor, "... 4"],
    ) -> Float[torch.Tensor, "... 4"]:
        left, top, right, bottom = bbox.unbind(dim=-1)
        return torch.stack(
            ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
            dim=-1,
        )

    def _xywh_to_ltrb(
        self,
        bbox: Float[torch.Tensor, "... 4"],
    ) -> Float[torch.Tensor, "... 4"]:
        x, y, width, height = bbox.unbind(dim=-1)
        return torch.stack(
            (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
            dim=-1,
        )


__all__ = [
    "PosterLlavaJsonElement",
    "PosterLlavaProcessor",
]
