"""Processor for LT-Net scene graphs and layout outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_layout_transformer import (
    DEFAULT_ID2LABEL,
    DEFAULT_RELATION_ID2LABEL,
)
from .modeling_layout_transformer import LayoutTransformerModelOutput
from .relation_schema import LayoutObject, LayoutRelation, SceneGraphInput
from .tokenization_layout_transformer import LayoutTransformerRelationTokenizer


OutputType = Literal["dataclass", "dict"]
InputTensor: TypeAlias = Int[torch.Tensor, "batch sequence"]
BoxBatchTensor: TypeAlias = Float[torch.Tensor, "batch elements 4"]
LabelBatchTensor: TypeAlias = Int[torch.Tensor, "batch elements"]
MaskBatchTensor: TypeAlias = Bool[torch.Tensor, "batch elements"]
RowBoxTensor: TypeAlias = Float[torch.Tensor, "elements 4"]
RowLabelTensor: TypeAlias = Int[torch.Tensor, "elements"]
RowMaskTensor: TypeAlias = Bool[torch.Tensor, "elements"]


class LayoutTransformerProcessor(ProcessorMixin):
    """Normalize scene graphs, tokenize LT-Net inputs, and postprocess boxes."""

    attributes = ["tokenizer"]
    tokenizer_class = "LayoutTransformerRelationTokenizer"

    def __init__(
        self,
        tokenizer: LayoutTransformerRelationTokenizer,
        dataset_name: str = "coco",
        max_sequence_length: int = 128,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        relation_id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        object_reduce: Literal["first", "last", "mean"] = "first",
    ) -> None:
        """Initialize processor label maps and tokenizer component."""
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_sequence_length = max_sequence_length
        self.id2label = {
            int(key): str(value)
            for key, value in (id2label or DEFAULT_ID2LABEL).items()
        }
        self.relation_id2label = {
            int(key): str(value)
            for key, value in (relation_id2label or DEFAULT_RELATION_ID2LABEL).items()
        }
        self.label2id = {value.lower(): key for key, value in self.id2label.items()}
        self.relation_label2id = {
            value.lower(): key for key, value in self.relation_id2label.items()
        }
        if object_reduce not in {"first", "last", "mean"}:
            raise ValueError("object_reduce must be 'first', 'last', or 'mean'")
        self.object_reduce = object_reduce
        super().__init__(tokenizer=tokenizer)

    @classmethod
    def from_config(
        cls,
        *,
        dataset_name: str = "coco",
        max_sequence_length: int = 128,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        relation_id2label: Mapping[int, str] | Mapping[str, str] | None = None,
    ) -> "LayoutTransformerProcessor":
        """Construct a processor and tokenizer without external files.

        Returns:
            Processor with synthetic vocabulary derived from the label maps.

        Examples:
            >>> processor = LayoutTransformerProcessor.from_config()
            >>> processor.tokenizer.cls_token_id
            1
        """
        object_labels = {
            int(key): str(value)
            for key, value in (id2label or DEFAULT_ID2LABEL).items()
        }
        relation_labels = {
            int(key): str(value)
            for key, value in (relation_id2label or DEFAULT_RELATION_ID2LABEL).items()
        }
        tokens = ["__image__"]
        tokens.extend(value for _, value in sorted(object_labels.items()))
        tokens.extend(value for _, value in sorted(relation_labels.items()))
        tokenizer = LayoutTransformerRelationTokenizer(tokens=tokens)
        return cls(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            id2label=object_labels,
            relation_id2label=relation_labels,
        )

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls,
        sub_processor_type: str,
        pretrained_model_name_or_path: str | PathLike[str],
        subfolder: str = "",
        **kwargs: object,
    ) -> LayoutTransformerRelationTokenizer:
        """Load tokenizer for ``ProcessorMixin.from_pretrained``."""
        _ = sub_processor_type
        path = Path(pretrained_model_name_or_path)
        tokenizer_path = path / subfolder if subfolder else path
        token = kwargs.get("token")
        return LayoutTransformerRelationTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cast(str | PathLike[str] | None, kwargs.get("cache_dir")),
            force_download=bool(kwargs.get("force_download", False)),
            local_files_only=bool(kwargs.get("local_files_only", False)),
            token=token if isinstance(token, str | bool) else None,
            revision=str(kwargs.get("revision", "main")),
        )

    def normalize_condition_type(
        self, condition_type: ConditionType | str
    ) -> ConditionType:
        """Normalize and validate the LT-Net public condition type."""
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.relation:
            raise ValueError(
                "LayoutTransformer only supports condition_type='relation' "
                "and aliases 'scene_graph', 'graph', or 'gen_r'."
            )
        return condition

    def _label_to_id(self, label: int | str) -> int:
        if isinstance(label, int):
            return label
        lowered = label.lower()
        if lowered in self.label2id:
            return self.label2id[lowered]
        raise ValueError(f"Unknown object label: {label}")

    def _relation_to_id(self, predicate: int | str) -> int:
        if isinstance(predicate, int):
            return predicate
        lowered = predicate.lower()
        if lowered in self.relation_label2id:
            return self.relation_label2id[lowered]
        raise ValueError(f"Unknown relation label: {predicate}")

    def _token_for_object(self, label_id: int) -> str:
        return self.id2label.get(label_id, str(label_id))

    def _token_for_relation(self, relation_id: int) -> str:
        return self.relation_id2label.get(relation_id, str(relation_id))

    def _normalize_scene_graph(
        self,
        scene_graph: SceneGraphInput | Mapping[str, object] | None,
        *,
        objects: Sequence[LayoutObject] | None,
        relations: Sequence[LayoutRelation] | None,
    ) -> SceneGraphInput:
        if isinstance(scene_graph, SceneGraphInput):
            return scene_graph
        if scene_graph is None:
            if objects is None:
                raise ValueError("scene_graph or objects must be provided")
            return SceneGraphInput(
                objects=tuple(objects),
                relations=tuple(relations or ()),
                id2label=self.id2label,
                relation_id2label=self.relation_id2label,
            )
        nodes = scene_graph.get("nodes", scene_graph.get("objects", ()))
        edges = scene_graph.get("edges", scene_graph.get("relations", ()))
        normalized_objects: list[LayoutObject] = []
        for node in cast(Sequence[object], nodes):
            item = cast(Mapping[str, object], node)
            node_id = cast(int | str, item["id"])
            label = cast(int | str, item.get("label_id", item.get("label")))
            bbox = cast(tuple[float, float, float, float] | None, item.get("bbox"))
            normalized_objects.append(LayoutObject(id=node_id, label=label, bbox=bbox))
        normalized_relations: list[LayoutRelation] = []
        for edge in cast(Sequence[object], edges):
            item = cast(Mapping[str, object], edge)
            subject = cast(int | str, item.get("source", item.get("subject")))
            predicate = cast(int | str, item.get("predicate_id", item.get("predicate")))
            target = cast(int | str, item.get("target", item.get("object")))
            normalized_relations.append(
                LayoutRelation(
                    subject=subject,
                    predicate=predicate,
                    object=target,
                    score=cast(float | None, item.get("score")),
                )
            )
        return SceneGraphInput(
            objects=tuple(normalized_objects),
            relations=tuple(normalized_relations),
            id2label=cast(dict[int, str] | None, scene_graph.get("id2label")),
            relation_id2label=cast(
                dict[int, str] | None,
                scene_graph.get("relation_id2label"),
            ),
        )

    def _serialize_graph(
        self,
        graph: SceneGraphInput,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        object_by_id = {item.id: item for item in graph.objects}
        object_ids = {item.id: idx + 1 for idx, item in enumerate(graph.objects)}
        tokens = [self.tokenizer.cls_token]
        input_obj_id = [0]
        segment_label = [0]
        token_type = [0]
        segment = 1
        for relation in graph.relations:
            subject = object_by_id[relation.subject]
            target = object_by_id[relation.object]
            subject_label = self._label_to_id(subject.label)
            target_label = self._label_to_id(target.label)
            relation_id = self._relation_to_id(relation.predicate)
            triple_tokens = [
                self._token_for_object(subject_label),
                self._token_for_relation(relation_id),
                self._token_for_object(target_label),
                self.tokenizer.sep_token,
            ]
            tokens.extend(triple_tokens)
            input_obj_id.extend([object_ids[subject.id], 0, object_ids[target.id], 0])
            segment_label.extend([segment] * 4)
            token_type.extend([1, 2, 3, 0])
            segment += 1
        if not graph.relations:
            for item in graph.objects:
                label_id = self._label_to_id(item.label)
                tokens.extend(
                    [self._token_for_object(label_id), self.tokenizer.sep_token]
                )
                input_obj_id.extend([object_ids[item.id], 0])
                segment_label.extend([segment, segment])
                token_type.extend([1, 0])
                segment += 1
        input_token = self.tokenizer.encode_scene_graph_tokens(tokens)
        length = min(len(input_token), self.max_sequence_length)
        input_token = input_token[:length]
        input_obj_id = input_obj_id[:length]
        segment_label = segment_label[:length]
        token_type = token_type[:length]
        src_mask = [1] * length
        pad_length = self.max_sequence_length - length
        input_token.extend([self.tokenizer.pad_token_id] * pad_length)
        input_obj_id.extend([0] * pad_length)
        segment_label.extend([0] * pad_length)
        token_type.extend([0] * pad_length)
        src_mask.extend([0] * pad_length)
        return input_token, input_obj_id, segment_label, token_type, src_mask

    def __call__(
        self,
        *,
        scene_graph: SceneGraphInput | Mapping[str, object] | None = None,
        objects: Sequence[LayoutObject] | None = None,
        relations: Sequence[LayoutRelation] | None = None,
        batch_size: int = 1,
        condition_type: ConditionType | str = ConditionType.relation,
        return_tensors: Literal["pt", "np"] = "pt",
        max_sequence_length: int | None = None,
    ) -> BatchEncoding:
        """Build LT-Net model tensors from public scene-graph inputs."""
        self.normalize_condition_type(condition_type)
        original_max_length = self.max_sequence_length
        try:
            if max_sequence_length is not None:
                self.max_sequence_length = max_sequence_length
            graph = self._normalize_scene_graph(
                scene_graph,
                objects=objects,
                relations=relations,
            )
            rows = [self._serialize_graph(graph) for _ in range(batch_size)]
        finally:
            self.max_sequence_length = original_max_length
        data = {
            "input_token": torch.tensor([row[0] for row in rows], dtype=torch.long),
            "input_obj_id": torch.tensor([row[1] for row in rows], dtype=torch.long),
            "segment_label": torch.tensor([row[2] for row in rows], dtype=torch.long),
            "token_type": torch.tensor([row[3] for row in rows], dtype=torch.long),
            "src_mask": torch.tensor(
                [row[4] for row in rows], dtype=torch.bool
            ).unsqueeze(1),
            "global_mask": torch.tensor([row[0] for row in rows], dtype=torch.long).ge(
                2
            ),
        }
        if return_tensors == "pt":
            return BatchEncoding(data)
        if return_tensors == "np":
            return BatchEncoding({key: value.numpy() for key, value in data.items()})
        raise ValueError("return_tensors must be 'pt' or 'np'")

    def post_process_layout_generation(
        self,
        model_outputs: LayoutTransformerModelOutput,
        *,
        input_token: InputTensor | None = None,
        input_obj_id: InputTensor,
        token_type: InputTensor,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Convert raw token-level boxes into public object-level layouts."""
        _ = (canvas_size, normalize_box_format(box_format))
        if not normalized:
            raise ValueError("LayoutTransformer outputs normalized boxes only")
        raw_box = model_outputs.refine_box
        if raw_box is None:
            raw_box = model_outputs.coarse_box
        if raw_box is None:
            raise ValueError("model_outputs must contain coarse_box or refine_box")
        batch_boxes: list[RowBoxTensor] = []
        batch_labels: list[RowLabelTensor] = []
        batch_masks: list[RowMaskTensor] = []
        token_rows = (
            [None] * raw_box.size(0)
            if input_token is None
            else list(input_token.unbind(dim=0))
        )
        for row_box, row_obj_id, row_type, row_token in zip(
            raw_box, input_obj_id, token_type, token_rows, strict=True
        ):
            object_positions = row_type.eq(1) | row_type.eq(3)
            object_ids = row_obj_id[object_positions]
            boxes = row_box[object_positions].clamp(0.0, 1.0)
            labels = (
                torch.clamp(object_ids - 1, min=0).long()
                if row_token is None
                else row_token[object_positions].long()
            )
            valid = object_ids.gt(0)
            boxes = boxes[valid]
            labels = labels[valid]
            object_ids = object_ids[valid]
            reduced_boxes: list[RowBoxTensor] = []
            reduced_labels: list[RowLabelTensor] = []
            for object_id in object_ids.unique(sorted=True):
                positions = object_ids.eq(object_id).nonzero().flatten()
                if self.object_reduce == "mean":
                    reduced_boxes.append(boxes[positions].mean(dim=0))
                    reduced_labels.append(labels[positions[0]])
                else:
                    selected = (
                        positions[0] if self.object_reduce == "first" else positions[-1]
                    )
                    reduced_boxes.append(boxes[selected])
                    reduced_labels.append(labels[selected])
            if reduced_boxes:
                batch_boxes.append(torch.stack(reduced_boxes))
                batch_labels.append(torch.stack(reduced_labels).long())
            else:
                batch_boxes.append(boxes)
                batch_labels.append(labels)
            batch_masks.append(
                torch.ones(len(reduced_boxes), dtype=torch.bool, device=row_box.device)
            )
        max_items = max((item.size(0) for item in batch_boxes), default=0)
        padded_boxes = raw_box.new_zeros((raw_box.size(0), max_items, 4))
        padded_labels = input_obj_id.new_zeros((raw_box.size(0), max_items))
        padded_masks = torch.zeros(
            (raw_box.size(0), max_items), dtype=torch.bool, device=raw_box.device
        )
        for idx, (boxes, labels, mask) in enumerate(
            zip(batch_boxes, batch_labels, batch_masks, strict=True)
        ):
            length = boxes.size(0)
            padded_boxes[idx, :length] = boxes
            padded_labels[idx, :length] = labels
            padded_masks[idx, :length] = mask
        intermediates = None
        if return_intermediates:
            intermediates = {
                "coarse_box": model_outputs.coarse_box,
                "refine_box": model_outputs.refine_box,
                "vocab_logits": model_outputs.vocab_logits,
                "obj_id_logits": model_outputs.obj_id_logits,
                "token_type_logits": model_outputs.token_type_logits,
            }
        output = LayoutGenerationOutput(
            bbox=padded_boxes,
            labels=padded_labels,
            mask=padded_masks,
            id2label=dict(self.id2label),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(output.items())
        return output

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> tuple[str, ...]:
        """Save processor metadata and tokenizer files."""
        paths = super().save_pretrained(
            save_directory, push_to_hub=push_to_hub, **kwargs
        )
        metadata = {
            "processor_class": self.__class__.__name__,
            "dataset_name": self.dataset_name,
            "max_sequence_length": self.max_sequence_length,
            "id2label": self.id2label,
            "relation_id2label": self.relation_id2label,
            "object_reduce": self.object_reduce,
        }
        with (Path(save_directory) / "preprocessor_config.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return paths
