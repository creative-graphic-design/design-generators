"""PyTorch Lightning module for LayoutFormer++ training."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Shaped
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.optim.lr_scheduler import LRScheduler

from ..configuration_layoutformerpp import LayoutFormerPPConfig
from ..labels import label_translation_for_dataset
from ..modeling_layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    generate_square_subsequent_mask,
)
from ..serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenR,
    T5LayoutSequenceForGenT,
    build_default_tokens,
)
from ..tasks import LayoutFormerPPTask, layoutformerpp_vendor_dataset_slug
from ..tokenization_layoutformerpp import LayoutFormerPPTokenizer
from .recipes import LayoutFormerPPTrainingRecipe, get_training_recipe
from .scheduler import LayoutFormerPPWarmupLR


class LayoutFormerPPTrainingConfig(TypedDict, total=False):
    """LightningCLI-safe constructor values for the runtime model config."""

    dataset: str
    task: str
    vocab_size: int
    max_position_embeddings: int
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_attention_heads: int
    decoder_attention_heads: int
    dim_feedforward: int
    dropout: float
    share_embedding: bool


@dataclass(frozen=True)
class LayoutFormerPPPreOptimizerTrace:
    """Fixed-batch values produced before a training optimizer mutation."""

    input_ids: Int[torch.Tensor, "batch tokens"]
    attention_mask: Bool[torch.Tensor, "batch tokens"] | None
    decoder_input_ids: Int[torch.Tensor, "batch target_tokens"]
    task_ids: Int[torch.Tensor, "batch"] | None
    encoder_memory: Float[torch.Tensor, "source_tokens batch channels"]
    decoder_hidden_state: Float[torch.Tensor, "target_tokens batch channels"]
    logits: Float[torch.Tensor, "batch target_tokens vocab"]
    per_token_loss: Float[torch.Tensor, "batch target_tokens"]
    pad_only_ce_contribution: Float[torch.Tensor, ""]
    loss: Float[torch.Tensor, ""]


def vendor_effective_cross_entropy(
    logits: Float[torch.Tensor, "batch tokens vocab"],
    targets: Int[torch.Tensor, "batch tokens"],
) -> Float[torch.Tensor, ""]:
    """Compute the pad-inclusive cross-entropy used by the reference recipe."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


_TASK_PROMPTS: dict[str, str] = {
    "refinement": "task-refinement :",
    "completion": "task-completion :",
    "ugen": "task-ugen :",
    "gen_t": "task-gen_t :",
    "gen_ts": "task-gen_ts :",
    "gen_r": "task-gen_r :",
}


def _package_tokenizer(recipe: LayoutFormerPPTrainingRecipe) -> LayoutFormerPPTokenizer:
    labels = tuple(
        label_translation_for_dataset(recipe.dataset).sequence_id2label.values()
    )
    tokens = build_default_tokens(
        labels,
        task=recipe.tasks[-1],
        grid=recipe.discrete_x_grid,
    )
    if "add_task_prompt" in recipe.serialization_flags:
        prompt_tokens: list[str] = []
        for task in ("refinement", "completion", "ugen", "gen_t", "gen_ts", "gen_r"):
            for token in _TASK_PROMPTS[task].split():
                if token not in prompt_tokens:
                    prompt_tokens.append(token)
        tokens.extend(prompt_tokens)
    return LayoutFormerPPTokenizer(tokens=tokens)


def _discretize_bboxes(
    bboxes: Float[torch.Tensor, "elements coordinates"],
    recipe: LayoutFormerPPTrainingRecipe,
) -> Int[torch.Tensor, "elements coordinates"]:
    maximum = recipe.discrete_x_grid - 1
    return torch.floor(bboxes.float().clamp(0.0, 1.0) * maximum).long()


def _sort_by_position(
    labels: Int[torch.Tensor, "elements"],
    bboxes: Float[torch.Tensor, "elements coordinates"],
    gold_bboxes: Float[torch.Tensor, "elements coordinates"],
) -> tuple[
    Int[torch.Tensor, "elements"],
    Float[torch.Tensor, "elements coordinates"],
    Float[torch.Tensor, "elements coordinates"],
]:
    order = sorted(
        range(len(labels)),
        key=lambda index: (float(bboxes[index, 1]), float(bboxes[index, 0])),
    )
    indices = torch.tensor(order, dtype=torch.long, device=labels.device)
    return labels[indices], bboxes[indices], gold_bboxes[indices]


def _sort_by_label(
    labels: Int[torch.Tensor, "elements"],
    bboxes: Float[torch.Tensor, "elements coordinates"],
    gold_bboxes: Float[torch.Tensor, "elements coordinates"],
    recipe: LayoutFormerPPTrainingRecipe,
) -> tuple[
    Int[torch.Tensor, "elements"],
    Float[torch.Tensor, "elements coordinates"],
    Float[torch.Tensor, "elements coordinates"],
]:
    order = sorted(range(len(labels)), key=lambda index: f"label_{int(labels[index])}")
    indices = torch.tensor(order, dtype=torch.long, device=labels.device)
    return labels[indices], bboxes[indices], gold_bboxes[indices]


def _shuffle_elements(
    labels: Int[torch.Tensor, "elements"],
    bboxes: Float[torch.Tensor, "elements coordinates"],
    gold_bboxes: Float[torch.Tensor, "elements coordinates"],
) -> tuple[
    Int[torch.Tensor, "elements"],
    Float[torch.Tensor, "elements coordinates"],
    Float[torch.Tensor, "elements coordinates"],
]:
    order = list(range(len(labels)))
    random.shuffle(order)
    indices = torch.tensor(order, dtype=torch.long, device=labels.device)
    return labels[indices], bboxes[indices], gold_bboxes[indices]


def _apply_ordering(
    task: LayoutFormerPPTask,
    labels: Int[torch.Tensor, "elements"],
    bboxes: Float[torch.Tensor, "elements coordinates"],
    gold_bboxes: Float[torch.Tensor, "elements coordinates"],
    recipe: LayoutFormerPPTrainingRecipe,
) -> tuple[
    Int[torch.Tensor, "elements"],
    Float[torch.Tensor, "elements coordinates"],
    Float[torch.Tensor, "elements coordinates"],
]:
    sort_by_position = "sort_by_dict" not in recipe.serialization_flags
    if task is LayoutFormerPPTask.completion:
        labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
        if "completion_sort_by_pos" in recipe.serialization_flags:
            return labels, bboxes, gold_bboxes
        return _sort_by_label(labels, bboxes, gold_bboxes, recipe)
    if task is LayoutFormerPPTask.refinement:
        if sort_by_position:
            return _sort_by_position(labels, bboxes, gold_bboxes)
        if "refinement_shuffle_before_sort_by_label" in recipe.serialization_flags:
            labels, bboxes, gold_bboxes = _shuffle_elements(labels, bboxes, gold_bboxes)
        elif (
            "refinement_sort_by_pos_before_sort_by_label" in recipe.serialization_flags
        ):
            labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
        return _sort_by_label(labels, bboxes, gold_bboxes, recipe)
    if task is LayoutFormerPPTask.gen_r:
        if sort_by_position:
            return _sort_by_position(labels, bboxes, gold_bboxes)
        if "gen_r_shuffle_before_sort_by_label" in recipe.serialization_flags:
            labels, bboxes, gold_bboxes = _shuffle_elements(labels, bboxes, gold_bboxes)
        elif "gen_r_sort_by_pos_before_sort_by_label" in recipe.serialization_flags:
            labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
        return _sort_by_label(labels, bboxes, gold_bboxes, recipe)
    if task in (LayoutFormerPPTask.gen_t, LayoutFormerPPTask.gen_ts):
        if sort_by_position:
            return _sort_by_position(labels, bboxes, gold_bboxes)
        flag = f"{task}_shuffle_before_sort_by_label"
        position_flag = f"{task}_sort_by_pos_before_sort_by_label"
        if flag in recipe.serialization_flags:
            labels, bboxes, gold_bboxes = _shuffle_elements(labels, bboxes, gold_bboxes)
        elif position_flag in recipe.serialization_flags:
            labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
        return _sort_by_label(labels, bboxes, gold_bboxes, recipe)
    labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
    if (
        task is LayoutFormerPPTask.ugen
        and "ugen_sort_by_pos" in recipe.serialization_flags
    ):
        return labels, bboxes, gold_bboxes
    flag = f"{task}_shuffle_before_sort_by_label"
    position_flag = f"{task}_sort_by_pos_before_sort_by_label"
    if flag in recipe.serialization_flags:
        labels, bboxes, gold_bboxes = _shuffle_elements(labels, bboxes, gold_bboxes)
    elif position_flag in recipe.serialization_flags:
        labels, bboxes, gold_bboxes = _sort_by_position(labels, bboxes, gold_bboxes)
    return _sort_by_label(labels, bboxes, gold_bboxes, recipe)


def _relation_size(
    box_a: Float[torch.Tensor, "coordinates"],
    box_b: Float[torch.Tensor, "coordinates"],
) -> int:
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    if area_b <= 0.9 * area_a:
        return 0
    if area_b < 1.1 * area_a:
        return 1
    return 2


def _relation_location(
    box_a: Float[torch.Tensor, "coordinates"],
    box_b: Float[torch.Tensor, "coordinates"],
    canvas: bool,
) -> int:
    if canvas:
        center_y = box_b[1] + box_b[3] / 2
        if center_y <= 1 / 3:
            return 3
        if center_y < 2 / 3:
            return 4
        return 5
    left_a, top_a, right_a, bottom_a = (
        box_a[0],
        box_a[1],
        box_a[0] + box_a[2],
        box_a[1] + box_a[3],
    )
    left_b, top_b, right_b, bottom_b = (
        box_b[0],
        box_b[1],
        box_b[0] + box_b[2],
        box_b[1] + box_b[3],
    )
    if bottom_b <= top_a:
        return 3
    if bottom_a <= top_b:
        return 5
    if right_b <= left_a:
        return 6
    if right_a <= left_b:
        return 7
    return 4


def _relations(
    labels: Int[torch.Tensor, "elements"],
    bboxes: Float[torch.Tensor, "elements coordinates"],
    generator: random.Random,
) -> list[tuple[int, int, int, int, int]]:
    canvas_labels = torch.cat((torch.zeros(1, dtype=torch.long), labels.cpu()))
    canvas_bboxes = torch.cat(
        (torch.tensor([[0.0, 0.0, 1.0, 1.0]]), bboxes.cpu()), dim=0
    ).float()
    label_counts: dict[int, int] = {}
    label_indices: list[int] = []
    for label in canvas_labels.tolist():
        label_value = int(label)
        label_counts[label_value] = label_counts.get(label_value, 0) + 1
        label_indices.append(label_counts[label_value])
    pairs = [
        (relation, pair)
        for relation in range(2)
        for pair in __import__("itertools").combinations(range(len(canvas_labels)), 2)
    ]
    sampled = set(generator.sample(pairs, int(len(pairs) * 0.1)))
    relations: list[tuple[int, int, int, int, int]] = []
    for first, second in __import__("itertools").combinations(
        range(len(canvas_labels)), 2
    ):
        if (0, (first, second)) in sampled and first != 0:
            relations.append(
                (
                    int(canvas_labels[first]),
                    label_indices[first],
                    int(canvas_labels[second]),
                    label_indices[second],
                    _relation_size(canvas_bboxes[first], canvas_bboxes[second]),
                )
            )
        if (1, (first, second)) in sampled:
            relations.append(
                (
                    int(canvas_labels[first]),
                    label_indices[first],
                    int(canvas_labels[second]),
                    label_indices[second],
                    _relation_location(
                        canvas_bboxes[first], canvas_bboxes[second], first == 0
                    ),
                )
            )
    return relations


class _LayoutFormerPPSample(TypedDict):
    """One processed source record after package serialization."""

    input_text: str
    output_text: str
    input_bytes: bytes
    output_bytes: bytes
    name: str
    task_name: str
    task_id: int


class _LayoutFormerPPBatch(TypedDict):
    """One collated package batch consumed by the training hooks."""

    input_ids: Int[torch.Tensor, "batch source_tokens"]
    attention_mask: Bool[torch.Tensor, "batch source_tokens"]
    labels: Int[torch.Tensor, "batch target_tokens"]
    target_attention_mask: Bool[torch.Tensor, "batch target_tokens"]
    task_ids: Int[torch.Tensor, "batch"]
    names: list[str]
    task_names: list[str]
    input_bytes: list[bytes]
    output_bytes: list[bytes]


class _LayoutFormerPPDataset(Dataset[_LayoutFormerPPSample]):
    def __init__(
        self,
        recipe: LayoutFormerPPTrainingRecipe,
        data_root: Path,
        split: str,
        *,
        train: bool,
    ) -> None:
        self.recipe = recipe
        self.split = split
        self.train = train
        dataset_slug = layoutformerpp_vendor_dataset_slug(recipe.dataset)
        label_count = 25 if dataset_slug == "rico" else 5
        path = (
            data_root / dataset_slug / f"pre_processed_20_{label_count}" / f"{split}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"LayoutFormer++ processed split is missing: {path}"
            )
        values = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(values, list) or not values:
            raise ValueError(f"LayoutFormer++ processed split is empty: {path}")
        self.data = cast(list[_LayoutFormerPPSample], values)
        self._relation_generator = random.Random(1024)
        self._partition_data: dict[int, int] = {}
        self._bucket_tasks: list[list[LayoutFormerPPTask]] = []
        if train and recipe.partition_buckets:
            grouped: dict[int, list[LayoutFormerPPTask]] = {}
            for task, bucket in zip(
                recipe.tasks, recipe.partition_buckets, strict=True
            ):
                grouped.setdefault(bucket, []).append(task)
            bucket_map: dict[int, int] = {}
            for bucket, tasks in grouped.items():
                bucket_map[bucket] = len(self._bucket_tasks)
                self._bucket_tasks.append(tasks)
            bucket_sizes = [
                math.ceil(len(self.data) * len(tasks) / len(recipe.tasks))
                for tasks in self._bucket_tasks
            ]
            permutation = np.random.RandomState(100).permutation(
                np.arange(len(self.data))
            )
            splits = np.split(permutation, np.cumsum(bucket_sizes)[:-1])
            for bucket_id, indices in enumerate(splits):
                for index in indices.tolist():
                    self._partition_data[int(index)] = bucket_id
            for bucket, tasks in grouped.items():
                if bucket < 0:
                    for positive, _ in grouped.items():
                        if positive >= 0:
                            self._bucket_tasks[bucket_map[positive]].extend(tasks)

    def __len__(self) -> int:
        return len(self.data)

    def _task_for_index(self, index: int) -> LayoutFormerPPTask:
        if self.train and self.recipe.partition_buckets:
            return random.choice(self._bucket_tasks[self._partition_data[index]])
        if self.train:
            probabilities = np.ones(len(self.recipe.tasks)) / len(self.recipe.tasks)
            selected = int(
                np.random.choice(len(self.recipe.tasks), 1, p=probabilities)[0]
            )
            return self.recipe.tasks[selected]
        return self.recipe.eval_tasks[0]

    def __getitem__(self, index: int) -> _LayoutFormerPPSample:
        record = self.data[index]
        labels_value = record.get("labels")
        bboxes_value = record.get("bboxes")
        name_value = record.get("name")
        if not isinstance(labels_value, torch.Tensor) or not isinstance(
            bboxes_value, torch.Tensor
        ):
            raise TypeError("processed records must contain tensor labels and bboxes")
        if not isinstance(name_value, str):
            raise TypeError("processed records must contain a string name")
        task = self._task_for_index(index)
        labels = labels_value.long().clone()
        bboxes = bboxes_value.float().clone()
        gold_bboxes = bboxes.clone()
        if task is LayoutFormerPPTask.refinement:
            element_with_noise = torch.bernoulli(torch.ones(len(bboxes)))
            bboxes = bboxes + torch.randn(bboxes.size()) * 0.01
            bboxes.clamp_(0.0, 1.0)
            bboxes = bboxes * element_with_noise.unsqueeze(-1) + gold_bboxes * (
                1 - element_with_noise.unsqueeze(-1)
            )
        if task is LayoutFormerPPTask.completion:
            input_labels, input_bboxes, gold_bboxes = _sort_by_position(
                labels, bboxes, gold_bboxes
            )
            if "completion_sort_by_pos" in self.recipe.serialization_flags:
                labels, bboxes = input_labels, input_bboxes
            else:
                labels, bboxes, gold_bboxes = _sort_by_label(
                    input_labels, input_bboxes, gold_bboxes, self.recipe
                )
        else:
            labels, bboxes, gold_bboxes = _apply_ordering(
                task, labels, bboxes, gold_bboxes, self.recipe
            )
            input_labels, input_bboxes = labels, bboxes
        discrete_bboxes = _discretize_bboxes(bboxes, self.recipe)
        discrete_gold_bboxes = _discretize_bboxes(gold_bboxes, self.recipe)
        relations: list[tuple[int, int, int, int, int]] = []
        if task is LayoutFormerPPTask.gen_r:
            relation_bboxes = discrete_gold_bboxes.float() / (
                self.recipe.discrete_x_grid - 1
            )
            relations = _relations(labels, relation_bboxes, self._relation_generator)
        label_count = len(
            label_translation_for_dataset(self.recipe.dataset).sequence_id2label
        )
        id2label = {index: f"label_{index}" for index in range(1, label_count + 1)}
        base = T5LayoutSequence(id2label)
        if task is LayoutFormerPPTask.refinement:
            input_text = base.build_seq(labels.tolist(), discrete_bboxes.tolist())
        elif task is LayoutFormerPPTask.completion:
            input_text = base.build_seq(
                input_labels[:1].tolist(),
                _discretize_bboxes(input_bboxes, self.recipe)[:1].tolist(),
            )
        elif task is LayoutFormerPPTask.ugen:
            input_text = ""
        elif task in (LayoutFormerPPTask.gen_t, LayoutFormerPPTask.gen_ts):
            serializer = T5LayoutSequenceForGenT(id2label)
            input_text = serializer.build_input_seq(
                task,
                labels.tolist(),
                discrete_bboxes.tolist(),
                add_unk_for_label="gen_t_add_unk_token"
                in self.recipe.serialization_flags,
                add_unk_for_label_size="gen_ts_add_unk_token"
                in self.recipe.serialization_flags,
            )
        else:
            serializer = T5LayoutSequenceForGenR(id2label)
            input_text = serializer.build_input_seq(
                labels.tolist(),
                relations,
                add_unk_token="gen_r_add_unk_token" in self.recipe.serialization_flags,
                compact="gen_r_compact" in self.recipe.serialization_flags,
            )
        if "add_task_prompt" in self.recipe.serialization_flags:
            input_text = f"{_TASK_PROMPTS[str(task)].lower()} {input_text}"
        input_text = input_text.lower()
        if "add_task_prompt" not in self.recipe.serialization_flags:
            input_text = input_text.strip()
        output_text = base.build_seq(labels.tolist(), discrete_gold_bboxes.tolist())
        return {
            "input_text": input_text,
            "output_text": output_text.lower().strip(),
            "input_bytes": input_text.encode("utf-8"),
            "output_bytes": output_text.lower().strip().encode("utf-8"),
            "name": name_value,
            "task_name": str(task),
            "task_id": self.recipe.task_ids[self.recipe.tasks.index(task)],
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    info = get_worker_info()
    if info is None:
        return
    seed = int(info.seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class LayoutFormerPPDataModule(LightningDataModule):
    """Package-local DataModule matching the pinned basic loader behavior."""

    def __init__(
        self,
        *,
        recipe_name: str,
        data_root: str | None = None,
        num_workers: int = 0,
        seed: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize source-root, recipe, and worker-seeding configuration.

        Args:
            recipe_name: Registered recipe used to select dataset and tasks.
            data_root: Processed source-data root, or ``None`` to read an
                environment variable at setup time.
            num_workers: Number of DataLoader worker processes.
            seed: Reserved loader seed recorded with the module configuration.
            pin_memory: Whether DataLoader batches use pinned host memory.
        """
        super().__init__()
        self.recipe = get_training_recipe(recipe_name)
        self.data_root = data_root
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_dataset: _LayoutFormerPPDataset | None = None
        self.val_dataset: _LayoutFormerPPDataset | None = None
        self.tokenizer = _package_tokenizer(self.recipe)

    def setup(self, stage: str | None = None) -> None:
        """Load the processed train and validation splits for ``stage``."""
        del stage
        root_value = self.data_root or os.environ.get("LAYOUTFORMERPP_DATA_ROOT")
        if not root_value:
            root_value = os.environ.get("LAYOUTFORMERPP_PARITY_DATA_ROOT")
        if not root_value:
            raise RuntimeError(
                "LayoutFormer++ DataModule requires data_root or "
                "LAYOUTFORMERPP_DATA_ROOT"
            )
        root = Path(root_value)
        self.train_dataset = _LayoutFormerPPDataset(
            self.recipe, root, "train", train=True
        )
        self.val_dataset = _LayoutFormerPPDataset(self.recipe, root, "val", train=False)

    def _collate(self, samples: list[_LayoutFormerPPSample]) -> _LayoutFormerPPBatch:
        input_text = [sample["input_text"] for sample in samples]
        output_text = [sample["output_text"] for sample in samples]
        inputs = self.tokenizer.encode_text(input_text, add_eos=True)
        targets = self.tokenizer.encode_text(output_text, add_eos=True)
        return {
            "input_ids": inputs["input_ids"].long(),
            "attention_mask": inputs["attention_mask"].bool(),
            "labels": targets["input_ids"].long(),
            "target_attention_mask": targets["attention_mask"].bool(),
            "task_ids": torch.tensor(
                [int(sample["task_id"]) for sample in samples], dtype=torch.long
            ),
            "names": [sample["name"] for sample in samples],
            "task_names": [sample["task_name"] for sample in samples],
            "input_bytes": [sample["input_bytes"] for sample in samples],
            "output_bytes": [sample["output_bytes"] for sample in samples],
        }

    def train_dataloader(self) -> DataLoader[_LayoutFormerPPSample]:
        """Return the shuffled, drop-last production training loader."""
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None:
            raise RuntimeError("train dataset was not initialized")
        return DataLoader(
            self.train_dataset,
            batch_size=self.recipe.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
            worker_init_fn=_seed_worker if self.num_workers else None,
            generator=None,
        )

    def val_dataloader(self) -> DataLoader[_LayoutFormerPPSample]:
        """Return the deterministic, drop-last production validation loader."""
        if self.val_dataset is None:
            self.setup("fit")
        if self.val_dataset is None:
            raise RuntimeError("validation dataset was not initialized")
        return DataLoader(
            self.val_dataset,
            batch_size=self.recipe.eval_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
            worker_init_fn=_seed_worker if self.num_workers else None,
            generator=None,
        )


class LayoutFormerPPTrainingModule(LightningModule):
    """Train the genuine runtime LayoutFormer++ model with faithful static wiring."""

    def __init__(
        self,
        *,
        recipe_name: str,
        config: LayoutFormerPPTrainingConfig,
        model: LayoutFormerPPForConditionalGeneration | None = None,
    ) -> None:
        """Initialize one immutable recipe and its runtime model."""
        super().__init__()
        self.recipe: LayoutFormerPPTrainingRecipe = get_training_recipe(recipe_name)
        runtime_config = LayoutFormerPPConfig(**config)
        self._validate_config(runtime_config)
        self.layoutformerpp_config = runtime_config
        self.model = model or LayoutFormerPPForConditionalGeneration(runtime_config)

    def _validate_config(self, config: LayoutFormerPPConfig) -> None:
        expected = self.recipe
        actual = {
            "dataset": config.dataset,
            "condition": config.condition_type,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
        }
        required = {
            "dataset": str(expected.dataset),
            "condition": str(expected.condition),
            "vocab_size": expected.vocab_size,
            "max_position_embeddings": expected.max_position_embeddings,
        }
        if actual != required:
            raise ValueError(
                f"LayoutFormer++ config does not match recipe {expected.name}: "
                f"expected {required}, got {actual}"
            )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        decoder_input_ids: Int[torch.Tensor, "batch target_tokens"] | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
    ) -> Shaped[torch.Tensor, "batch target_tokens vocab"]:
        """Return logits from the owned runtime model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            task_ids=task_ids,
            return_dict=True,
        )
        return cast(torch.Tensor, outputs.logits)

    def _loss(
        self,
        batch: Mapping[
            str, Shaped[torch.Tensor, "..."] | list[str] | list[bytes] | None
        ],
    ) -> Float[torch.Tensor, ""]:
        """Return the package loss from the pre-optimizer trace."""
        return self.pre_optimizer_trace(batch).loss

    def pre_optimizer_trace(
        self,
        batch: Mapping[
            str, Shaped[torch.Tensor, "..."] | list[str] | list[bytes] | None
        ],
    ) -> LayoutFormerPPPreOptimizerTrace:
        """Capture package-model inputs, logits, and loss before an optimizer step."""
        input_ids_value = batch["input_ids"]
        labels_value = batch["labels"]
        if not isinstance(input_ids_value, torch.Tensor) or not isinstance(
            labels_value, torch.Tensor
        ):
            raise TypeError("pre_optimizer_trace requires tensor input_ids and labels")
        input_ids = input_ids_value.long()
        labels = labels_value.long()
        attention_mask_value = batch.get("attention_mask")
        attention_mask = (
            attention_mask_value.bool()
            if isinstance(attention_mask_value, torch.Tensor)
            else None
        )
        effective_attention_mask = (
            attention_mask
            if attention_mask is not None
            else input_ids.ne(self.model.pad_token_id)
        )
        task_ids_value = batch.get("task_ids")
        if task_ids_value is not None and not isinstance(task_ids_value, torch.Tensor):
            raise TypeError("task_ids must be a tensor or None")
        task_ids = task_ids_value.long() if task_ids_value is not None else None
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        enc_hs, enc_padding_mask = self.model.encode(
            input_ids, ~effective_attention_mask, task_ids
        )
        dec_input = self.model.dec_pos_embedding(
            self.model.dec_embedding(decoder_input_ids).permute(1, 0, 2)
        )
        decoder_hidden_state = self.model.decoder(
            tgt=dec_input,
            memory=enc_hs,
            tgt_mask=generate_square_subsequent_mask(
                dec_input.size(0), dec_input.device
            ),
            memory_key_padding_mask=enc_padding_mask,
        )
        logits = self.model.out(decoder_hidden_state.permute(1, 0, 2))
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        pad_mask = labels.eq(self.model.pad_token_id)
        pad_only_ce_contribution = (
            per_token_loss.masked_select(pad_mask).sum() / labels.numel()
        )
        return LayoutFormerPPPreOptimizerTrace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            task_ids=task_ids,
            encoder_memory=enc_hs,
            decoder_hidden_state=decoder_hidden_state,
            logits=logits,
            per_token_loss=per_token_loss,
            pad_only_ce_contribution=pad_only_ce_contribution,
            loss=vendor_effective_cross_entropy(logits, labels),
        )

    def training_step(
        self,
        batch: Mapping[
            str, Shaped[torch.Tensor, "..."] | list[str] | list[bytes] | None
        ],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Compute the package training loss without altering runtime loss semantics."""
        del batch_idx
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: Mapping[
            str, Shaped[torch.Tensor, "..."] | list[str] | list[bytes] | None
        ],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Compute aggregate validation loss used for checkpoint selection."""
        del batch_idx
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct basic-mode Adam and the post-update logarithmic scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.recipe.learning_rate,
        )
        scheduler = LayoutFormerPPWarmupLR(
            optimizer,
            warmup_num_steps=self.recipe.warmup_num_steps,
            warmup_max_lr=self.recipe.learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(
        self,
        scheduler: LRScheduler,
        metric: float | None,
    ) -> None:
        """Advance exactly once after Lightning completes an optimizer update."""
        del metric
        scheduler.step()


__all__ = [
    "LayoutFormerPPDataModule",
    "LayoutFormerPPPreOptimizerTrace",
    "LayoutFormerPPTrainingConfig",
    "LayoutFormerPPTrainingModule",
    "vendor_effective_cross_entropy",
]
