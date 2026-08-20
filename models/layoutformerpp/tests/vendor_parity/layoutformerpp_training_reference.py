"""Plain-PyTorch static reference adapter for LayoutFormer++ S0."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, TypedDict, cast

import numpy as np
import torch

from laygen.common.vendor import vendor_root

from layoutformerpp import LayoutFormerPPTokenizer
from layoutformerpp.training.recipes import LayoutFormerPPTrainingRecipe


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_ROOT = vendor_root(
    "ms-layout-generation",
    marker="LayoutFormer++/src/model/layout_transformer/model.py",
    repo_root=PROJECT_ROOT,
)
REFERENCE_SRC = REFERENCE_ROOT / "LayoutFormer++/src"


class _OptimizerDefaults(TypedDict):
    lr: float
    betas: list[float]
    eps: float
    weight_decay: float
    amsgrad: bool


class _OptimizerSchedulerState(TypedDict):
    deepspeed_version: str
    deepspeed_source_sha256: str
    optimizer: str
    scheduler: str
    optimizer_defaults: _OptimizerDefaults
    lr_sequences: dict[str, list[float]]
    post_optimizer_steps: dict[str, dict[str, float]]


class _ReferenceLayoutSequence:
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer


def _source_tree(relative_path: str) -> ast.Module:
    return ast.parse((REFERENCE_SRC / relative_path).read_text(encoding="utf-8"))


def _class_literal(relative_path: str, class_name: str, attribute: str) -> object:
    for node in _source_tree(relative_path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == attribute
                for target in statement.targets
            ):
                return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing {class_name}.{attribute} in {relative_path}")


def _task_prompts() -> tuple[str, ...]:
    wanted = ("refinement", "completion", "ugen", "gen_t", "gen_ts", "gen_r")
    tree = _source_tree("tasks/task_config.py")
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "TASK_CONFIG"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        prompts: dict[str, str] = {}
        for task_node, config_node in zip(
            node.value.keys,
            node.value.values,
            strict=True,
        ):
            if task_node is None:
                raise AssertionError("TASK_CONFIG contains a dictionary unpack")
            task = ast.literal_eval(task_node)
            if task not in wanted or not isinstance(config_node, ast.Dict):
                continue
            for key_node, value_node in zip(
                config_node.keys,
                config_node.values,
                strict=True,
            ):
                if key_node is None:
                    raise AssertionError("task config contains a dictionary unpack")
                if ast.literal_eval(key_node) == "prompt":
                    prompts[task] = ast.literal_eval(value_node)
        return tuple(prompts[task] for task in wanted)
    raise AssertionError("Missing TASK_CONFIG prompts")


def _task_config_values(key: str) -> dict[str, object]:
    tree = _source_tree("tasks/task_config.py")
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "TASK_CONFIG"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        values: dict[str, object] = {}
        for task_node, config_node in zip(
            node.value.keys,
            node.value.values,
            strict=True,
        ):
            if task_node is None or not isinstance(config_node, ast.Dict):
                continue
            task = ast.literal_eval(task_node)
            for key_node, value_node in zip(
                config_node.keys,
                config_node.values,
                strict=True,
            ):
                if key_node is not None and ast.literal_eval(key_node) == key:
                    values[str(task)] = ast.literal_eval(value_node)
                    break
        return values
    raise AssertionError(f"Missing TASK_CONFIG values for {key}")


def reference_task_ids() -> dict[str, int]:
    """Return the six active task ids from the original TASK_CONFIG literal."""
    active = {"refinement", "completion", "ugen", "gen_t", "gen_ts", "gen_r"}
    values = _task_config_values("task_id")
    selected = {task: value for task, value in values.items() if task in active}
    if not all(isinstance(value, int) for value in selected.values()):
        raise AssertionError("Original task ids must be integer literals")
    return {task: cast(int, value) for task, value in selected.items()}


def _compiled_classes(
    relative_path: str,
    class_names: tuple[str, ...],
    namespace: dict[str, object],
) -> dict[str, type]:
    tree = _source_tree(relative_path)
    class_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    if {node.name for node in class_nodes} != set(class_names):
        raise AssertionError(f"Missing original classes in {relative_path}")
    module = ast.Module(body=cast(list[ast.stmt], class_nodes), type_ignores=[])
    exec(compile(module, str(REFERENCE_SRC / relative_path), "exec"), namespace)
    return {name: cast(type, namespace[name]) for name in class_names}


def reference_serialization(
    recipe: LayoutFormerPPTrainingRecipe,
    task: str,
    *,
    labels: list[int],
    bboxes: list[list[int]],
    relations: list[tuple[int, int, int, int, int]],
    input_labels: list[int] | None = None,
    input_bboxes: list[list[int]] | None = None,
    input_relations: list[tuple[int, int, int, int, int]] | None = None,
) -> dict[str, bytes | int | str]:
    """Execute original serializer classes for one active recipe task."""
    base_class = _compiled_classes(
        "tasks/refinement.py",
        ("T5LayoutSequence",),
        {
            "LayoutSequence": _ReferenceLayoutSequence,
            "List": list,
            "Tuple": tuple,
            "re": __import__("re"),
        },
    )["T5LayoutSequence"]
    gen_t_class = _compiled_classes(
        "tasks/gen_t.py",
        ("T5LayoutSequenceForGenT",),
        {"T5LayoutSequence": base_class},
    )["T5LayoutSequenceForGenT"]
    relation_classes = _compiled_classes(
        "tasks/gen_r.py",
        ("RelationTypes", "T5LayoutSequenceForGenR"),
        {
            "T5LayoutSequence": base_class,
            "transforms": SimpleNamespace(
                DiscretizeBoundingBox=lambda *args, **kwargs: object()
            ),
        },
    )
    tokenizer = package_tokenizer(recipe)
    index2label = reference_label_map(recipe)
    label2index = {value: key for key, value in index2label.items()}
    add_sep_token = "add_sep_token" in recipe.serialization_flags
    base = base_class(tokenizer, index2label, label2index, add_sep_token)
    label_tensor = torch.tensor(labels)
    bbox_tensor = torch.tensor(bboxes)
    input_label_tensor = torch.tensor(labels if input_labels is None else input_labels)
    input_bbox_tensor = torch.tensor(bboxes if input_bboxes is None else input_bboxes)
    relation_tensor = torch.tensor(
        relations if input_relations is None else input_relations
    )
    if task == "refinement":
        input_text = base.build_seq(input_label_tensor, input_bbox_tensor)
    elif task == "completion":
        input_text = base.build_seq(input_label_tensor[:1], input_bbox_tensor[:1])
    elif task == "ugen":
        input_text = ""
    elif task in {"gen_t", "gen_ts"}:
        serializer = gen_t_class(
            task,
            tokenizer,
            index2label,
            label2index,
            add_sep_token=add_sep_token,
            gen_t_add_unk_token=("gen_t_add_unk_token" in recipe.serialization_flags),
            gen_ts_add_unk_token=("gen_ts_add_unk_token" in recipe.serialization_flags),
        )
        input_text = serializer.build_input_seq(input_label_tensor, input_bbox_tensor)
    elif task == "gen_r":
        relation_class = relation_classes["T5LayoutSequenceForGenR"]
        serializer = relation_class(
            tokenizer,
            index2label,
            label2index,
            add_sep_token=add_sep_token,
            discrete_x_grid=recipe.discrete_x_grid,
            discrete_y_grid=recipe.discrete_y_grid,
            gen_r_add_unk_token=("gen_r_add_unk_token" in recipe.serialization_flags),
            gen_r_compact="gen_r_compact" in recipe.serialization_flags,
        )
        input_text = serializer.build_input_seq(label_tensor, relation_tensor)
    else:
        raise AssertionError(f"Unsupported active task: {task}")
    input_text = input_text.lower().strip()
    output_text = base.build_seq(label_tensor, bbox_tensor).lower().strip()
    prompt = str(_task_config_values("prompt")[task]).lower()
    if "add_task_prompt" in recipe.serialization_flags:
        input_text = f"{prompt} {input_text}"
    return {
        "input_bytes": input_text.encode("utf-8"),
        "output_bytes": output_text.encode("utf-8"),
        "prompt": prompt,
        "task_id": reference_task_ids()[task],
    }


def _deepspeed_pin() -> str:
    requirements = (REFERENCE_ROOT / "LayoutFormer++/requirements.txt").read_text(
        encoding="utf-8"
    )
    for line in requirements.splitlines():
        if line.startswith("deepspeed=="):
            return line.partition("==")[2]
    raise AssertionError("Original requirements do not pin deepspeed")


def reference_optimizer_scheduler_state(
    warmup_steps: tuple[int, ...],
) -> _OptimizerSchedulerState:
    """Construct the pinned original Adam/WarmupLR in an isolated uv overlay."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for original DeepSpeed construction")
    pin = _deepspeed_pin()
    script = textwrap.dedent(
        f"""
        import hashlib
        import importlib.metadata
        import importlib.util
        import json
        import sys
        import types
        import torch

        path = importlib.metadata.distribution("deepspeed").locate_file(
            "deepspeed/runtime/lr_schedules.py"
        )
        deepspeed_module = types.ModuleType("deepspeed")
        runtime_module = types.ModuleType("deepspeed.runtime")
        constants_module = types.ModuleType("deepspeed.runtime.constants")
        utils_module = types.ModuleType("deepspeed.utils")
        utils_module.logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
        deepspeed_module.runtime = runtime_module
        sys.modules.update({{
            "deepspeed": deepspeed_module,
            "deepspeed.runtime": runtime_module,
            "deepspeed.runtime.constants": constants_module,
            "deepspeed.utils": utils_module,
        }})
        spec = importlib.util.spec_from_file_location(
            "deepspeed.runtime.lr_schedules", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        sequences = {{}}
        post_optimizer_steps = {{}}
        optimizer_defaults = None
        optimizer_name = None
        scheduler_name = None
        for warmup_steps in {list(warmup_steps)!r}:
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.Adam([parameter], lr=1e-4)
            scheduler = module.WarmupLR(
                optimizer,
                warmup_max_lr=1e-4,
                warmup_num_steps=warmup_steps,
            )
            values = [optimizer.param_groups[0]["lr"]]
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])
            sequences[str(warmup_steps)] = values
            step_parameter = torch.nn.Parameter(torch.tensor(1.0))
            step_optimizer = torch.optim.Adam([step_parameter], lr=1e-4)
            step_scheduler = module.WarmupLR(
                step_optimizer,
                warmup_max_lr=1e-4,
                warmup_num_steps=warmup_steps,
            )
            step_parameter.grad = torch.ones_like(step_parameter)
            step_optimizer.step()
            step_scheduler.step()
            post_optimizer_steps[str(warmup_steps)] = {{
                "optimizer_step": float(step_optimizer.state[step_parameter]["step"]),
                "last_batch_iteration": float(step_scheduler.last_batch_iteration),
                "lr": float(step_optimizer.param_groups[0]["lr"]),
            }}
            optimizer_name = type(optimizer).__module__ + "." + type(optimizer).__name__
            scheduler_name = type(scheduler).__module__ + "." + type(scheduler).__name__
            optimizer_defaults = {{
                "lr": optimizer.defaults["lr"],
                "betas": list(optimizer.defaults["betas"]),
                "eps": optimizer.defaults["eps"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "amsgrad": optimizer.defaults["amsgrad"],
            }}
        print(json.dumps({{
            "deepspeed_version": importlib.metadata.version("deepspeed"),
            "deepspeed_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "optimizer_defaults": optimizer_defaults,
            "lr_sequences": sequences,
            "post_optimizer_steps": post_optimizer_steps,
        }}, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [uv, "run", "--with", f"deepspeed=={pin}", "python", "-c", script],
        cwd=PROJECT_ROOT,
        env={**os.environ, "UV_NO_PROGRESS": "1", "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(_OptimizerSchedulerState, json.loads(completed.stdout))


def _compiled_method(
    relative_path: str,
    class_name: str,
    method_name: str,
    namespace: dict[str, object],
):
    tree = _source_tree(relative_path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(REFERENCE_SRC / relative_path),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def _checkpoint_expression() -> tuple[str, bool, bool]:
    tree = _source_tree("trainer/multitask_trainer.py")
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MultiTaskTrainer"
    )
    call = next(
        node
        for node in ast.walk(trainer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "save"
    )
    expression = ast.unparse(call.args[0])
    text = ast.unparse(trainer)
    return expression, "optimizer.state_dict" in text, "scheduler.state_dict" in text


def _reference_device_case(
    cuda_available: bool, device_count: int
) -> dict[str, object]:
    class FakeModel:
        data_parallel = False

        def to(self, device: str):
            self.device = device
            return self

    class FakeDataParallel(FakeModel):
        def __init__(self, model: FakeModel) -> None:
            self.model = model
            self.data_parallel = True

    fake_cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        device=lambda value: value,
    )
    setup_model = _compiled_method(
        "trainer/basic_trainer.py",
        "Trainer",
        "_setup_model",
        {"torch": fake_torch, "nn": SimpleNamespace(DataParallel=FakeDataParallel)},
    )
    owner = SimpleNamespace(model=FakeModel())
    setup_model(owner)
    return {
        "device": owner.device,
        "data_parallel": bool(owner.model.data_parallel),
    }


def reference_checkpoint_behavior() -> dict[str, object]:
    """Execute original checkpoint selection, pointer, and device branches."""
    measurement = _compiled_classes(
        "trainer/utils.py",
        ("CheckpointMeasurement",),
        {"np": np, "torch": torch, "List": list, "Tuple": tuple},
    )["CheckpointMeasurement"](20, "eval_loss")
    updates = [measurement.update(value) for value in (5.0, 4.0, 6.0)]

    writes: dict[str, str] = {}

    class CapturedFile:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def write(self, value: str) -> None:
            writes[self.path] = value

    checkpoint = _compiled_method(
        "trainer/multitask_trainer.py",
        "MultiTaskTrainer",
        "do_checkpointing",
        {
            "os": os,
            "json": json,
            "open": lambda path, mode: CapturedFile(path),
        },
    )
    owner = SimpleNamespace(args=SimpleNamespace(out_dir="captured"))
    best_miou = {"gen_t": {"epoch": 7, "value": 0.5}}
    checkpoint(owner, 7, True, best_miou)
    best_epoch_path = os.path.join("captured", "best_epoch")
    best_miou_path = os.path.join("captured", "best_miou.json")
    before_non_best = dict(writes)
    checkpoint(owner, 8, False, {"gen_t": {"epoch": 8, "value": 0.4}})

    get_trainer_tree = _source_tree("trainer/__init__.py")
    get_trainer_node = next(
        node
        for node in get_trainer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_trainer"
    )
    trainer_namespace = {
        "Trainer": object(),
        "MultiTaskTrainer": object(),
        "DSMultiTaskTrainer": object(),
    }
    exec(
        compile(
            ast.Module(body=[get_trainer_node], type_ignores=[]),
            str(REFERENCE_SRC / "trainer/__init__.py"),
            "exec",
        ),
        trainer_namespace,
    )
    get_trainer = cast(Callable[[object], object], trainer_namespace["get_trainer"])
    modes = []
    for mode in ("basic", "deepspeed"):
        get_trainer(SimpleNamespace(trainer=mode))
        modes.append(mode)
    try:
        get_trainer(SimpleNamespace(trainer="ddp"))
    except NotImplementedError:
        ddp_rejected = True
    else:
        ddp_rejected = False

    expression, has_optimizer, has_scheduler = _checkpoint_expression()
    return {
        "eval_loss_updates": updates,
        "best_eval_loss": float(measurement.best_value),
        "best_epoch_written": writes[best_epoch_path],
        "best_miou_written": json.loads(writes[best_miou_path]),
        "non_best_preserves_epoch": writes == before_non_best,
        "epoch_checkpoint_expression": expression,
        "checkpoint_has_optimizer_state": has_optimizer,
        "checkpoint_has_scheduler_state": has_scheduler,
        "basic_device_cases": {
            "cpu": _reference_device_case(False, 0),
            "cuda_single": _reference_device_case(True, 1),
            "cuda_multi": _reference_device_case(True, 2),
        },
        "trainer_modes": modes,
        "ddp_rejected": ddp_rejected,
    }


def reference_label_map(recipe: LayoutFormerPPTrainingRecipe) -> dict[int, str]:
    """Read the original dataset label order directly from its class literal."""
    class_name = "RicoDataset" if recipe.dataset == "rico25" else "PubLayNetDataset"
    labels = _class_literal("data/base.py", class_name, "labels")
    if not isinstance(labels, list) or not all(
        isinstance(item, str) for item in labels
    ):
        raise AssertionError(f"Invalid {class_name}.labels literal")
    return {index: str(label) for index, label in enumerate(labels, start=1)}


def reference_tokens(recipe: LayoutFormerPPTrainingRecipe) -> list[str]:
    """Reproduce ``create_tokenizer`` from original source literals only."""
    label_map = reference_label_map(recipe)
    tokens = [f"label_{index}" for index in label_map]
    tokens.extend(map(str, range(recipe.discrete_x_grid)))
    sep_token = _class_literal("tasks/refinement.py", "T5LayoutSequence", "SEP_TOKEN")
    if "add_sep_token" in recipe.serialization_flags:
        tokens.append(str(sep_token))
    if any(str(item) == "gen_r" for item in recipe.tasks):
        tokens.append("label_0")
        relation_types = _class_literal("tasks/gen_r.py", "RelationTypes", "types")
        if not isinstance(relation_types, list):
            raise AssertionError("Invalid RelationTypes.types literal")
        tokens.extend(f"relation_{index}" for index, _ in enumerate(relation_types))
        tokens.extend(f"index_{index}" for index in range(1, 21))
        for attribute in ("REL_BEG_TOKEN", "REL_SEP_TOKEN", "REL_ELE_SEP_TOKEN"):
            tokens.append(
                str(
                    _class_literal(
                        "tasks/gen_r.py",
                        "T5LayoutSequenceForGenR",
                        attribute,
                    )
                )
            )
    if "add_task_prompt" in recipe.serialization_flags:
        prompt_tokens: list[str] = []
        for prompt in _task_prompts():
            for token in prompt.split():
                if token not in prompt_tokens:
                    prompt_tokens.append(token)
        tokens.extend(prompt_tokens)
    return tokens


def package_tokenizer(recipe: LayoutFormerPPTrainingRecipe) -> LayoutFormerPPTokenizer:
    """Construct the package tokenizer for one static recipe."""
    return LayoutFormerPPTokenizer(tokens=reference_tokens(recipe))


def reference_classes() -> tuple[type, type]:
    """Load model and tokenizer classes directly without dataset imports."""
    model_module = _load_module(
        "layoutformerpp_s0_reference_model",
        REFERENCE_SRC / "model/layout_transformer/model.py",
    )
    tokenizer_module = _load_module(
        "layoutformerpp_s0_reference_tokenizer",
        REFERENCE_SRC / "model/layout_transformer/tokenizer.py",
    )
    return model_module.LayoutTransformer, tokenizer_module.LayoutTransformerTokenizer


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_arguments(recipe: LayoutFormerPPTrainingRecipe) -> dict[str, str | bool]:
    """Parse the effective long options from one released shell recipe."""
    task = str(recipe.condition)
    task_slug = {
        "label": "gen_t",
        "label_size": "gen_ts",
        "relation": "gen_r",
        "unconditional": "ugen",
    }.get(task, task)
    dataset_slug = "rico" if recipe.dataset == "rico25" else "publaynet"
    path = REFERENCE_SRC / "scripts" / f"{dataset_slug}_{task_slug}.sh"
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    command = text[text.index("$COMMAND main.py") :]
    tokens = shlex.split(command)
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token.removeprefix("--")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            parsed[key] = tokens[index + 1].removeprefix("\\")
            index += 2
        else:
            parsed[key] = True
            index += 1
    return parsed


def _reference_loader_args(
    recipe: LayoutFormerPPTrainingRecipe, data_root: Path
) -> SimpleNamespace:
    """Build the original basic-trainer arguments from the pinned recipe script."""
    parsed = script_arguments(recipe)
    args = SimpleNamespace(
        dataset="rico" if recipe.dataset == "rico25" else "publaynet",
        tasks=parsed["tasks"],
        eval_tasks=parsed.get("eval_tasks"),
        data_dir=str(data_root),
        max_num_elements=recipe.max_num_elements,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.01,
        train_bernoulli_beta=1.0,
        discrete_x_grid=recipe.discrete_x_grid,
        discrete_y_grid=recipe.discrete_y_grid,
        sort_by_dict=True,
        add_sep_token=True,
        add_task_prompt=False,
        task_weights=None,
        partition_training_data=False,
        partition_training_data_task_buckets=None,
        fine_grained_partition_training_data=False,
        fine_grained_partition_training_data_task_size=None,
        single_task_per_batch=False,
        remove_too_long_layout=False,
        gen_t_add_unk_token=False,
        gen_ts_add_unk_token=False,
        gen_r_compact=False,
        gen_r_add_unk_token=False,
        gen_r_discrete_before_induce_relations=False,
        gen_r_shuffle_before_sort_by_label=False,
        gen_r_sort_by_pos_before_sort_by_label=False,
        refinement_shuffle_before_sort_by_label=False,
        refinement_sort_by_pos_before_sort_by_label=False,
        completion_sort_by_pos=False,
        completion_shuffle_before_sort_by_label=False,
        completion_sort_by_pos_before_sort_by_label=False,
        ugen_sort_by_pos=False,
        ugen_shuffle_before_sort_by_label=False,
        ugen_sort_by_pos_before_sort_by_label=False,
        gen_t_shuffle_before_sort_by_label=False,
        gen_t_sort_by_pos_before_sort_by_label=False,
        gen_ts_shuffle_before_sort_by_label=False,
        gen_ts_sort_by_pos_before_sort_by_label=False,
    )
    for key, value in parsed.items():
        if key in {"${MODE}", "data_dir", "out_dir"}:
            continue
        if isinstance(value, bool):
            setattr(args, key, value)
        elif key in {"tasks", "eval_tasks", "partition_training_data_task_buckets"}:
            setattr(args, key, str(value))
        elif key in {
            "max_num_elements",
            "batch_size",
            "eval_batch_size",
            "discrete_x_grid",
            "discrete_y_grid",
            "warmup_num_steps",
            "eval_seed",
            "eval_interval",
        }:
            setattr(args, key, int(str(value)))
        elif key in {
            "gaussian_noise_mean",
            "gaussian_noise_std",
            "train_bernoulli_beta",
        }:
            setattr(args, key, float(str(value)))
    args.batch_size = recipe.batch_size
    args.eval_batch_size = recipe.eval_batch_size
    args.partition_training_data = bool(recipe.partition_buckets)
    if recipe.partition_buckets:
        args.partition_training_data_task_buckets = ",".join(
            str(bucket) for bucket in recipe.partition_buckets
        )
    return args


def reference_loader_stream(
    recipe: LayoutFormerPPTrainingRecipe,
    data_root: Path,
    split: str,
    *,
    batches: int = 2,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Iterate the pinned original dataset and basic DataLoader for one split."""
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported LayoutFormer++ loader split: {split}")
    vendor_source = str(REFERENCE_SRC)
    if vendor_source not in sys.path:
        sys.path.insert(0, vendor_source)
    from tasks.task_config import TASK_CONFIG
    from tasks.task_utils import create_dataset, create_tokenizer
    from torch.utils.data import DataLoader
    from utils import utils

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = _reference_loader_args(recipe, data_root)
    task_names = tuple(str(task) for task in recipe.tasks)
    tokenizer = create_tokenizer(
        list(task_names),
        str(args.dataset),
        recipe.discrete_x_grid,
        add_sep_token=True,
        add_task_prompt=bool(args.add_task_prompt),
    )
    dataset = create_dataset(
        args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split=split,
        sort_by_pos=not args.sort_by_dict,
    )
    if split == "val":
        dataset.switch_task(str(recipe.eval_tasks[0]))
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size if split == "train" else args.eval_batch_size,
        collate_fn=utils.collate_fn,
        drop_last=True,
        shuffle=split == "train",
    )
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader):
        input_text = [str(value) for value in batch["in_str"]]
        output_text = [str(value) for value in batch["out_str"]]
        input_encoding = tokenizer(input_text, add_eos=True, add_bos=False)
        target_encoding = tokenizer(output_text, add_eos=True, add_bos=False)
        rows.append(
            {
                "names": tuple(str(value) for value in batch["name"]),
                "task_names": tuple(str(value) for value in batch["task_name"]),
                "task_ids": tuple(int(value) for value in batch["task_id"]),
                "input_bytes": tuple(value.encode("utf-8") for value in input_text),
                "output_bytes": tuple(value.encode("utf-8") for value in output_text),
                "input_ids": input_encoding["input_ids"].long(),
                "attention_mask": input_encoding["mask"].bool(),
                "labels": target_encoding["input_ids"].long(),
                "target_mask": target_encoding["mask"].bool(),
            }
        )
        if batch_index + 1 == batches:
            break
    if len(rows) != batches:
        raise AssertionError(
            f"{recipe.name} {split} original loader yielded {len(rows)} batches; "
            f"expected {batches}"
        )
    return rows


def source_facts() -> dict[str, bool | str]:
    """Record construction-order and trainer-branch facts from source text."""
    main_text = (REFERENCE_SRC / "main.py").read_text(encoding="utf-8")
    trainer_text = (REFERENCE_SRC / "trainer/__init__.py").read_text(encoding="utf-8")
    basic_text = (REFERENCE_SRC / "trainer/basic_trainer.py").read_text(
        encoding="utf-8"
    )
    multitask_text = (REFERENCE_SRC / "trainer/multitask_trainer.py").read_text(
        encoding="utf-8"
    )
    measurement_text = (REFERENCE_SRC / "trainer/utils.py").read_text(encoding="utf-8")
    ds_text = (REFERENCE_SRC / "trainer/ds_trainer.py").read_text(encoding="utf-8")
    main_tree = ast.parse(main_text)
    imported_names = {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in main_tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
        for alias in node.names
    }
    train_node = next(
        node
        for node in main_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    called_names = {
        node.func.id
        for node in ast.walk(train_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    optimizer_name = imported_names.get("Adam", "")
    scheduler_name = imported_names.get("WarmupLR", "")
    return {
        "model_before_trainer": main_text.index("model = build_model")
        < main_text.index("trainer = trainer_class"),
        "seed_inside_trainer_setup": "utils.init_experiment" in basic_text,
        "ddp_training_rejected": "NotImplementedError" in trainer_text,
        "basic_supported": "basic" in trainer_text,
        "deepspeed_supported": "deepspeed" in trainer_text,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "optimizer_call_found": "Adam" in called_names,
        "scheduler_call_found": "WarmupLR" in called_names,
        "scheduler_after_optimizer_update": multitask_text.index(
            "self.optimizer.step()"
        )
        < multitask_text.index("self.scheduler.step()"),
        "basic_device_cpu_fallback": 'else "cpu"' in basic_text,
        "basic_device_cuda_zero": 'torch.device("cuda:0"' in basic_text,
        "basic_multigpu_data_parallel": "nn.DataParallel(self.model)" in basic_text,
        "basic_multigpu_scales_batch_size": (
            "self.args.batch_size * torch.cuda.device_count()" in multitask_text
            and "self.args.eval_batch_size * torch.cuda.device_count()"
            in multitask_text
        ),
        "deepspeed_initializes_distributed": "deepspeed.init_distributed" in ds_text,
        "checkpoint_selector": "min_eval_loss_best_epoch",
        "checkpoint_measure_is_eval_loss": (
            "checkpoint_measure=CheckpointMeasurement.EVAL_LOSS" in main_text
            and "self._measurement == self.EVAL_LOSS" in measurement_text
            and "measure_value < self._best_value" in measurement_text
            and "'best_epoch'" in multitask_text
        ),
        "checkpoint_payload_is_model_state_only": (
            "torch.save(self.model.state_dict(), normal_ckpt_path)" in multitask_text
            and "optimizer.state_dict" not in multitask_text
            and "scheduler.state_dict" not in multitask_text
        ),
        "multitask_writes_every_epoch_state": (
            "f'epoch_{epoch}_checkpoint.pth.tar'" in multitask_text
            and "torch.save(self.model.state_dict(), normal_ckpt_path)"
            in multitask_text
        ),
        "multitask_best_epoch_is_external_pointer": (
            "'best_epoch'" in multitask_text and "shutil.copy" not in multitask_text
        ),
        "amp_disabled": (
            "autocast" not in main_text + basic_text + multitask_text
            and "GradScaler" not in main_text + basic_text + multitask_text
        ),
        "ema_disabled": "ema" not in (main_text + basic_text + multitask_text).lower(),
    }
