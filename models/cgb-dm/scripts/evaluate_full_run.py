"""Evaluate CGB-DM full-run checkpoints with the original metric formulas."""

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
import yaml
from cgb_dm.configuration_cgb_dm import cgb_dm_config_for_dataset
from cgb_dm.conversion import build_pipeline_from_checkpoint
from jaxtyping import Bool, Float, Int
from PIL import Image
from torch.utils.data import DataLoader
from transformers import set_seed


@dataclass(frozen=True)
class DatasetEvalSpec:
    """Dataset-specific settings for the original S5 evaluation protocol."""

    config_name: str
    package_dataset_name: str
    fixture_dataset_name: str


DATASET_EVAL_SPECS = {
    "pku_posterlayout": DatasetEvalSpec(
        config_name="pku.yaml",
        package_dataset_name="pku_posterlayout",
        fixture_dataset_name="pku",
    ),
    "cgl": DatasetEvalSpec(
        config_name="cgl.yaml",
        package_dataset_name="cgl",
        fixture_dataset_name="cgl",
    ),
}


class ConfigPathsSplit(Protocol):
    """Reference split paths consumed by the metric implementation."""

    sal_dir: str
    sal_sub_dir: str
    inp_dir: str


class ConfigPaths(Protocol):
    """Reference path container consumed by the metric implementation."""

    test: ConfigPathsSplit


class EvalConfig(Protocol):
    """Dynamic reference config attributes used by this evaluator."""

    width: int
    height: int
    n_head: int
    d_model: int
    feature_dim: int
    num_class: int
    n_layers: int
    max_elem: int
    num_workers: int
    test_batch_size: int
    imgname_order_dir: str
    paths: ConfigPaths


class EvalProcessor(Protocol):
    """Processor attributes used by package-side sampling."""

    max_seq_length: int
    seq_dim: int
    num_labels: int


class SchedulerStep(Protocol):
    """Scheduler step output consumed by this evaluator."""

    prev_sample: Float[torch.Tensor, "batch elements features"]


class EvalScheduler(Protocol):
    """Scheduler surface used by package-side sampling."""

    timesteps: Float[torch.Tensor, "steps"] | Int[torch.Tensor, "steps"]

    def set_timesteps(
        self, num_inference_steps: int | None, *, device: torch.device
    ) -> None:
        """Initialize inference timesteps."""

    def initial_sample(
        self,
        batch_size: int,
        max_seq_length: int,
        seq_dim: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> Float[torch.Tensor, "batch elements features"]:
        """Return the initial denoising sample."""

    def step(
        self,
        model_output: Float[torch.Tensor, "batch elements features"],
        timestep: Int[torch.Tensor, "batch"],
        sample: Float[torch.Tensor, "batch elements features"],
        _step_index: int,
        *,
        generator: torch.Generator,
    ) -> SchedulerStep:
        """Run one scheduler step."""


class ModelOutput(Protocol):
    """Model output consumed by scheduler sampling."""

    sample: Float[torch.Tensor, "batch elements features"]


class EvalModel(Protocol):
    """Model call surface used by package-side sampling."""

    def __call__(
        self,
        sample: Float[torch.Tensor, "batch elements features"],
        image: Float[torch.Tensor, "batch channels height width"],
        sal_box: Float[torch.Tensor, "batch ..."],
        timestep: Int[torch.Tensor, "batch"],
    ) -> ModelOutput:
        """Return predicted denoising sample."""


class EvalPipeline(Protocol):
    """Pipeline surface used for package-side evaluation."""

    model: EvalModel
    scheduler: EvalScheduler
    processor: EvalProcessor


class DiffusionModel(Protocol):
    """Reference diffusion surface used for original-side evaluation."""

    model: torch.nn.Module

    def reverse_ddim(
        self,
        image: Float[torch.Tensor, "batch channels height width"],
        sal_box: Float[torch.Tensor, "batch ..."],
        cfg: EvalConfig,
        *,
        save_inter: bool,
    ) -> tuple[
        Float[torch.Tensor, "batch elements 4"],
        Int[torch.Tensor, "batch elements 1"],
        Float[torch.Tensor, "..."] | None,
    ]:
        """Return generated boxes and classes."""
        del save_inter
        raise NotImplementedError


class Cv2Module(Protocol):
    """Small OpenCV surface required for unreadability metrics."""

    COLOR_RGB2GRAY: int

    def setNumThreads(self, count: int) -> None:
        """Set OpenCV thread count."""

    def cvtColor(
        self, image: Float[np.ndarray, "height width channels"], code: int
    ) -> Float[np.ndarray, "height width"]:
        """Convert an image between color spaces."""

    def Sobel(
        self,
        image: Float[np.ndarray, "height width"],
        _ddepth: int,
        _dx: int,
        _dy: int,
    ) -> Float[np.ndarray, "height width"]:
        """Compute one Sobel derivative."""


class MetricRecord(TypedDict):
    """One seeded CGB-DM metric record."""

    seed: int
    num_samples: int
    metrics: dict[str, float]


class MetricSummary(TypedDict):
    """CGB-DM metric mean/std summary."""

    runs: list[MetricRecord]
    mean: dict[str, float]
    std: dict[str, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_EVAL_SPECS),
        default="pku_posterlayout",
        help="Dataset/protocol to evaluate.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Extracted dataset split root matching --dataset.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics.")
    parser.add_argument("--backend", choices=["ours", "reference"], default="ours")
    parser.add_argument("--gpu", type=int, default=0, help="Visible CUDA device index.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    """Run generation and metric evaluation."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    reference_root = repo_root / "vendor" / "layout-dit"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(reference_root))
    dataloader_mod = importlib.import_module("data_process.dataloader")
    util_mod = importlib.import_module("utils.util")

    cv2 = cast(Cv2Module, importlib.import_module("cv2"))
    cv2.setNumThreads(1)
    torch.set_num_threads(8)

    dataset_spec = DATASET_EVAL_SPECS[args.dataset]
    config_path = reference_root / "configs" / dataset_spec.config_name
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    raw_config["paths"]["base"] = str(Path(args.data_root).resolve())
    raw_config["imgname_order_dir"] = str(output_dir / "image_name_order")
    Path(raw_config["imgname_order_dir"]).mkdir(parents=True, exist_ok=True)
    cfg = cast(EvalConfig, util_mod.Config(util_mod.process_paths(raw_config)))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    pipe: EvalPipeline | None = None
    diffusion_model: DiffusionModel | None = None
    if args.backend == "ours":
        pipe = cast(
            EvalPipeline,
            build_pipeline_from_checkpoint(
                args.checkpoint,
                config=cgb_dm_config_for_dataset(dataset_spec.package_dataset_name),
            ).to(device),
        )
    else:
        diffusion_cls = importlib.import_module("cgbdm.diffusion").Diffusion
        diffusion_model = cast(
            DiffusionModel,
            diffusion_cls(
                num_timesteps=1000,
                ddim_num_steps=100,
                n_head=cfg.n_head,
                dim_model=cfg.d_model,
                feature_dim=cfg.feature_dim,
                seq_dim=cfg.num_class + 4,
                num_layers=cfg.n_layers,
                device=device,
                max_elem=cfg.max_elem,
            ),
        )
        model_weights = torch.load(args.checkpoint, map_location=device)
        diffusion_model.model.load_state_dict(model_weights)
        diffusion_model.model.eval()

    results: list[MetricRecord] = []
    for seed in args.seeds:
        set_seed(seed)
        cfg.imgname_order_dir = str(
            output_dir
            / "image_name_order"
            / f"seed_{seed}_{dataset_spec.fixture_dataset_name}_unanno_test.pt"
        )
        testing_set = dataloader_mod.test_uncond_dataset(cfg)
        testing_dl = DataLoader(
            testing_set,
            num_workers=cfg.num_workers,
            batch_size=getattr(cfg, "batch_size", cfg.test_batch_size),
            shuffle=False,
        )
        test_output = _generate_samples(
            backend=args.backend,
            pipe=pipe,
            diffusion_model=diffusion_model,
            testing_dl=testing_dl,
            cfg=cfg,
            device=device,
            seed=seed,
        )
        img_names = torch.load(cfg.imgname_order_dir, weights_only=False)
        img_names = img_names[: test_output.shape[0]]
        metrics = compute_metrics(img_names, test_output, cfg, cv2=cv2)
        record: MetricRecord = {
            "seed": seed,
            "num_samples": int(test_output.shape[0]),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        torch.save(test_output, output_dir / f"{args.backend}_seed_{seed}_samples.pt")
        (output_dir / f"{args.backend}_seed_{seed}_metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(record)

    summary = _summarize(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _generate_samples(
    *,
    backend: str,
    pipe: EvalPipeline | None,
    diffusion_model: DiffusionModel | None,
    testing_dl: DataLoader[
        tuple[
            Float[torch.Tensor, "batch channels height width"],
            Float[torch.Tensor, "batch ..."],
        ]
    ],
    cfg: EvalConfig,
    device: torch.device,
    seed: int,
) -> Float[torch.Tensor, "samples elements features"]:
    sample_output: list[Float[torch.Tensor, "batch elements features"]] = []
    count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for image, sal_box in testing_dl:
            image = image.to(device)
            sal_box = sal_box.to(device)
            if backend == "ours":
                if pipe is None:
                    raise ValueError("Package pipeline is required for backend='ours'")
                pipe.scheduler.set_timesteps(None, device=device)
                sample = pipe.scheduler.initial_sample(
                    image.shape[0],
                    pipe.processor.max_seq_length,
                    pipe.processor.seq_dim,
                    device=device,
                    generator=generator,
                )
                for index, timestep in enumerate(pipe.scheduler.timesteps):
                    timestep_batch = torch.full(
                        (image.shape[0],),
                        int(timestep.item()),
                        device=device,
                        dtype=torch.long,
                    )
                    model_out = pipe.model(sample, image, sal_box, timestep_batch)
                    step = pipe.scheduler.step(
                        model_out.sample,
                        timestep_batch,
                        sample,
                        len(pipe.scheduler.timesteps) - index - 1,
                        generator=generator,
                    )
                    sample = step.prev_sample
                labels = sample[:, :, : pipe.processor.num_labels].argmax(
                    dim=-1, keepdim=True
                )
                boxes = (
                    sample[:, :, pipe.processor.num_labels :].clamp(-1.0, 1.0) / 2 + 0.5
                )
                sample_output.append(torch.cat([labels, boxes], dim=2).cpu())
            else:
                if diffusion_model is None:
                    raise ValueError(
                        "Reference diffusion model is required for backend='reference'"
                    )
                bbox, cls, _ = diffusion_model.reverse_ddim(
                    image, sal_box, cfg, save_inter=False
                )
                sample_output.append(torch.cat([cls, bbox], dim=2).cpu())
            count += image.shape[0]
            print(f"{backend} seed {seed}: created {count} samples", flush=True)
    return torch.concat(sample_output, dim=0)


def compute_metrics(
    img_names: list[str],
    test_output: Float[torch.Tensor, "samples elements features"],
    cfg: EvalConfig,
    *,
    cv2: Cv2Module,
) -> dict[str, float]:
    """Compute the original validity, overlap, underlay, occlusion, and readability metrics."""
    metric_mod = importlib.import_module("utils.metric")
    util_mod = importlib.import_module("utils.util")

    print("Calculating metrics...", flush=True)
    clses, boxes = test_output[:, :, :1], test_output[:, :, 1:]
    boxes = torch.clamp(util_mod.box_cxcywh_to_xyxy(boxes), 0, 1)
    metrics = {"val": float(metric_mod.validity_cal(clses, boxes))}
    clses = metric_mod.getRidOfInvalid(clses, boxes)
    metrics["ove"] = float(metric_mod.overlap_cal(clses, boxes))
    undl, unds = metric_mod.underlay_cal(clses, boxes)
    metrics["undl"] = float(undl)
    metrics["unds"] = float(unds)

    boxes[:, :, ::2] *= cfg.width
    boxes[:, :, 1::2] *= cfg.height
    boxes = boxes.round().int()
    metrics["occ"] = float(_occlusion_cal(img_names, clses, boxes, cfg))
    metrics["rea"] = float(_unreadability_cal(img_names, clses, boxes, cfg, cv2=cv2))
    for key, value in metrics.items():
        print(f"{key}:{value:.6f}", flush=True)
    return metrics


def _box_mask(
    clses: Int[torch.Tensor, "elements ..."],
    boxes: Int[torch.Tensor, "elements 4"],
    *,
    height: int,
    width: int,
    class_id: int | None = None,
) -> Bool[np.ndarray, "height width"]:
    cls = np.asarray(clses.cpu(), dtype=int).reshape(-1)
    box = np.asarray(boxes.cpu(), dtype=int)
    selected = box[cls > 0] if class_id is None else box[cls == class_id]
    mask = np.zeros((height, width), dtype=bool)
    for left, top, right, bottom in selected:
        left = int(np.clip(left, 0, width))
        right = int(np.clip(right, 0, width))
        top = int(np.clip(top, 0, height))
        bottom = int(np.clip(bottom, 0, height))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _occlusion_cal(
    img_names: list[str],
    clses: Int[torch.Tensor, "samples elements ..."],
    boxes: Int[torch.Tensor, "samples elements 4"],
    cfg: EvalConfig,
) -> float:
    total = 0.0
    image_size = (cfg.width, cfg.height)
    for index, name in enumerate(img_names):
        sal_1 = np.asarray(
            Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L")
        )
        sal_2 = np.asarray(
            Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L")
        )
        sal_map = Image.fromarray(np.maximum(sal_1, sal_2)).resize(image_size)
        saliency = np.asarray(sal_map, dtype=np.float32) / 255.0
        mask = _box_mask(clses[index], boxes[index], height=cfg.height, width=cfg.width)
        covered = saliency[mask]
        if covered.size:
            total += float(covered.mean())
    return total / len(img_names)


def _unreadability_cal(
    img_names: list[str],
    clses: Int[torch.Tensor, "samples elements ..."],
    boxes: Int[torch.Tensor, "samples elements 4"],
    cfg: EvalConfig,
    *,
    cv2: Cv2Module,
) -> float:
    values: list[float] = []
    image_size = (cfg.width, cfg.height)
    for index, name in enumerate(img_names):
        image = Image.open(os.path.join(cfg.paths.test.inp_dir, name)).convert("RGB")
        image_npy = np.asarray(image.resize(image_size), dtype=np.float32)
        image_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(image_gray, -1, 1, 0)
        grad_y = cv2.Sobel(image_gray, -1, 0, 1)
        grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
        grad_max = float(np.max(grad_xy))
        if grad_max > 0:
            grad_xy = grad_xy / grad_max
        text_mask = _box_mask(
            clses[index], boxes[index], height=cfg.height, width=cfg.width, class_id=1
        )
        underlay_mask = _box_mask(
            clses[index], boxes[index], height=cfg.height, width=cfg.width, class_id=3
        )
        unreadability = grad_xy[text_mask & ~underlay_mask]
        values.append(float(unreadability.mean()) if unreadability.size else 0.0)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(results: list[MetricRecord]) -> MetricSummary:
    metric_keys = sorted(results[0]["metrics"]) if results else []
    summary: MetricSummary = {"runs": results, "mean": {}, "std": {}}
    for key in metric_keys:
        values = torch.tensor(
            [run["metrics"][key] for run in results], dtype=torch.float64
        )
        summary["mean"][key] = float(values.mean().item())
        summary["std"][key] = float(values.std(unbiased=False).item())
    return summary


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
