"""Replay and compare Flex-DM loader streams through both training paths."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[4]
_ID_KEY = "__s4_fingerprint"
_VENDOR = "f2bcc9f8b8958d4072f11f259bc61242bfe99d7c"
_PACKAGE_ROOTS = ("flex_dm", "traingen", "traingen_parity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worktree_provenance(
    root: Path = _ROOT, *, dry_run: bool = False
) -> dict[str, str | bool]:
    root = root.resolve()
    top_level = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    if root != top_level:
        raise RuntimeError("harness root does not match the git worktree top level")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not status
    if not clean and not dry_run:
        raise RuntimeError("stream capture requires a clean git worktree")

    return {
        "git_head_sha": head,
        "git_top_level": str(top_level),
        "git_worktree_clean": clean,
    }


def _package_module_paths(
    root: Path, modules: Mapping[str, ModuleType | None] | None = None
) -> dict[str, str]:
    root = root.resolve()
    loaded_modules = sys.modules if modules is None else modules
    paths = {}
    for name, module in sorted(loaded_modules.items()):
        if module is None or not any(
            name == package or name.startswith(f"{package}.")
            for package in _PACKAGE_ROOTS
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"package module {name} has no resolved file path")
        path = Path(module_file).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"package module {name} resolves outside git worktree: {path}"
            ) from error
        paths[name] = str(path)

    if "flex_dm" not in paths:
        raise RuntimeError("FlexDM package module was not imported")

    return paths


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _normalized(value):
    import numpy as np

    array = np.asarray(_array(value))
    if array.dtype.kind in "iu":
        return array.astype(np.int64)
    if array.dtype.kind == "f":
        return array.astype(np.float32)
    if array.dtype.kind == "b":
        return array.astype(np.bool_)
    return array


def _digest(digests: dict[str, object], key: str, value) -> None:
    array = _normalized(value)
    digest = digests.setdefault(key, hashlib.sha256())
    digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())


def _compare_mapping(digests: dict[str, object], name: str, reference, actual) -> None:
    if set(reference) != set(actual):
        raise AssertionError(f"{name} keys differ: {set(reference) ^ set(actual)}")

    for key in reference:
        expected = _normalized(reference[key])
        received = _normalized(actual[key])
        if expected.shape != received.shape:
            raise AssertionError(f"{name}/{key} differs")
        if not (expected == received).all():
            import numpy as np

            index = tuple(np.argwhere(expected != received)[0])
            raise AssertionError(
                f"{name}/{key} differs at {index}: "
                f"{expected[index]} != {received[index]}"
            )
        _digest(digests, f"{name}/{key}", expected)


def _write_indices(path: Path, indices: list[int]) -> None:
    path.write_text("".join(f"{index}\n" for index in indices), encoding="utf-8")


def _fingerprint_map(
    tf, data_dir: Path, dataset: str, split: str
) -> tuple[dict[bytes, int], list[str]]:
    from flex_dm.training.dataset import FlexDmTfrecordDataset

    reader = FlexDmTfrecordDataset(data_dir, dataset, split)
    mapping: dict[bytes, int] = {}
    ids = [reader.sample_id(index) for index in range(len(reader))]
    index = 0
    shards = sorted(data_dir.glob(f"{split}-*.tfrecord"))
    for shard in shards:
        records = tf.data.TFRecordDataset(str(shard)).batch(512)
        for raw_batch in records:
            fingerprints = tf.raw_ops.Fingerprint(
                data=raw_batch, method="farmhash64"
            ).numpy()
            for fingerprint in fingerprints:
                key = bytes(fingerprint)
                if key in mapping:
                    raise ValueError(f"duplicate source record fingerprint in {split}")
                mapping[key] = index
                index += 1

    if index != len(reader):
        raise ValueError(
            f"TensorFlow found {index} {split} records; package found {len(reader)}"
        )
    return mapping, ids


def _worker(args: argparse.Namespace) -> None:
    provenance = _worktree_provenance(dry_run=args.dry_run)
    import tensorflow as tf
    import torch
    from unittest.mock import patch

    if tf.__version__ != "2.15.1":
        raise RuntimeError(f"expected TensorFlow 2.15.1, got {tf.__version__}")
    if tf.config.list_physical_devices("GPU") or torch.cuda.is_available():
        raise RuntimeError("CPU-only stream checks found an available GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' for CPU-only stream checks")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError("PYTHONHASHSEED must be set before Python starts")
    if os.environ.get("TF_DETERMINISTIC_OPS") != "1":
        raise RuntimeError("TF_DETERMINISTIC_OPS=1 is required")
    tf.config.experimental.enable_op_determinism()
    from tensorflow.core.example import example_pb2, feature_pb2

    protobuf = ModuleType("tfrecord.example_pb2")
    for source in (feature_pb2, example_pb2):
        for name in (
            "BytesList",
            "FloatList",
            "Int64List",
            "Feature",
            "Features",
            "FeatureList",
            "FeatureLists",
            "Example",
            "SequenceExample",
        ):
            if hasattr(source, name):
                setattr(protobuf, name, getattr(source, name))
    sys.modules["tfrecord.example_pb2"] = protobuf

    vendor_path = _ROOT / "vendor" / "flex-dm"
    for path in (vendor_path, vendor_path / "src" / "mfp"):
        sys.path.insert(0, str(path))

    from mfp.data.spec import DataSpec
    from mfp.main import main as original_main
    from mfp.models import mfp as original_mfp
    from mfp.models.architecture.mask import get_seq_mask as original_seq_mask
    from mfp.models.mfp import MFP
    from flex_dm.data_specs import attribute_groups_for_dataset
    from flex_dm.masking import get_seq_mask as package_seq_mask
    from flex_dm.training import masking as package_masking
    from flex_dm.training.datamodule import FlexDmDataModule
    from training_reference import original_args

    package_module_paths = _package_module_paths(Path(str(provenance["git_top_level"])))

    asset_dir = Path(args.asset_dir)
    data_dir = asset_dir / "data" / args.dataset
    if (data_dir / args.dataset).is_dir():
        data_dir /= args.dataset
    original_config = original_args(args.dataset)
    method = str(original_config["masking_method"])
    fingerprints = {
        split: _fingerprint_map(tf, data_dir, args.dataset, split)
        for split in ("train", "val")
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {split: output_dir / f"{split}-indices.txt" for split in ("train", "val")}
    package_data = None
    package_iters = None
    package_ids = None
    if args.phase == "compare":
        package_data = FlexDmDataModule(
            data_dir=str(data_dir),
            dataset_name=args.dataset,
            batch_size=256,
            shuffle_train=True,
            train_order_path=str(streams["train"]),
            val_order_path=str(streams["val"]),
        )
        package_data.setup("fit")
        package_iters = {
            "train": iter(package_data.train_dataloader()),
            "val": iter(package_data.val_dataloader()),
        }
        package_ids = {
            split: [
                package_data.datasets[split].sample_id(index)
                for index in range(len(package_data.datasets[split]))
            ]
            for split in ("train", "val")
        }

    raw_tfrecord_dataset = tf.data.TFRecordDataset

    def mark_record(serialized):
        fingerprint = tf.raw_ops.Fingerprint(
            data=tf.expand_dims(serialized, 0), method="farmhash64"
        )
        return serialized, tf.squeeze(fingerprint, axis=0)

    def instrumented_tfrecord_dataset(*dataset_args, **dataset_kwargs):
        return raw_tfrecord_dataset(*dataset_args, **dataset_kwargs).map(mark_record)

    tf.data.TFRecordDataset = instrumented_tfrecord_dataset
    raw_parse = DataSpec.parse_fn

    def parse_with_fingerprint(spec, serialized, fingerprint):
        parsed = raw_parse(spec, serialized)
        parsed[_ID_KEY] = fingerprint
        return parsed

    DataSpec.parse_fn = parse_with_fingerprint
    digests: dict[str, object] = {}
    captured_indices: dict[str, list[int]] = {"train": [], "val": []}
    observed_ids: dict[str, list[str]] = {"train": [], "val": []}
    batch_counts = {"train": 0, "val": 0}

    def stream_batches(split: str, source, model, limit: int | None) -> None:
        expected_map, source_ids = fingerprints[split]
        expected_stream = (
            [int(value) for value in streams[split].read_text().splitlines()]
            if args.phase == "compare"
            else []
        )
        items = source.take(limit) if limit is not None else source
        offset = 0
        for batch in items:
            batch_indices = [expected_map[bytes(row)] for row in batch[_ID_KEY].numpy()]
            batch_ids = [source_ids[index] for index in batch_indices]
            if args.phase == "capture":
                captured_indices[split].extend(batch_indices)
                observed_ids[split].extend(batch_ids)
                offset += len(batch_indices)
                batch_counts[split] += 1
                if split == "train" and batch_counts[split] % 100 == 0:
                    print(
                        f"{args.phase} {split} completed {batch_counts[split]} batches",
                        flush=True,
                    )
                continue

            if batch_indices != expected_stream[offset : offset + len(batch_indices)]:
                raise AssertionError(
                    f"original {split} stream no longer reproduces capture"
                )
            if batch_ids != package_ids[split][offset : offset + len(batch_ids)]:
                raise AssertionError(
                    f"package {split} sample IDs do not replay the original"
                )
            package_batch = next(package_iters[split])
            fields = {
                key: value
                for key, value in batch.items()
                if key not in {_ID_KEY, "id", "uuid"}
            }
            _compare_mapping(digests, f"fields/{split}", fields, package_batch)
            length = batch["length"]
            maxlen = int(batch["left"].shape[1])
            _compare_mapping(
                digests,
                f"padding/{split}",
                {"valid": original_seq_mask(length, maxlen=maxlen)},
                {"valid": package_seq_mask(package_batch["length"], maxlen=maxlen)},
            )
            if split == "train":
                _compare_training_masks(
                    tf,
                    torch,
                    patch,
                    original_mfp,
                    package_masking,
                    batch,
                    package_batch,
                    model,
                    package_data.datasets["train"]._columns,
                    attribute_groups_for_dataset(args.dataset),
                    digests,
                )
            observed_ids[split].extend(batch_ids)
            offset += len(batch_indices)
            batch_counts[split] += 1
            if split == "train" and batch_counts[split] % 100 == 0:
                print(
                    f"{args.phase} {split} completed {batch_counts[split]} batches",
                    flush=True,
                )

        if args.phase == "compare":
            if offset != len(expected_stream):
                raise AssertionError(f"incomplete replay of original {split} stream")
        print(
            f"{args.phase} {split} completed {batch_counts[split]} batches",
            flush=True,
        )

    def intercept_fit(model, train_dataset, *fit_args, **fit_kwargs):
        del fit_args
        validation_dataset = fit_kwargs["validation_data"]
        stream_batches("train", train_dataset, model, args.batches)
        stream_batches("val", validation_dataset, model, None)
        return None

    MFP.fit = intercept_fit
    MFP.evaluate = lambda self, *call_args, **call_kwargs: []
    MFP.save_weights = lambda self, *call_args, **call_kwargs: None
    import importlib

    trainer = importlib.import_module("mfp.train")
    trainer.get_callbacks = lambda *call_args, **call_kwargs: []
    sys.argv = [
        "mfp",
        "--dataset_name",
        args.dataset,
        "--data_dir",
        str(data_dir),
        "--job-dir",
        str(output_dir / "original-cli-run"),
        "--seed",
        str(args.seed),
        "--batch_size",
        "256",
        "--masking_method",
        method,
        "--num_epochs",
        "500",
        "--validation_freq",
        "10",
        "--latent_dim",
        str(original_config["latent_dim"]),
        "--num_blocks",
        str(original_config["num_blocks"]),
        "--arch_type",
        str(original_config["arch_type"]),
        "--block_type",
        str(original_config["block_type"]),
        "--l2",
        str(original_config["l2"]),
        "--dropout",
        str(original_config["dropout"]),
        "--seq_type",
        str(original_config["seq_type"]),
        "--input_dtype",
        str(original_config["input_dtype"]),
        "--learning_rate",
        str(original_config["learning_rate"]),
        "--verbose",
        "0",
    ]
    original_cli = list(sys.argv)
    original_main()
    imported_paths = _package_module_paths(Path(str(provenance["git_top_level"])))
    for name, path in imported_paths.items():
        if name in package_module_paths and package_module_paths[name] != path:
            raise RuntimeError(f"package module {name} changed its resolved file path")
    package_module_paths.update(imported_paths)

    if args.phase == "capture":
        for split in streams:
            _write_indices(streams[split], captured_indices[split])
        manifest = {
            **provenance,
            "package_module_paths": package_module_paths,
            "dataset": args.dataset,
            "seed": args.seed,
            "batch_size": 256,
            "train_batches": args.batches,
            "validation_batches": batch_counts["val"],
            "sample_ids": {split: ids[:8] for split, ids in observed_ids.items()},
            "stream_sha256": {split: _sha256(path) for split, path in streams.items()},
            "original_cli": original_cli,
            "vendor_commit": _VENDOR,
            "tensorflow": tf.__version__,
            "determinism": {
                "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                "TF_DETERMINISTIC_OPS": os.environ["TF_DETERMINISTIC_OPS"],
                "enable_op_determinism": True,
                "training_seed_flag": args.seed,
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return

    report = {
        "status": "passed",
        "dataset": args.dataset,
        "seed": args.seed,
        "train_batches": batch_counts["train"],
        "validation_batches": batch_counts["val"],
        "compared_sha256": {key: digest.hexdigest() for key, digest in digests.items()},
        "package_module_paths": package_module_paths,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _compare_training_masks(
    tf,
    torch,
    patch,
    original_mfp,
    package_masking,
    original_batch,
    package_batch,
    model,
    package_columns,
    groups,
    digests,
) -> None:
    inputs = {key: original_batch[key] for key in model.input_columns}
    tasks = model.task_cat_dist.sample(tf.shape(inputs["left"])[0])
    draws = []
    raw_uniform, raw_normal = tf.random.uniform, tf.random.normal

    def traced_uniform(shape, *args, **kwargs):
        value = raw_uniform(shape, *args, **kwargs)
        draws.append(
            (
                "uniform_int" if value.dtype.is_integer else "uniform_float",
                tuple(value.shape),
                value.numpy(),
            )
        )
        return value

    def traced_normal(shape, *args, **kwargs):
        value = raw_normal(shape, *args, **kwargs)
        draws.append(("normal", tuple(value.shape), value.numpy()))
        return value

    with (
        patch.object(tf.random, "uniform", traced_uniform),
        patch.object(tf.random, "normal", traced_normal),
    ):
        targets, modified, masks = original_mfp.preprocess_for_train(
            inputs, model.input_columns, tasks, input_dtype=model.input_dtype
        )

    draw_index = 0

    def replay(operation, size, *, device, dtype):
        nonlocal draw_index
        shape = (
            tuple(size[0])
            if len(size) == 1 and isinstance(size[0], tuple)
            else tuple(size)
        )
        expected_operation, expected_shape, values = draws[draw_index]
        if (operation, shape) != (expected_operation, expected_shape):
            raise AssertionError("package masking draw order differs from reference")
        draw_index += 1
        if operation == "normal":
            values = values / 0.1
        return torch.as_tensor(values, device=device, dtype=dtype)

    def replay_rand(*size, device=None, dtype=None, generator=None):
        del generator
        return replay("uniform_float", size, device=device, dtype=dtype)

    def replay_randint(low, high, size, *, device=None, dtype=None, generator=None):
        del low, high, generator
        return replay("uniform_int", (size,), device=device, dtype=dtype)

    def replay_randn(*size, device=None, dtype=None, generator=None):
        del generator
        return replay("normal", size, device=device, dtype=dtype)

    with (
        patch.object(package_masking, "rand", replay_rand),
        patch.object(package_masking, "randint", replay_randint),
        patch.object(package_masking, "randn", replay_randn),
    ):
        actual = package_masking.prepare_training_batch(
            package_batch,
            package_columns,
            torch.as_tensor(tasks.numpy(), dtype=torch.long),
            attribute_groups=groups,
            input_dtype=model.input_dtype,
        )

    if draw_index != len(draws):
        raise AssertionError("package masking did not consume every reference draw")
    for name, reference, received in zip(
        ("targets", "modified", "masks"), (targets, modified, masks), actual
    ):
        _compare_mapping(digests, f"training/{name}", reference, received)
    _digest(digests, "training/tasks", tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("crello", "rico"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--batches", type=int, default=1500)
    parser.add_argument(
        "--asset-dir", default=os.environ.get("FLEX_DM_ORIGINAL_ASSET_DIR")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("capture", "compare"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        provenance = _worktree_provenance(dry_run=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {**provenance, "dry_run": True, "package_module_paths": {}}
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if not args.asset_dir:
        parser.error("set --asset-dir or FLEX_DM_ORIGINAL_ASSET_DIR")

    provenance = _worktree_provenance()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase:
        _worker(args)
        return

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": str(args.seed),
            "TF_DETERMINISTIC_OPS": "1",
            "FLEX_DM_ORIGINAL_ASSET_DIR": args.asset_dir,
        }
    )
    started = time.perf_counter()
    for phase in ("capture", "compare"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                args.dataset,
                "--seed",
                str(args.seed),
                "--batches",
                str(args.batches),
                "--asset-dir",
                args.asset_dir,
                "--output-dir",
                str(args.output_dir),
                "--phase",
                phase,
            ],
            check=True,
            env=env,
        )
    elapsed = time.perf_counter() - started
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in provenance.items()):
        raise RuntimeError("git provenance changed during stream capture")
    if _worktree_provenance() != provenance:
        raise RuntimeError("git provenance changed during stream comparison")
    comparison = json.loads(
        (args.output_dir / "comparison.json").read_text(encoding="utf-8")
    )
    package_module_paths = manifest["package_module_paths"]
    for name, path in comparison["package_module_paths"].items():
        if name in package_module_paths and package_module_paths[name] != path:
            raise RuntimeError(f"package module {name} changed its resolved file path")
    package_module_paths.update(comparison["package_module_paths"])
    manifest.update(
        {
            "elapsed_seconds_capture_and_compare": elapsed,
            "batches_per_second": args.batches / elapsed,
            "generation_command": {
                "PYTHONHASHSEED": args.seed,
                "TF_DETERMINISTIC_OPS": 1,
                "CUDA_VISIBLE_DEVICES": "",
                "command": "python models/flex-dm/tests/vendor_parity/capture_flex_dm_training_s4.py",
                "training_entrypoint": "python -m mfp --dataset_name <dataset> --seed <seed>",
            },
            "package_module_paths": package_module_paths,
            "comparison": comparison,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "dataset",
                    "seed",
                    "train_batches",
                    "validation_batches",
                    "elapsed_seconds_capture_and_compare",
                    "batches_per_second",
                    "comparison",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
