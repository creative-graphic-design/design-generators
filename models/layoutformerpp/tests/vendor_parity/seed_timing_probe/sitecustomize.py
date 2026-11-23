from __future__ import annotations

import builtins
import functools
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

arm = os.environ["PROBE_ARM"]
record: dict[str, object] = {"seed": int(os.environ["LAYOUTFORMERPP_TRAIN_SEED"])}
record_path = Path(os.environ["PROBE_RECORD_PATH"])
stage = "before_model"
patched: set[str] = set()


def digests() -> dict[str, str]:
    state = np.random.get_state()
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(
            state[0].encode() + state[1].tobytes() + repr(state[2:]).encode()
        ).hexdigest(),
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def persist() -> None:
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def snapshot(name: str) -> None:
    record[f"{name}_rng_sha256"] = digests()
    persist()


def weights_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class StopAfterMeasurement(BaseException):
    pass


def patch_package_module() -> None:
    module_name = "layoutformerpp.training.lightning_module"
    module = sys.modules.get(module_name)
    if module is None or module_name in patched:
        return
    patched.add(module_name)
    original_init = module.LayoutFormerPPTrainingModule.__init__

    @functools.wraps(original_init)
    def measured_init(self, *args, **kwargs):
        snapshot("pre_model")
        original_init(self, *args, **kwargs)
        record["initial_weights_sha256"] = weights_digest(self.model)
        persist()
        globals()["stage"] = "after_model"
        if os.environ.get("PROBE_MODEL_ONLY") == "True":
            os._exit(0)

    module.LayoutFormerPPTrainingModule.__init__ = measured_init


def patch_loader() -> None:
    loader = torch.utils.data.DataLoader
    if getattr(loader, "_seed_timing_probe", False):
        return
    original_init, original_iter = loader.__init__, loader.__iter__

    def measured_init(self, *args, **kwargs):
        training = kwargs.get("shuffle") is True
        if training:
            snapshot("pre_loader")
        original_init(self, *args, **kwargs)
        self._seed_timing_probe_training = training

    def measured_iter(self):
        if getattr(self, "_seed_timing_probe_training", False):
            batch = next(original_iter(self))
            key = "name" if arm == "original" else "names"
            record["first_loader_sample_ids"] = list(batch[key])
            persist()
            if arm == "original":
                raise StopAfterMeasurement
            os._exit(0)
        return original_iter(self)

    loader.__init__ = measured_init
    loader.__iter__ = measured_iter
    loader._seed_timing_probe = True


def patch_vendor_task_utils() -> None:
    module = sys.modules.get("tasks.task_utils")
    if module is None or "tasks.task_utils" in patched:
        return
    patched.add("tasks.task_utils")
    original_build = module.build_model

    def measured_build(*args, **kwargs):
        snapshot("pre_model")
        model = original_build(*args, **kwargs)
        globals()["stage"] = "after_model"
        return model

    module.build_model = measured_build


original_import = builtins.__import__


def instrument_import(name, globals=None, locals=None, fromlist=(), level=0):
    imported = original_import(name, globals, locals, fromlist, level)
    if arm == "package" and name.startswith("layoutformerpp.training"):
        patch_package_module()
    if arm == "original":
        if name == "tasks.task_utils":
            patch_vendor_task_utils()
    return imported


builtins.__import__ = instrument_import
patch_loader()
