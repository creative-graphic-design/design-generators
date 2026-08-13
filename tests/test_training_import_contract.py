"""Core-only import contracts for training-enabled model packages."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class TrainingImportContract:
    """Model-specific values for the core-only import contract."""

    package_name: str
    training_module: str
    eager_exports: tuple[str, ...]
    lightning_exports: tuple[str, ...]
    optional_roots: tuple[str, ...]
    training_leaf_modules: tuple[str, ...]


CONTRACTS = (
    TrainingImportContract(
        package_name="cgb_dm",
        training_module="cgb_dm.training",
        eager_exports=("CGBDMSeedMode",),
        lightning_exports=("CGBDMDataModule", "CGBDMTrainingModule"),
        optional_roots=("lightning", "torchmetrics", "h5py", "h5pickle"),
        training_leaf_modules=(
            "cgb_dm.training.datamodule",
            "cgb_dm.training.lightning_module",
        ),
    ),
    TrainingImportContract(
        package_name="dlt",
        training_module="dlt.training",
        eager_exports=("DLTSeedMode",),
        lightning_exports=(
            "DLTDataModule",
            "DLTReferenceEpochSamplingCallback",
            "DLTTrainingModule",
            "DLTWarmupCosineSchedulerFactory",
        ),
        optional_roots=("lightning", "h5py"),
        training_leaf_modules=(
            "dlt.training.callbacks",
            "dlt.training.datamodule",
            "dlt.training.lightning_module",
        ),
    ),
    TrainingImportContract(
        package_name="layout_dm",
        training_module="layout_dm.training",
        eager_exports=(
            "LayoutDMDataset",
            "LayoutDMProcessedDataset",
            "LayoutDMSeedMode",
            "LayoutDMSyntheticDataset",
            "LayoutDMTimeSampler",
            "LayoutDMTrainingDatasetName",
            "LayoutDMTrainingDatasetSource",
            "LayoutDMTrainingScheduler",
            "LayoutDMTrainingSplit",
        ),
        lightning_exports=("LayoutDMDataModule", "LayoutDMTrainingModule"),
        optional_roots=("lightning", "datasets"),
        training_leaf_modules=(
            "layout_dm.training.datamodule",
            "layout_dm.training.lightning_module",
        ),
    ),
    TrainingImportContract(
        package_name="layout_flow",
        training_module="layout_flow.training",
        eager_exports=(
            "LayoutFlowConditionPolicy",
            "LayoutFlowH5Dataset",
            "LayoutFlowSeedMode",
            "LayoutFlowTrainingDatasetName",
            "LayoutFlowTrainingScheduler",
            "LayoutFlowTrainingSplit",
            "collate_layout_flow_batch",
        ),
        lightning_exports=("LayoutFlowDataModule", "LayoutFlowTrainingModule"),
        optional_roots=("lightning", "h5pickle"),
        training_leaf_modules=(
            "layout_flow.training.datamodule",
            "layout_flow.training.lightning_module",
        ),
    ),
    TrainingImportContract(
        package_name="layoutdiffusion",
        training_module="layoutdiffusion.training",
        eager_exports=(
            "LayoutDiffusionDataset",
            "LayoutDiffusionProcessedDataset",
            "LayoutDiffusionSeedMode",
            "LayoutDiffusionSyntheticDataset",
            "LayoutDiffusionTimeSampler",
            "LayoutDiffusionTrainingDatasetName",
            "LayoutDiffusionTrainingDatasetSource",
            "LayoutDiffusionTrainingScheduler",
            "LayoutDiffusionTrainingSplit",
            "LayoutDiffusionTrainingTransform",
        ),
        lightning_exports=(
            "LayoutDiffusionDataModule",
            "LayoutDiffusionTrainingModule",
        ),
        optional_roots=("lightning", "datasets"),
        training_leaf_modules=(
            "layoutdiffusion.training.datamodule",
            "layoutdiffusion.training.lightning_module",
        ),
    ),
)


def _assert_training_import_contract(contract: TrainingImportContract) -> None:
    """Check inference-only package and core-only training imports in isolation."""
    script = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        _real_find_spec = importlib.util.find_spec

        def _find_spec(name, *args, **kwargs):
            if name == "lightning":
                return None
            return _real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = _find_spec

        import {contract.package_name}

        assert "{contract.training_module}" not in sys.modules
        optional_roots = {contract.optional_roots!r}
        assert not any(
            module == root or module.startswith(root + ".")
            for module in sys.modules
            for root in optional_roots
        )

        import {contract.training_module} as training

        for name in {contract.eager_exports!r}:
            assert name in training.__dict__
        assert "__all__" not in training.__dict__
        for name in {contract.lightning_exports!r}:
            assert name not in training.__dict__
        assert "__getattr__" not in training.__dict__
        assert "__dir__" not in training.__dict__
        for module in {contract.training_leaf_modules!r}:
            assert module not in sys.modules
        assert not any(
            module == root or module.startswith(root + ".")
            for module in sys.modules
            for root in optional_roots
        )
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda value: value.package_name)
def test_training_import_contract(contract: TrainingImportContract) -> None:
    """Keep model imports inference-only and training namespaces core-only."""
    _assert_training_import_contract(contract)
