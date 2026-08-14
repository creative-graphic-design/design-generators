"""Lightning data module for local RADM training inputs."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .config import RADMEffectiveConfig
from .dataset import RADMCOCODataset, RADMDataCollator, RADMTrainingExample

from lightning.pytorch import LightningDataModule


class RADMDataModule(LightningDataModule):
    """Load explicit local train/validation COCO and feature paths."""

    def __init__(
        self,
        *,
        train_annotations: str | Path,
        train_image_root: str | Path,
        train_text_feature_root: str | Path,
        val_annotations: str | Path | None = None,
        val_image_root: str | Path | None = None,
        val_text_feature_root: str | Path | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        allow_missing_text_features: bool = False,
        effective: RADMEffectiveConfig,
    ) -> None:
        """Initialize explicit local train and validation data paths."""
        super().__init__()
        self.train_annotations = Path(train_annotations)
        self.train_image_root = Path(train_image_root)
        self.train_text_feature_root = Path(train_text_feature_root)
        self.val_annotations = Path(val_annotations) if val_annotations else None
        self.val_image_root = Path(val_image_root) if val_image_root else None
        self.val_text_feature_root = (
            Path(val_text_feature_root) if val_text_feature_root else None
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_missing_text_features = allow_missing_text_features
        self.effective = effective
        self.train_dataset: RADMCOCODataset | None = None
        self.val_dataset: RADMCOCODataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested Lightning stage."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset(
                self.train_annotations,
                self.train_image_root,
                self.train_text_feature_root,
            )

            if (
                self.val_annotations is not None
                and self.val_image_root is not None
                and self.val_text_feature_root is not None
            ):
                self.val_dataset = self._dataset(
                    self.val_annotations,
                    self.val_image_root,
                    self.val_text_feature_root,
                )

    def train_dataloader(self) -> DataLoader[RADMTrainingExample]:
        """Return the deterministic worker-zero training loader."""
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None:
            raise RuntimeError("training dataset was not initialized")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[RADMTrainingExample] | None:
        """Return the validation loader when explicit validation paths exist."""
        if self.val_dataset is None:
            self.setup("fit")
        return (
            None
            if self.val_dataset is None
            else self._loader(self.val_dataset, shuffle=False)
        )

    def _dataset(
        self, annotations: Path, image_root: Path, text_root: Path
    ) -> RADMCOCODataset:
        return RADMCOCODataset(
            annotation_path=annotations,
            image_root=image_root,
            text_feature_root=text_root,
            effective=self.effective,
            allow_missing_text_features=self.allow_missing_text_features,
        )

    def _loader(
        self, dataset: RADMCOCODataset, *, shuffle: bool
    ) -> DataLoader[RADMTrainingExample]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=RADMDataCollator(effective=self.effective),
        )
