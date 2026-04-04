from pathlib import Path
import os

from torch.utils.data import DataLoader
from timm.data import ImageDataset
from clearml import Dataset as ClearMLDataset
import lightning as pl


class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path_or_id: str,  # Path to dataset or ClearML dataset ID or ClearML dataset name in the form "project_name:dataset_name"
        train_transforms,
        val_transforms,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = (
            num_workers
            if num_workers > 0 and num_workers <= os.cpu_count()
            else os.cpu_count()
        )
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.dataset_path_or_id = Path(dataset_path_or_id)

        self.train_dataset: ImageDataset = None
        self.val_dataset: ImageDataset = None
        self.test_dataset: ImageDataset = None

    @property
    def class_labels(self) -> None | dict[str, int]:
        return self.train_dataset.reader.class_to_idx if self.train_dataset else None

    def prepare_data(self):
        if not Path(self.dataset_path_or_id).exists():
            try:
                if ":" in str(self.dataset_path_or_id):
                    project_name = str(self.dataset_path_or_id).split(":", maxsplit=1)[
                        0
                    ]
                    dataset_name = str(self.dataset_path_or_id).split(":")[1]
                    self.dataset_path_or_id = Path(
                        ClearMLDataset.get(
                            dataset_project=project_name, dataset_name=dataset_name
                        ).get_local_copy()
                    )
                else:
                    self.dataset_path_or_id = Path(
                        ClearMLDataset.get(
                            dataset_id=str(self.dataset_path_or_id)
                        ).get_local_copy()
                    )
            except Exception as exc:
                raise ValueError(
                    f"{self.dataset_path_or_id} does not exist and is not a valid ClearML dataset ID"
                ) from exc

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = ImageDataset(
                self.dataset_path_or_id / "train", transform=self.train_transforms
            )
            self.val_dataset = ImageDataset(
                self.dataset_path_or_id / "val", transform=self.val_transforms
            )
        if stage == "test":
            self.test_dataset = ImageDataset(
                self.dataset_path_or_id / "test", transform=self.val_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
