from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset

from document_classification.language_model.dataset_preparer.custom_collate_function import (
    custom_collate_fn,
)
from logger import logger

if TYPE_CHECKING:
    from pathlib import Path


class BaseDatasetPreparer(ABC):
    """Json -> DataLoader."""

    @abstractmethod
    def process_data(self, ocr_json_path: Path) -> tuple[list[Any], list[str]]:
        """Load data from json and return a tuple of data (text, bboxes) and labels."""

    @abstractmethod
    def create_dataset(self, data: Any, labels: list[int]) -> Dataset:  # noqa: ANN401
        """Return a PyTorch Dataset object."""

    def prepare_data(
        self,
        ocr_json_path: Path,
        batch_size: int,
        processed_data_path: Path,
    ) -> tuple[DataLoader, DataLoader, LabelEncoder]:
        """Prepare data for training and validation."""
        if processed_data_path.exists():
            logger.info("Loading preprocessed data...")
            with processed_data_path.open("rb") as f:
                train_dataset, val_dataset, label_encoder = pickle.load(f)  # noqa: S301
            logger.warning("Using pickle to load data. Ensure the source is trusted.")
        else:
            logger.info("Processing OCR data...")
            data, labels = self.process_data(ocr_json_path)

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

            train_data, val_data, train_labels, val_labels = train_test_split(
                data,
                encoded_labels,
                test_size=0.2,
                random_state=42,
            )

            train_dataset = self.create_dataset(train_data, train_labels)
            val_dataset = self.create_dataset(val_data, val_labels)
            logger.info("Saving preprocessed data...")
            with processed_data_path.open("wb") as f:
                pickle.dump((train_dataset, val_dataset, label_encoder), f)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
        )

        return train_dataloader, val_dataloader, label_encoder
