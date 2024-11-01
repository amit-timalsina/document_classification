from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.calibration import LabelEncoder  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

from common.parsers.default_parser import DefaultParser
from common.parsers.layout_preserving_formatter import LayoutPreservingFormatter
from common.utils.json_to_ocr_text import json_to_ocr_text
from language_model.ocr_dataset import OCRDataset
from logger import logger

if TYPE_CHECKING:
    from pathlib import Path


class SLMDatasetPreparer:
    """Prepares datasets for model training."""

    def __init__(self) -> None:  # noqa: D107
        pass

    @staticmethod
    def process_ocr_data(ocr_json_path: Path) -> tuple[list[str], list[str]]:
        """Process OCR data and return texts and labels."""
        texts: list[str] = []
        labels: list[str] = []
        parser = DefaultParser()
        formatter = LayoutPreservingFormatter()

        for label_folder_path in ocr_json_path.iterdir():
            if label_folder_path.is_dir():
                label = label_folder_path.name
                for file_path in label_folder_path.iterdir():
                    if file_path.is_file():
                        ocr_text = json_to_ocr_text(file_path, parser, formatter)
                        texts.append(ocr_text)
                        labels.append(label)
        return texts, labels

    def prepare_data(
        self,
        ocr_json_path: Path,
        batch_size: int,
        processed_data_path: Path,
    ) -> tuple[DataLoader, DataLoader, LabelEncoder]:
        """Prepare data for model training."""
        if processed_data_path.exists():
            logger.info("Loading preprocessed data...")
            with processed_data_path.open("rb") as f:
                train_dataset, val_dataset, label_encoder = pickle.load(f)  # noqa: S301
            logger.warning("Using pickle to load data. Ensure the source is trusted.")
        else:
            logger.info("Processing OCR data...")
            texts, labels = self.process_ocr_data(ocr_json_path)

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

            x_train, x_val, y_train, y_val = train_test_split(
                texts,
                encoded_labels,
                test_size=0.2,
                random_state=42,
            )

            train_dataset = OCRDataset(x_train, y_train)
            val_dataset = OCRDataset(x_val, y_val)

            logger.info("Saving preprocessed data...")
            with processed_data_path.open("wb") as f:
                pickle.dump((train_dataset, val_dataset, label_encoder), f)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader, label_encoder
