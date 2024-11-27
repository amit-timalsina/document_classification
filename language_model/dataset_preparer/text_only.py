from __future__ import annotations

from typing import TYPE_CHECKING

from common.utils.file_utils import json_to_dataframe, load_json_file
from language_model.dataset_preparer.base import BaseDatasetPreparer
from language_model.schemas.ocr_dataset import OCRDataset

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import Dataset


class TextOnlyDatasetPreparer(BaseDatasetPreparer):
    def process_data(self, ocr_json_path: Path) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        labels: list[str] = []

        for label_folder_path in ocr_json_path.iterdir():
            if label_folder_path.is_dir():
                label = label_folder_path.name
                for file_path in label_folder_path.iterdir():
                    if file_path.is_file():
                        json_data = load_json_file(file_path)
                        ocr_df = json_to_dataframe(json_data)
                        ocr_text = " ".join(ocr_df["text"])
                        texts.append(ocr_text)
                        labels.append(label)
        return texts, labels

    def create_dataset(self, data: list[str], labels: list[int]) -> Dataset:
        return OCRDataset(data, labels)
