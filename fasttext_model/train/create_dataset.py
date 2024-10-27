from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fasttext_model.text_preprocessor import TextPreprocessor


class OcrTextPreparer:
    """Loads and preprocesses text files."""

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        """Initialize the TextLoader with a TextPreprocessor."""
        self.preprocessor = preprocessor

    def load_and_preprocess(self, folder_path: str) -> list[str]:
        """Load and preprocess text files from a folder."""
        preprocessed_texts = []
        for file_path in Path(folder_path).rglob("*.json"):
            if file_path.is_file():
                with file_path.open(encoding="utf-8") as file:
                    data = json.load(file)
                    if data:
                        data = sorted(data, key=lambda x: x["index_sort"])
                        text = " ".join(i["text"] for i in data)
                        preprocessed_texts.append(self.preprocessor.preprocess_text(text))
        return preprocessed_texts


class DatasetPreparer:
    """Prepares datasets for model training."""

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        """Initialize DatasetPreparer."""
        self.text_loader = OcrTextPreparer(preprocessor)

    @staticmethod
    def split_data(
        data: list[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[list[str], list[str], list[str]]:
        """Split data into training, validation, and test sets."""
        random.shuffle(data)
        total = len(data)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        return data[:train_end], data[train_end:val_end], data[val_end:]

    @staticmethod
    def save_split_data(
        train_data: list[str],
        val_data: list[str],
        test_data: list[str],
        output_folder_path: str,
    ) -> None:
        """Save split data to files in the specified folder path."""
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        paths = [("train.txt", train_data), ("validation.txt", val_data), ("test.txt", test_data)]

        for filename, dataset in paths:
            file_path = Path(output_folder_path) / filename
            with file_path.open("w", encoding="utf-8") as file:
                for line in dataset:
                    file.write(line + "\n")

    def create_dataset(self, data_folder_path: str, output_folder_path: str) -> None:
        """Create and save datasets for fastText from given data folder."""
        all_train_data, all_val_data, all_test_data = [], [], []

        for label_folder_path in Path(data_folder_path).iterdir():
            if label_folder_path.is_dir():
                files = self.text_loader.load_and_preprocess(str(label_folder_path))
                labeled_data = [f"__label__{label_folder_path.name} {text}" for text in files]
                train_data, val_data, test_data = self.split_data(labeled_data)

                all_train_data.extend(train_data)
                all_val_data.extend(val_data)
                all_test_data.extend(test_data)

        self.save_split_data(all_train_data, all_val_data, all_test_data, output_folder_path)
