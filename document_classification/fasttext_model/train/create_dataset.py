from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from document_classification.ocr.readers.file_reader import FileReader
from logger import logger

if TYPE_CHECKING:
    from document_classification.fasttext_model.text_preprocessor import TextPreprocessor
    from document_classification.ocr.base import OCRProvider


class OcrTextPreparer:
    """Executes OCR and preprocesses text files."""

    def __init__(self, preprocessor: TextPreprocessor, ocr_provider: OCRProvider) -> None:
        """Initialize the OcrTextPreparer with a TextPreprocessor and OCR provider."""
        self.preprocessor = preprocessor
        self.ocr_provider = ocr_provider

    def execute_ocr(self, files_directory: Path, ocr_text_directory: Path) -> None:
        """Execute OCR on the given files and save the results."""
        ocr_text_directory.mkdir(parents=True, exist_ok=True)

        for doc_type in files_directory.iterdir():
            if doc_type.is_dir():
                for file_path in doc_type.iterdir():
                    if file_path.is_file():
                        try:
                            logger.debug(f"Processing file: {file_path}")
                            ocr_text_path = ocr_text_directory / doc_type.name
                            ocr_text_path.mkdir(parents=True, exist_ok=True)

                            output_file_name = f"{file_path.stem}.json"
                            ocr_text_file_path = ocr_text_path / output_file_name

                            if ocr_text_file_path.exists():
                                logger.debug(f"OCR results already exist for file: {file_path}")
                                continue

                            images = FileReader.read_file_from_path(str(file_path))

                            ocr_results: list[dict] = []
                            for image in images:
                                ocr_json = self.ocr_provider.perform_ocr(image).ocr_dict
                                ocr_results.extend(ocr_json)

                            with ocr_text_file_path.open("w", encoding="utf-8") as file:
                                json.dump(ocr_results, file, ensure_ascii=False, indent=2)

                        except Exception as e:  # noqa: BLE001
                            logger.exception(f"Error processing file {file_path}: {e}")

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


class FasttextDatasetPreparer:
    """Prepares datasets for model training."""

    def __init__(self, ocr_text_preparer: OcrTextPreparer) -> None:
        """Initialize DatasetPreparer."""
        self.text_preparer = ocr_text_preparer

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
        output_folder_path: Path,
    ) -> None:
        """Save split data to files in the specified folder path."""
        output_folder_path.mkdir(parents=True, exist_ok=True)
        paths = [("train.txt", train_data), ("validation.txt", val_data), ("test.txt", test_data)]

        for filename, dataset in paths:
            file_path = output_folder_path / filename
            with file_path.open("w", encoding="utf-8") as file:
                for line in dataset:
                    file.write(line + "\n")

    def create_dataset(
        self,
        files_directory: Path,
        ocr_text_directory: Path,
        output_folder_path: Path,
    ) -> None:
        """Create and save datasets for fastText from given data folder."""
        self.text_preparer.execute_ocr(files_directory, ocr_text_directory)

        all_train_data, all_val_data, all_test_data = [], [], []

        for label_folder_path in ocr_text_directory.iterdir():
            if label_folder_path.is_dir():
                files = self.text_preparer.load_and_preprocess(str(label_folder_path))
                labeled_data = [f"__label__{label_folder_path.name} {text}" for text in files]
                train_data, val_data, test_data = self.split_data(labeled_data)

                all_train_data.extend(train_data)
                all_val_data.extend(val_data)
                all_test_data.extend(test_data)

        self.save_split_data(all_train_data, all_val_data, all_test_data, output_folder_path)
