from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from document_classification.fasttext_model.evaluate.evaluate import evaluate
from document_classification.fasttext_model.predictor.predictor import FasttextPredictor
from document_classification.fasttext_model.text_preprocessor import TextPreprocessor
from document_classification.fasttext_model.train.create_dataset import (
    FasttextDatasetPreparer,
    OcrTextPreparer,
)
from document_classification.fasttext_model.train.train import train_fasttext
from document_classification.logger import logger
from document_classification.ocr.providers.closed_source.google_vision.ocr import GoogleVisionOCR

if TYPE_CHECKING:
    from document_classification.ocr.base import OCRProvider

app = typer.Typer()


@dataclass
class Config:
    """Configuration class for FastText model training and inference."""

    files_directory: Path
    ocr_json_directory: Path
    fasttext_dataset_directory: Path
    output_model_path: Path
    metrics_path: Path


def get_ocr_provider() -> OCRProvider:
    """
    Get the OCR provider instance for text extraction.

    Returns:
        OCRProvider: An instance of the selected OCR provider.

    """
    return GoogleVisionOCR()


@app.command()
def train_and_evaluate(
    files_directory: Path | None = None,
    ocr_json_directory: Path | None = None,
    fasttext_dataset_directory: Path | None = None,
    output_model_path: Path | None = None,
    metrics_path: Path | None = None,
) -> None:
    """
    Train and evaluate the FastText model.

    Args:
        files_directory: Directory containing input files. Defaults to None.
        ocr_json_directory: Directory for OCR JSON outputs. Defaults to None.
        fasttext_dataset_directory: Directory for FastText dataset. Defaults to None.
        output_model_path: Path to save trained model. Defaults to None.
        metrics_path: Path to save evaluation metrics. Defaults to None.

    """
    config = Config(
        files_directory=Path("files"),
        ocr_json_directory=Path("ocr_jsons_tesseract"),
        fasttext_dataset_directory=Path("fasttext_model/dataset"),
        output_model_path=Path("fasttext_model/model.bin"),
        metrics_path=Path("fasttext_model/metrics.json"),
    )
    files_directory = files_directory or config.files_directory
    ocr_json_directory = ocr_json_directory or config.ocr_json_directory
    fasttext_dataset_directory = fasttext_dataset_directory or config.fasttext_dataset_directory
    output_model_path = output_model_path or config.output_model_path
    metrics_path = metrics_path or config.metrics_path

    ocr_provider = get_ocr_provider()
    dataset_preparer = FasttextDatasetPreparer(
        ocr_text_preparer=OcrTextPreparer(TextPreprocessor(), ocr_provider=ocr_provider),
    )
    dataset_preparer.create_dataset(
        files_directory=files_directory,
        ocr_text_directory=ocr_json_directory,
        output_folder_path=fasttext_dataset_directory,
    )
    train_fasttext(dataset_dir_path=fasttext_dataset_directory, output_model_path=output_model_path)
    evaluate(
        dataset_dir_path=fasttext_dataset_directory,
        model_path=output_model_path,
        save_metrices_path=metrics_path,
    )


@app.command()
def inference(
    file_path: Path,
    model_path: Path,
) -> None:
    """
    Perform inference on a single file using the trained FastText model.

    Args:
        file_path: Path to the input file for prediction.
        model_path: Path to the trained model. Defaults to None.

    """
    ocr_provider = get_ocr_provider()
    predictor = FasttextPredictor(model_path=model_path, preprocessor=TextPreprocessor())
    prediction = predictor.predict_from_file(
        file_path=file_path,
        ocr_provider=ocr_provider,
    )
    logger.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    app()
