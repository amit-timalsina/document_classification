from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fasttext_model.evaluate.evaluate import evaluate
from fasttext_model.inference.infer import Infer
from fasttext_model.text_preprocessor import TextPreprocessor
from fasttext_model.train.create_dataset import DatasetPreparer, OcrTextPreparer
from fasttext_model.train.train import train_fasttext
from logger import logger
from ocr.providers.closed_source.google_vision.ocr import GoogleVisionOCR

if TYPE_CHECKING:
    from ocr.base import OCRProvider

app = typer.Typer()


@dataclass
class Config:
    """Configuration class for FastText model training and inference."""

    files_directory: Path = Path("files")
    ocr_json_directory: Path = Path("ocr_jsons_tesseract")
    fasttext_dataset_directory: Path = Path("fasttext_model/dataset")
    output_model_path: Path = Path("fasttext_model/model.bin")
    metrics_path: Path = Path("fasttext_model/metrics.json")


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
    config = Config()
    files_directory = files_directory or config.files_directory
    ocr_json_directory = ocr_json_directory or config.ocr_json_directory
    fasttext_dataset_directory = fasttext_dataset_directory or config.fasttext_dataset_directory
    output_model_path = output_model_path or config.output_model_path
    metrics_path = metrics_path or config.metrics_path

    ocr_provider = get_ocr_provider()
    dataset_preparer = DatasetPreparer(
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
    model_path: Path | None = None,
) -> None:
    """
    Perform inference on a single file using the trained FastText model.

    Args:
        file_path (Path): Path to the input file for prediction.
        model_path (Path | None, optional): Path to the trained model. Defaults to None.

    """
    config = Config()
    model_path = model_path or config.output_model_path

    ocr_provider = get_ocr_provider()
    infer = Infer(model_path=model_path, preprocessor=TextPreprocessor())
    prediction = infer.predict_from_file(
        file_path=file_path,
        ocr_provider=ocr_provider,
    )
    logger.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    app()
