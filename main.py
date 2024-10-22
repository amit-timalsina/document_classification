# TODO(Amit): Remove this file from the package.

import json
from pathlib import Path

from fasttext_model.evaluate.evaluate import evaluate
from logger import logger
from ocr.providers.closed_source.google_vision.ocr import GoogleVisionOCR
from ocr.readers.file_reader import FileReader


def execute_ocr(files_directory: Path, ocr_text_directory: Path) -> None:
    """Execute OCR on the given files and save the results."""
    ocr = GoogleVisionOCR()
    ocr_text_directory.mkdir(parents=True, exist_ok=True)

    for doc_type in files_directory.iterdir():
        if doc_type.is_dir():
            for file_path in doc_type.iterdir():
                if file_path.is_file():
                    try:
                        logger.debug(f"Processing file: {file_path}")
                        ocr_text_path = ocr_text_directory / doc_type.name
                        ocr_text_path.mkdir(parents=True, exist_ok=True)

                        # Create JSON file for the document OCR results
                        output_file_name = f"{file_path.stem}.json"
                        ocr_text_file_path = ocr_text_path / output_file_name

                        if ocr_text_file_path.exists():
                            logger.debug(f"OCR results already exist for file: {file_path}")
                            continue

                        images = FileReader.read_file_from_path(str(file_path))

                        ocr_results: list[dict] = []
                        for image in images:
                            ocr_json = ocr.perform_ocr(image).ocr_dict
                            ocr_results.extend(ocr_json)

                        # Writing the combined OCR results as a JSON array
                        with ocr_text_file_path.open("w", encoding="utf-8") as file:
                            json.dump(ocr_results, file, ensure_ascii=False, indent=2)

                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    files_directory = Path("files")
    ocr_json_directory = Path("ocr_jsons")
    fasttext_dataset_directory = Path("fasttext_model/dataset")
    output_model_path = Path("fasttext_model/model.bin")
    metrics_path = Path("fasttext_model/metrics.json")
    # execute_ocr(files_directory, ocr_json_directory)
    # create_fasttext_dataset(
    #     data_folder_path=str(ocr_json_directory),
    #     output_folder_path=str(fasttext_dataset_directory),
    # )
    # train_fasttext(dataset_dir_path=fasttext_dataset_directory, output_model_path=output_model_path)
    evaluate(
        dataset_dir_path=fasttext_dataset_directory,
        model_path=output_model_path,
        save_metrices_path=metrics_path,
    )
