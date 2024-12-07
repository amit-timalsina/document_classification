from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.ocr.readers.image_reader import ImageReader
from document_classification.ocr.readers.pdf_reader import PdfReader

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class FileReader:
    """Facilitates reading from multiple file types including images and PDFs."""

    @staticmethod
    def read_file_from_path(file_path: Path) -> list[np.ndarray]:
        """
        Read and process files based on their type (image or PDF).

        Args:
            file_path: The path to the file to be processed.

        Returns:
            list[np.ndarray]: A list of processed images.

        Raises:
            ValueError: If the file format is unsupported.

        """
        extension = file_path.suffix.lower()
        if extension in [".jpg", ".jpeg", ".png"]:
            return [ImageReader.read_image_from_path(file_path)]
        if extension == ".pdf":
            return PdfReader.convert_pdf_to_images_from_path(file_path)
        msg = f"Unsupported file format: {extension}"
        raise ValueError(msg)
