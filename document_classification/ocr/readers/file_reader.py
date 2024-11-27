from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from document_classification.ocr.readers.image_reader import ImageReader
from document_classification.ocr.readers.pdf_reader import PdfReader

if TYPE_CHECKING:
    import numpy as np


class FileReader:
    """Facilitates reading from multiple file types including images and PDFs."""

    @staticmethod
    def read_file_from_path(file_path: str) -> list[np.ndarray]:
        """Read and process files based on their type (image or PDF)."""
        extension = Path(file_path).suffix.lower()
        if extension in [".jpg", ".jpeg", ".png"]:
            return [ImageReader.read_image_from_path(file_path)]
        if extension == ".pdf":
            return PdfReader.convert_pdf_to_images_from_path(file_path)
        msg = f"Unsupported file format: {extension}"
        raise ValueError(msg)
