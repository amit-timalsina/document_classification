from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from common.utils.pdf_images import PdfImages

if TYPE_CHECKING:
    import numpy as np


class PdfReader:
    """Handles reading and converting PDF files to image sequences."""

    @staticmethod
    def convert_pdf_to_images_from_path(pdf_path: str) -> list[np.ndarray]:
        """Convert a PDF file into a list of images, one per page."""
        try:
            pdf_images = PdfImages(pdf_path)
            return [cv2.imread(str(image_path)) for image_path in pdf_images.images]
        except Exception as e:
            msg = f"Failed to process PDF: {e}"
            raise RuntimeError(msg) from e
