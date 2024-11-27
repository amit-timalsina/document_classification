from __future__ import annotations

import io
from base64 import b64decode
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from google.cloud import vision
from PIL import Image

from document_classification.common.utils.pdf_images import PdfImages


class GoogleVisionImageProcessor:
    """
    A class for processing images using Google Vision API.

    This class provides methods to handle various image formats and prepare them
    for use with Google Vision API services.
    """

    @staticmethod
    def construct_vision_image(
        image: str | bytes | np.ndarray | Path | Image.Image,
    ) -> vision.Image:
        """Construct Google Vision Image object from various types of input."""
        if isinstance(image, bytes):
            return vision.Image(content=image)

        if isinstance(image, str):
            return GoogleVisionImageProcessor._process_string_input(image)

        if isinstance(image, np.ndarray):
            return GoogleVisionImageProcessor._process_numpy_array(image)

        if isinstance(image, (Path, str)):
            return GoogleVisionImageProcessor._process_file_path(image)
        if isinstance(image, Image.Image):
            return GoogleVisionImageProcessor._process_pil_image(image)

        msg = "Invalid image"
        raise ValueError(msg)

    @staticmethod
    def _process_string_input(image: str) -> vision.Image:
        try:
            if image.endswith("=") and image.find(" ") == -1:
                content = b64decode(bytes(image, encoding="utf8"))
                return vision.Image(content=content)
        except AttributeError:
            pass

        # check if url
        url = urlparse(image)
        if url.scheme and url.netloc:
            image_obj = vision.Image()
            image_obj.source.image_uri = image
            return image_obj
        return GoogleVisionImageProcessor._process_file_path(image)

    @staticmethod
    def _process_numpy_array(image: np.ndarray) -> vision.Image:
        content = cv2.imencode(".jpg", image)[1].tobytes()
        return vision.Image(content=content)

    @staticmethod
    def _process_file_path(image: str | Path) -> vision.Image:
        ext = Path(image).suffix.lower()
        if ext == ".pdf":
            pdf_image = PdfImages(str(image))[0]
            content = cv2.imencode(".jpg", pdf_image)[1].tobytes()
            return vision.Image(content=content)
        if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"]:
            content = open(image, "rb").read()  # noqa: PTH123, SIM115
            return vision.Image(content=content)
        msg = "Unsupported file type"
        raise ValueError(msg)

    @staticmethod
    def _process_pil_image(image: Image.Image) -> vision.Image:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return vision.Image(content=buffer.getvalue())
