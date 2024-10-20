from common.exceptions import BaseError


class OcrProcessingError(BaseError):
    """An exception raised when an OCR processing error occurs."""

    status_code = 500
    detail = "An internal error occurred during OCR processing."


class InvalidImageError(BaseError):
    """An exception raised when an the image format is not supported.."""

    status_code = 500
    detail = "The image format is not supported. Please provide a valid image file."


class ImageNotFoundError(BaseError):
    """An exception raised when an image is not found."""

    status_code = 404
    detail = "The image was not found. Please provide a valid image file."
