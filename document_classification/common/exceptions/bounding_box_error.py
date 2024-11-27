from __future__ import annotations

from document_classification.common.exceptions.base_error import BaseError


class BoundingBoxError(BaseError):
    """Custom exception for errors in BoundingBox validation."""
