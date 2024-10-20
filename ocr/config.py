from typing import List

from pydantic import BaseModel, Field


class OCRConfig(BaseModel):
    """Configuration for OCR processing."""

    output_columns: List[str] = Field(default_factory=list)  # noqa: FA100


ocr_config = OCRConfig(
    output_columns=[
        "index_sort",
        "text",
        "page",
        "block",
        "paragraph",
        "line",
        "word_num",
        "x0",
        "y0",
        "x2",
        "y2",
        "space_type",
        "confidence",
    ],
)
