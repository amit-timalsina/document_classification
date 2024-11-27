from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.common.parsers.parse_and_format import parse_and_format
from document_classification.common.utils.file_utils import json_to_dataframe, load_json_file

if TYPE_CHECKING:
    from pathlib import Path

    from document_classification.common.parsers.default_parser import DefaultParser
    from document_classification.common.parsers.layout_preserving_formatter import (
        LayoutPreservingFormatter,
    )


def json_to_ocr_text(
    file_path: Path,
    parser: DefaultParser,
    formatter: LayoutPreservingFormatter,
) -> str:
    """Process a JSON file and return the OCR text."""
    json_data = load_json_file(file_path)
    ocr_df = json_to_dataframe(json_data)
    return parse_and_format(ocr_df, parser, formatter)
