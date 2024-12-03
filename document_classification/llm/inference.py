from __future__ import annotations

from typing import TYPE_CHECKING, Any

from document_classification.common.parsers.default_parser import DefaultParser
from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.utils.file_utils import json_to_dataframe, load_json_file
from document_classification.common.utils.parse_and_format import parse_and_format

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from document_classification.llm.classifier import LLMClassifier


async def run_inference(
    classifier: LLMClassifier,
    file_path: Path,
    classification_model: type[BaseModel],
) -> dict[str, Any]:
    """Run inference on a single file."""
    json_data = load_json_file(file_path)
    ocr_df = json_to_dataframe(json_data)

    parser = DefaultParser()
    formatter = LayoutPreservingFormatter()
    ocr_text = parse_and_format(ocr_df, parser, formatter)
    results = await classifier.classify_documents([ocr_text], classification_model)
    return results[0]
