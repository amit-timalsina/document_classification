from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.parsers.default_parser import DefaultParser
from common.parsers.layout_preserving_formatter import LayoutPreservingFormatter
from common.parsers.parse_and_format import parse_and_format
from common.utils.file_utils import json_to_dataframe, load_json_file

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from llm.classifier import LLMClassifier


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
